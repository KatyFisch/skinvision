"""Serve the local SkinVision research demo."""

import argparse
import json
import os
from _thread import LockType
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Lock

import torch
from PIL import Image, ImageOps

from classifiers.lesion_crop import crop_with_skin_margin, frame_photo
from classifiers.model import DermFMClassifier
from classifiers.segmentation import LesionSegmenter
from classifiers.single_photo import PhotoClassifier
from demo.samples import SAMPLE_ASSETS, SAMPLE_FOLDER, available_samples
from demo.segmentation import outline_path, overlay_png

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ANALYSIS_STREAM = "application/x-ndjson"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class DemoHandler(BaseHTTPRequestHandler):
    def __init__(
        self,
        *args,
        classifier: PhotoClassifier | DermFMClassifier,
        segmenter: LesionSegmenter,
        inference_lock: LockType,
        padded_crop: bool = False,
        samples_dir: Path = ROOT.parent / "data" / SAMPLE_FOLDER,
        **kwargs,
    ) -> None:
        self.classifier = classifier
        self.segmenter = segmenter
        self.inference_lock = inference_lock
        self.padded_crop = padded_crop
        self.samples_dir = samples_dir
        super().__init__(*args, **kwargs)

    def start_reply(self, status: HTTPStatus, content_type: str, length: int | None = None) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        if length is not None:
            self.send_header("Content-Length", str(length))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()

    def reply(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.start_reply(status, content_type, len(body))
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def reply_json(self, status: HTTPStatus, body: dict[str, str | float | None]) -> None:
        self.reply(status, "application/json", json.dumps(body).encode())

    def send_event(self, body: dict[str, str | float | None]) -> None:
        self.wfile.write((json.dumps(body) + "\n").encode())
        self.wfile.flush()

    def analyze(self, photo: Image.Image, stream: bool) -> None:
        try:
            if stream:
                self.start_reply(HTTPStatus.OK, ANALYSIS_STREAM)
            with self.inference_lock:
                if self.padded_crop:
                    photo = frame_photo(photo)
                mask = self.segmenter.mask(photo)
                has_mask = mask.getbbox() is not None
                overlay_mask = Image.new("L", photo.size, 255) if self.padded_crop and not has_mask else mask
                overlay = overlay_png(overlay_mask)
                if stream:
                    self.send_event({"overlay": overlay, "outline": outline_path(mask)})

                result: dict[str, str | float | None] = {"research_prediction": None, "route": None}
                if self.padded_crop:
                    result.update(self.classifier.predict(crop_with_skin_margin(photo, mask)))
                    result["analysis_input"] = "padded_crop" if has_mask else "original_photo"
                elif has_mask:
                    masked = Image.composite(photo, Image.new("RGB", photo.size), mask)
                    result.update(self.classifier.predict(masked))
            if stream:
                self.send_event(result)
            else:
                self.reply_json(HTTPStatus.OK, {**result, "overlay": overlay})
        except RuntimeError as error:
            self.log_error("Analysis failed: %s", error)
            message: dict[str, str | float | None] = {"error": "Analysis failed. Try again or restart the demo."}
            if stream:
                self.send_event(message)
            else:
                self.reply_json(HTTPStatus.INTERNAL_SERVER_ERROR, message)

    def do_GET(self) -> None:
        if self.path == "/api/samples":
            catalog = json.dumps(available_samples(self.samples_dir)).encode()
            self.reply(HTTPStatus.OK, "application/json", catalog)
            return

        asset = ASSETS.get(self.path)
        if asset is not None:
            filename, content_type = asset
            self.reply(HTTPStatus.OK, content_type, (ROOT / filename).read_bytes())
            return

        sample = SAMPLE_ASSETS.get(self.path)
        if sample is not None:
            try:
                photo = (self.samples_dir / sample.filename).read_bytes()
            except FileNotFoundError:
                pass
            else:
                self.reply(HTTPStatus.OK, "image/jpeg", photo)
                return
        self.send_error(HTTPStatus.NOT_FOUND)

    def do_POST(self) -> None:
        if self.path != "/api/predict":
            self.send_error(HTTPStatus.NOT_FOUND)
            return

        # The browser can send this content type only from the demo's own origin.
        host = self.headers.get("Host", "").split(":", 1)[0]
        if host not in {"127.0.0.1", "localhost"} or self.headers.get("Content-Type") != "application/octet-stream":
            self.reply_json(HTTPStatus.FORBIDDEN, {"error": "Open the demo on localhost and try again."})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            length = 0
        if not 0 < length <= MAX_UPLOAD_BYTES:
            self.reply_json(HTTPStatus.BAD_REQUEST, {"error": "Choose a JPEG or PNG under 20 MB."})
            return

        try:
            with Image.open(BytesIO(self.rfile.read(length)), formats=("JPEG", "PNG")) as image:
                photo = ImageOps.exif_transpose(image).convert("RGB")
        except (OSError, Image.DecompressionBombError):
            self.reply_json(HTTPStatus.BAD_REQUEST, {"error": "This photo could not be opened. Try another JPEG or PNG."})
            return

        try:
            self.analyze(photo, stream=self.headers.get("Accept") == ANALYSIS_STREAM)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    data_dir = Path(os.environ.get("MILK10K_DATA_DIR", ROOT.parent / "data")).expanduser()
    padded_model = data_dir / "models/padded-crop"
    default_model = padded_model if padded_model.is_dir() else data_dir / "models/single-photo/model.json"
    parser.add_argument("--model", type=Path, default=default_model, help="A padded-crop experiment directory or a legacy model.json")
    segmentation_model = ROOT.parent / "models/unet_segmentation_full"
    if not segmentation_model.is_dir():
        segmentation_model = segmentation_model.with_suffix(".pth")
    parser.add_argument(
        "--segmentation-model", type=Path, default=segmentation_model, help="Segmentation checkpoint file or extracted folder"
    )
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    model = args.model.expanduser()
    padded_crop = model.is_dir()
    if padded_crop:
        if not all((model / name).is_file() for name in ("metrics.json", "heads.joblib")):
            parser.error("Pass --model with the padded-crop run directory containing metrics.json and heads.joblib.")
    elif not model.is_file():
        parser.error("Pass --model with a padded-crop run directory or a legacy model.json.")
    checkpoint = args.segmentation_model.expanduser()
    if not (checkpoint.is_file() or (checkpoint / "data.pkl").is_file()):
        parser.error("Pass --segmentation-model with the U-Net checkpoint or its extracted folder.")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    classifier = DermFMClassifier(model, data_dir, device, expected_task="padded_crop") if padded_crop else PhotoClassifier(model, device)
    segmenter = LesionSegmenter(checkpoint, device)
    handler = partial(
        DemoHandler,
        classifier=classifier,
        segmenter=segmenter,
        inference_lock=Lock(),
        padded_crop=padded_crop,
        samples_dir=data_dir / SAMPLE_FOLDER,
    )
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"SkinVision is running at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
