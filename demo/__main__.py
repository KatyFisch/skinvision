"""Serve the local SkinVision research demo."""

import argparse
import json
import os
import sys
from _thread import LockType
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from threading import Lock, Thread

import torch
from huggingface_hub.errors import HfHubHTTPError
from PIL import Image, ImageOps

from classifiers.dermfm_zero import WEIGHTS_FILENAME, checkpoint_dir, download
from classifiers.export_padded_crop import PaddedCropTorchModel
from classifiers.lesion_crop import crop_with_skin_margin, frame_photo, skin_margin_box
from classifiers.model import DermFMClassifier
from classifiers.segmentation import LesionSegmenter
from classifiers.single_photo import LesionRouter, PhotoClassifier
from demo.gradcam_heatmap import heatmap_overlay, padded_crop_heatmap
from demo.model_assets import DEFAULT_ROUTER_CUTOFF, ROUTER_MANIFEST, SEGMENTATION_CHECKPOINT, download_checkpoints
from demo.samples import BUNDLED_SAMPLE_DIR, SAMPLE_ASSETS, available_samples
from demo.segmentation import outline_path, overlay_png

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ANALYSIS_STREAM = "application/x-ndjson"
MODEL_STATUS_PATH = "/api/model-status"
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class ModelStatus(StrEnum):
    DOWNLOADING = "downloading"
    ACCESS_REQUIRED = "access_required"
    DOWNLOAD_FAILED = "download_failed"
    LOADING = "loading"
    LOAD_FAILED = "load_failed"
    READY = "ready"


@dataclass(frozen=True)
class DemoModels:
    classifier: PhotoClassifier | PaddedCropTorchModel
    segmenter: LesionSegmenter
    padded_crop: bool
    routing: tuple[LesionRouter, DermFMClassifier] | None = None


@dataclass(frozen=True)
class DemoState:
    status: ModelStatus
    models: DemoModels | None = None


class DemoHandler(BaseHTTPRequestHandler):
    def __init__(
        self, *args, model_state: Callable[[], DemoState], inference_lock: LockType, samples_dir: Path = BUNDLED_SAMPLE_DIR, **kwargs
    ) -> None:
        self.model_state = model_state
        self.inference_lock = inference_lock
        self.samples_dir = samples_dir
        super().__init__(*args, **kwargs)

    def log_request(self, code: int | str = "-", size: int | str = "-") -> None:
        if code != HTTPStatus.OK or self.path != MODEL_STATUS_PATH:
            super().log_request(code, size)

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

    def analyze(self, models: DemoModels, photo: Image.Image, stream: bool) -> None:
        classifier, padded_crop, routing = models.classifier, models.padded_crop, models.routing
        try:
            if stream:
                self.start_reply(HTTPStatus.OK, ANALYSIS_STREAM)
            with self.inference_lock:
                if padded_crop:
                    photo = frame_photo(photo)
                mask = models.segmenter.mask(photo)
                has_mask = mask.getbbox() is not None
                overlay_mask = Image.new("L", photo.size, 255) if padded_crop and not has_mask else mask
                overlay = overlay_png(overlay_mask)
                if stream:
                    self.send_event({"overlay": overlay, "outline": outline_path(mask)})

                result: dict[str, str | float | None] = {"research_prediction": None, "route": None}
                if routing and has_mask and not routing[0].uses_specialist(photo):
                    diagnosis, score = routing[1].predict_other(photo)
                    result.update(research_prediction=diagnosis, diagnosis_score=score, route="OTHER", analysis_input="full_frame")
                elif padded_crop:
                    prepared = crop_with_skin_margin(photo, mask)
                    result.update(classifier.predict(prepared))
                    result["analysis_input"] = "padded_crop" if has_mask else "original_photo"
                    if stream and isinstance(classifier, PaddedCropTorchModel) and (box := skin_margin_box(photo, mask)) is not None:
                        try:
                            heat = padded_crop_heatmap(prepared, classifier, str(result["research_prediction"]))
                            result["heatmap"] = heatmap_overlay(heat, mask, box)
                        except RuntimeError as error:
                            self.log_error("Grad-CAM unavailable: %s", error)
                elif has_mask:
                    masked = Image.composite(photo, Image.new("RGB", photo.size), mask)
                    result.update(classifier.predict(masked))
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
        if self.path == MODEL_STATUS_PATH:
            self.reply_json(HTTPStatus.OK, {"status": self.model_state().status})
            return

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
        state = self.model_state()
        if state.models is None:
            self.reply_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "The analysis model is unavailable.", "status": state.status})
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
            self.analyze(state.models, photo, stream=self.headers.get("Accept") == ANALYSIS_STREAM)
        except (BrokenPipeError, ConnectionResetError):
            pass


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    data_dir = Path(os.environ.get("MILK10K_DATA_DIR", ROOT.parent / "data")).expanduser()
    default_model = ROOT.parent / "models/mel-vs-nv-padded-crop.pth"
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model,
        help="Exported padded-crop heads (.pth), a padded-crop run directory, or a legacy model.json",
    )
    parser.add_argument("--router-model", type=Path, help="Use a local router manifest instead of the pinned demo router")
    parser.add_argument("--other-model", type=Path, help="Use a local direct 11-class run instead of the bundled heads")
    parser.add_argument("--segmentation-model", type=Path, help="Use a local U-Net checkpoint file or extracted folder")
    parser.add_argument("--mel-nv-only", action="store_true", help="Use the padded-crop classifier without routing")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    model = args.model.expanduser()
    run_dir = model.is_dir()
    if run_dir:
        if not all((model / name).is_file() for name in ("metrics.json", "heads.joblib")):
            parser.error("Pass --model with the padded-crop run directory containing metrics.json and heads.joblib.")
    elif not model.is_file():
        parser.error("Pass --model with exported padded-crop heads (.pth), a padded-crop run directory, or a legacy model.json.")
    padded_crop = run_dir or model.suffix == ".pth"
    use_routing = padded_crop and not args.mel_nv_only
    if not use_routing and (args.router_model or args.other_model):
        parser.error("A separate router and other-diagnosis model require a padded-crop --model with routing enabled.")
    router_model = args.router_model
    if router_model is not None:
        router_model = router_model.expanduser()
        if not router_model.is_file():
            parser.error("Pass --router-model with a saved router manifest.")
    local_other = data_dir / "models/direct-11"
    other_default = local_other if local_other.is_dir() else ROOT.parent / "models/direct-11"
    other_model = (args.other_model or other_default).expanduser()
    if use_routing and not all((other_model / name).is_file() for name in ("metrics.json", "heads.joblib")):
        parser.error("Pass --other-model with a direct 11-class directory containing metrics.json and heads.joblib.")

    local_checkpoint = ROOT.parent / "models" / SEGMENTATION_CHECKPOINT
    extracted = local_checkpoint.with_suffix("")
    checkpoint = args.segmentation_model.expanduser() if args.segmentation_model else None
    if checkpoint is not None and not (checkpoint.is_file() or (checkpoint / "data.pkl").is_file()):
        parser.error("Pass --segmentation-model with the U-Net checkpoint or its extracted folder.")
    if checkpoint is None:
        if (extracted / "data.pkl").is_file():
            checkpoint = extracted
        elif local_checkpoint.is_file():
            checkpoint = local_checkpoint

    if checkpoint is None or (use_routing and router_model is None):
        try:
            bundled = download_checkpoints(ROOT.parent, data_dir)
        except (OSError, RuntimeError) as error:
            parser.error(str(error))
        checkpoint = checkpoint or bundled / SEGMENTATION_CHECKPOINT
        router_model = router_model or bundled / ROUTER_MANIFEST
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    assert checkpoint is not None

    def load_models(segmenter_checkpoint: Path) -> DemoModels:
        classifier = PaddedCropTorchModel(model, data_dir, device) if padded_crop else PhotoClassifier(model, device)
        routing = None
        if use_routing and isinstance(classifier, PaddedCropTorchModel):
            assert router_model is not None
            router = LesionRouter(router_model, device, cutoff=DEFAULT_ROUTER_CUTOFF if args.router_model is None else None)
            other = DermFMClassifier(other_model, data_dir, device, expected_task="direct_11", encoder=classifier.encoder)
            routing = (router, other)
        return DemoModels(classifier, LesionSegmenter(segmenter_checkpoint, device), padded_crop, routing)

    needs_download = padded_crop and not (checkpoint_dir(data_dir) / WEIGHTS_FILENAME).is_file()
    state = DemoState(ModelStatus.DOWNLOADING) if needs_download else DemoState(ModelStatus.READY, load_models(checkpoint))

    def download_and_load(segmenter_checkpoint: Path) -> None:
        nonlocal state
        # Publish failures at the worker boundary so the browser never waits on a stopped thread.
        try:
            download(WEIGHTS_FILENAME, data_dir)
        except Exception as error:  # noqa: BLE001
            access_denied = isinstance(error, HfHubHTTPError) and error.response.status_code in {401, 403}
            state = DemoState(ModelStatus.ACCESS_REQUIRED if access_denied else ModelStatus.DOWNLOAD_FAILED)
            print(f"Could not download the DermFM model from Hugging Face ({type(error).__name__}).", file=sys.stderr, flush=True)
            return

        state = DemoState(ModelStatus.LOADING)
        try:
            state = DemoState(ModelStatus.READY, load_models(segmenter_checkpoint))
        except Exception as error:  # noqa: BLE001
            state = DemoState(ModelStatus.LOAD_FAILED)
            print(f"Could not load the demo models ({type(error).__name__}).", file=sys.stderr, flush=True)

    handler = partial(DemoHandler, model_state=lambda: state, inference_lock=Lock())
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"SkinVision is running at http://127.0.0.1:{server.server_port}", flush=True)
        if needs_download:
            Thread(target=download_and_load, args=(checkpoint,), daemon=True).start()
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
