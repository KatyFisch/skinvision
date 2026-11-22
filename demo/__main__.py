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
from PIL import Image

from classifiers.single_photo import PhotoClassifier

ROOT = Path(__file__).resolve().parent
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
ASSETS = {
    "/": ("index.html", "text/html; charset=utf-8"),
    "/style.css": ("style.css", "text/css; charset=utf-8"),
    "/app.js": ("app.js", "text/javascript; charset=utf-8"),
}


class DemoHandler(BaseHTTPRequestHandler):
    def __init__(self, *args, classifier: PhotoClassifier, inference_lock: LockType, **kwargs) -> None:
        self.classifier = classifier
        self.inference_lock = inference_lock
        super().__init__(*args, **kwargs)

    def reply(self, status: HTTPStatus, content_type: str, body: bytes) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Security-Policy", "default-src 'self'; img-src 'self' blob:; object-src 'none'; frame-ancestors 'none'")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def reply_json(self, status: HTTPStatus, body: dict[str, str]) -> None:
        self.reply(status, "application/json", json.dumps(body).encode())

    def do_GET(self) -> None:
        asset = ASSETS.get(self.path)
        if asset is None:
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        filename, content_type = asset
        self.reply(HTTPStatus.OK, content_type, (ROOT / filename).read_bytes())

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
                image.load()
                with self.inference_lock:
                    result = self.classifier.predict(image)
        except (OSError, Image.DecompressionBombError):
            self.reply_json(HTTPStatus.BAD_REQUEST, {"error": "This photo could not be opened. Try another JPEG or PNG."})
            return
        except RuntimeError:
            self.reply_json(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "Analysis failed. Try again or restart the demo."})
            return
        self.reply_json(HTTPStatus.OK, result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    data_dir = Path(os.environ.get("MILK10K_DATA_DIR", ROOT.parent / "data")).expanduser()
    parser.add_argument("--model", type=Path, default=data_dir / "models/single-photo/model.json")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    if not args.model.expanduser().is_file():
        parser.error("Pass --model with the model.json from your single-photo evaluation run.")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    classifier = PhotoClassifier(args.model, device)
    handler = partial(DemoHandler, classifier=classifier, inference_lock=Lock())
    with ThreadingHTTPServer(("127.0.0.1", args.port), handler) as server:
        print(f"SkinVision is running at http://127.0.0.1:{server.server_port}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
