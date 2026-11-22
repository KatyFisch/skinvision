import json
from base64 import b64decode
from functools import partial
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer
from io import BytesIO
from threading import Event, Lock, Thread
from unittest import TestCase
from unittest.mock import Mock

from PIL import Image, ImageChops

from classifiers.lesion_crop import crop_with_skin_margin
from demo.__main__ import ANALYSIS_STREAM, DemoHandler


class DemoSegmentationTest(TestCase):
    def setUp(self) -> None:
        self.photo = Image.new("RGB", (600, 450), (190, 140, 110))
        self.mask = Image.new("L", self.photo.size)
        self.classifier = Mock()
        segmenter = Mock()
        segmenter.mask.return_value = self.mask
        self.segmenter = segmenter

    def start_client(self, *, padded_crop: bool = False, routing: tuple[Mock, Mock] | None = None) -> None:
        handler = partial(
            DemoHandler,
            classifier=self.classifier,
            segmenter=self.segmenter,
            inference_lock=Lock(),
            padded_crop=padded_crop,
            routing=routing,
        )
        server = self.enterContext(ThreadingHTTPServer(("127.0.0.1", 0), handler))
        Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.addCleanup(server.shutdown)
        self.client = HTTPConnection("127.0.0.1", server.server_port, timeout=5)
        self.addCleanup(self.client.close)

    def send_photo(self, accept: str) -> None:
        with BytesIO() as buffer:
            self.photo.save(buffer, format="PNG")
            self.client.request("POST", "/api/predict", buffer.getvalue(), {"Content-Type": "application/octet-stream", "Accept": accept})

    def test_gallery_serves_all_bundled_photos(self) -> None:
        self.start_client()
        self.client.request("GET", "/api/samples")
        with self.client.getresponse() as response:
            self.assertEqual(response.status, 200)
            samples = json.load(response)
        self.assertEqual(len(samples), 7)
        self.assertIn("Basal cell carcinoma", {sample["title"] for sample in samples})
        self.assertIn("Seborrheic keratosis", {sample["title"] for sample in samples})
        for sample in samples:
            with self.subTest(url=sample["url"]):
                self.assertTrue(sample["credit"] and sample["license"] and sample["source"])
                self.client.request("GET", sample["url"])
                with self.client.getresponse() as response:
                    self.assertEqual(response.status, 200)
                    self.assertEqual(response.getheader("Content-Type"), "image/jpeg")
                    with Image.open(BytesIO(response.read())) as photo:
                        self.assertEqual(photo.size, self.photo.size)
                        self.assertFalse(photo.getexif())

    def test_streams_outline_before_returning_prediction_on_the_same_mask(self) -> None:
        self.start_client()
        self.mask.paste(255, (200, 120, 390, 290))
        self.mask.paste(255, (0, 20, 18, 40))
        release_classifier = Event()
        self.addCleanup(release_classifier.set)

        def predict(_image: Image.Image) -> dict[str, str]:
            if not release_classifier.wait(timeout=5):
                raise RuntimeError("Client did not receive the outline before classification finished")
            return {"research_prediction": "NV", "route": "MEL_OR_NV"}

        self.classifier.predict.side_effect = predict
        self.send_photo(ANALYSIS_STREAM)

        with self.client.getresponse() as response:
            self.assertEqual(response.getheader("Content-Type"), ANALYSIS_STREAM)
            event = json.loads(response.readline())
            outlines = event["outline"].removesuffix("Z").split("Z")
            self.assertEqual(len(outlines), 2)
            bounds = []
            for outline in outlines:
                self.assertTrue(outline.startswith("M"))
                xs, ys = zip(*(map(float, point.split(",")) for point in outline[1:].split()), strict=True)
                bounds.append((min(xs), min(ys), max(xs), max(ys)))
            self.assertEqual(sorted(bounds), [(0, 20, 18, 40), (200, 120, 390, 290)])
            with Image.open(BytesIO(b64decode(event["overlay"]))) as overlay:
                self.assertEqual(overlay.size, self.photo.size)
                self.assertEqual(overlay.getpixel((10, 10)), (0, 0, 0, 217))
                self.assertEqual(overlay.getpixel((250, 180)), (0, 0, 0, 0))
                self.assertIsNone(
                    ImageChops.difference(overlay.getchannel("A").point(lambda value: 0 if value else 255), self.mask).getbbox()
                )

            release_classifier.set()
            self.assertEqual(json.loads(response.readline()), {"research_prediction": "NV", "route": "MEL_OR_NV"})

        classified = self.classifier.predict.call_args.args[0]
        expected = Image.composite(self.photo, Image.new("RGB", self.photo.size), self.mask)
        self.assertIsNone(ImageChops.difference(classified, expected).getbbox())

    def test_empty_mask_skips_classification(self) -> None:
        self.start_client()
        self.send_photo("application/json")
        with self.client.getresponse() as response:
            self.assertEqual(response.status, 200)
            result = json.load(response)
        self.assertIsNone(result["research_prediction"])
        self.assertIsNone(result["route"])
        with Image.open(BytesIO(b64decode(result["overlay"]))) as overlay:
            self.assertEqual(overlay.getchannel("A").getextrema(), (217, 217))
        self.send_photo(ANALYSIS_STREAM)
        with self.client.getresponse() as response:
            self.assertEqual(json.loads(response.readline())["outline"], "")
            self.assertEqual(json.loads(response.readline()), {"research_prediction": None, "route": None})
        self.classifier.predict.assert_not_called()

    def test_padded_classifier_receives_the_training_crop(self) -> None:
        router, other = Mock(), Mock()
        router.uses_specialist.return_value = True
        self.start_client(padded_crop=True, routing=(router, other))
        self.photo.paste((210, 25, 40), (220, 170, 255, 190))
        self.mask.paste(255, (220, 170, 255, 190))
        self.classifier.predict.return_value = {"research_prediction": "MEL", "route": "MEL_OR_NV"}

        self.send_photo("application/json")
        with self.client.getresponse() as response:
            self.assertEqual(response.status, 200)
            result = json.load(response)
        self.assertEqual(result["research_prediction"], "MEL")
        self.assertEqual(result["analysis_input"], "padded_crop")
        classified = self.classifier.predict.call_args.args[0]
        self.assertIsNone(ImageChops.difference(classified, crop_with_skin_margin(self.photo, self.mask)).getbbox())
        self.assertIsNone(ImageChops.difference(router.uses_specialist.call_args.args[0], self.photo).getbbox())
        other.predict_other.assert_not_called()

    def test_other_branch_uses_the_full_photo_and_skips_the_specialist(self) -> None:
        router, other = Mock(), Mock()
        router.uses_specialist.return_value = False
        other.predict_other.return_value = ("BCC", 0.23145678)
        self.start_client(padded_crop=True, routing=(router, other))
        self.photo.paste((210, 25, 40), (220, 170, 255, 190))
        self.mask.paste(255, (220, 170, 255, 190))

        self.send_photo(ANALYSIS_STREAM)
        with self.client.getresponse() as response:
            self.assertTrue(json.loads(response.readline())["outline"])
            self.assertEqual(
                json.loads(response.readline()),
                {"research_prediction": "BCC", "diagnosis_score": 0.23145678, "route": "OTHER", "analysis_input": "full_frame"},
            )
        other.predict_other.assert_called_once()
        self.assertIsNone(ImageChops.difference(other.predict_other.call_args.args[0], self.photo).getbbox())
        self.classifier.predict.assert_not_called()

    def test_padded_classifier_uses_the_original_photo_with_an_empty_mask(self) -> None:
        router, other = Mock(), Mock()
        self.start_client(padded_crop=True, routing=(router, other))
        self.classifier.predict.return_value = {"research_prediction": "NV", "route": "MEL_OR_NV"}

        self.send_photo(ANALYSIS_STREAM)
        with self.client.getresponse() as response:
            first = json.loads(response.readline())
            result = json.loads(response.readline())
        self.assertEqual(first["outline"], "")
        with Image.open(BytesIO(b64decode(first["overlay"]))) as overlay:
            self.assertEqual(overlay.getchannel("A").getextrema(), (0, 0))
        self.assertEqual(result["analysis_input"], "original_photo")
        self.assertEqual(result["research_prediction"], "NV")
        classified = self.classifier.predict.call_args.args[0]
        self.assertIsNone(ImageChops.difference(classified, self.photo).getbbox())
        router.uses_specialist.assert_not_called()
        other.predict_other.assert_not_called()
