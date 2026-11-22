import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from classifiers.score_bands import HIGH_MELANOMA_SCORE, MODERATE_MELANOMA_SCORE, melanoma_score_band
from classifiers.single_photo import LesionRouter, prepare_photo


class SinglePhotoTest(TestCase):
    def test_router_cutoff_defaults_to_manifest_and_accepts_override(self) -> None:
        with TemporaryDirectory() as directory, patch("classifiers.single_photo.load_routed_models"):
            manifest = Path(directory) / "model.json"
            manifest.write_text(json.dumps({"cutoffs": {"router": 0.1}, "weights": [{"routed": "fold.pt"}]}))
            device = torch.device("cpu")

            self.assertEqual(LesionRouter(manifest, device).cutoff, 0.1)
            self.assertEqual(LesionRouter(manifest, device, cutoff=0.2).cutoff, 0.2)

    def test_melanoma_score_band_boundaries(self) -> None:
        for cutoff, lower_band, new_band in (
            (0.001, "below", "low"),
            (MODERATE_MELANOMA_SCORE, "low", "moderate"),
            (HIGH_MELANOMA_SCORE, "moderate", "high"),
        ):
            with self.subTest(cutoff=cutoff):
                self.assertEqual(melanoma_score_band(float(np.nextafter(cutoff, -np.inf)), 0.001), lower_band)
                self.assertEqual(melanoma_score_band(cutoff, 0.001), new_band)

    def test_large_photos_keep_the_entire_frame_and_aspect_ratio(self) -> None:
        landscape = Image.new("RGB", (1200, 900))
        portrait = Image.new("RGB", (300, 400))
        self.assertEqual(tuple(prepare_photo(landscape).shape), (3, 450, 600))
        self.assertEqual(tuple(prepare_photo(portrait).shape), (3, 400, 300))
