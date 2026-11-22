from unittest import TestCase

import numpy as np
from PIL import Image

from classifiers.single_photo import HIGH_MELANOMA_SCORE, MODERATE_MELANOMA_SCORE, melanoma_score_band, prepare_photo


class SinglePhotoTest(TestCase):
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
