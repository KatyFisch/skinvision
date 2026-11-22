from unittest import TestCase

import numpy as np
from PIL import Image

from classifiers.compare_masked_mel_nv import cutoff_at_target, dim_background, summarize
from classifiers.lesion_crop import crop_with_skin_margin


class MaskedMelNvComparisonTest(TestCase):
    def test_dimming_preserves_the_lesion_and_nearby_skin_at_native_size(self) -> None:
        photo = Image.new("RGB", (600, 450), (200, 100, 50))
        mask = Image.new("L", photo.size)
        mask.paste(255, (200, 200, 220, 220))

        dimmed = dim_background(photo, mask)

        self.assertEqual(dimmed.size, photo.size)
        self.assertEqual(dimmed.getpixel((210, 210)), (200, 100, 50))
        self.assertEqual(dimmed.getpixel((230, 210)), (200, 100, 50))
        self.assertEqual(dimmed.getpixel((233, 210)), (100, 50, 25))

    def test_crop_keeps_an_edge_lesion_in_frame_without_stretching(self) -> None:
        photo = Image.new("RGB", (600, 450), (180, 180, 180))
        mask = Image.new("L", photo.size)
        bounds = (580, 440, 600, 450)
        photo.paste((255, 0, 0), bounds)
        mask.paste(255, bounds)

        crop = crop_with_skin_margin(photo, mask)
        pixels = np.asarray(crop)
        y, x = np.where((pixels[:, :, 0] > 220) & (pixels[:, :, 1] < 90))
        width, height = x.max() - x.min() + 1, y.max() - y.min() + 1

        self.assertEqual(crop.size, photo.size)
        self.assertEqual((x.max(), y.max()), (599, 449))
        self.assertAlmostEqual(width / height, 2, delta=0.05)
        self.assertEqual(crop.getpixel((0, 0)), (180, 180, 180))

    def test_empty_masks_count_against_coverage_and_can_make_the_sensitivity_target_unreachable(self) -> None:
        melanoma = np.array([True] * 20 + [False] * 3)
        scores = np.array([0.8] * 18 + [np.nan, np.nan, 0.1, 0.9, np.nan])
        cutoff = cutoff_at_target(melanoma, scores)
        result = summarize(melanoma, scores, cutoff)

        self.assertEqual(cutoff, 0.8)
        self.assertEqual(result["melanoma_sensitivity"]["count"], 18)
        self.assertEqual(result["melanoma_sensitivity"]["total"], 20)
        self.assertEqual(result["nevus_specificity"]["count"], 1)
        self.assertEqual(result["nevus_specificity"]["total"], 3)
        self.assertEqual(result["inconclusive"], {"MEL": 2, "NV": 1})
        self.assertIsNone(result["matched_95_specificity"])
