from unittest import TestCase

import numpy as np

from classifiers.compare_masked_mel_nv import cutoff_at_target, summarize


class MaskedMelNvComparisonTest(TestCase):
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
