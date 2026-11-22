from unittest import TestCase

import numpy as np
import pandas as pd
from PIL import Image

from classifiers.single_photo import decisions, lock_cutoffs, prepare_photo, summarize_pipeline


class SinglePhotoTest(TestCase):
    def test_large_photos_keep_the_entire_frame_and_aspect_ratio(self) -> None:
        landscape = Image.new("RGB", (1200, 900))
        portrait = Image.new("RGB", (300, 400))
        self.assertEqual(tuple(prepare_photo(landscape).shape), (3, 450, 600))
        self.assertEqual(tuple(prepare_photo(portrait).shape), (3, 400, 300))

    def test_cutoffs_target_end_to_end_sensitivity_and_account_for_both_failures(self) -> None:
        scores = np.arange(1, 201) / 200
        melanomas = pd.DataFrame({"diagnosis_code": "MEL", "router_score": scores, "melanoma_score": 1 - scores, "other_prediction": "BCC"})
        negatives = pd.DataFrame(
            [
                {"diagnosis_code": "NV", "router_score": 0.1, "melanoma_score": 0.9, "other_prediction": "SCCKA"},
                {"diagnosis_code": "BCC", "router_score": 0.01, "melanoma_score": 0.9, "other_prediction": "BCC"},
            ]
        )
        calibration = pd.concat((melanomas, negatives), ignore_index=True)
        scored = decisions(calibration, lock_cutoffs(calibration))
        summary = summarize_pipeline(scored)
        self.assertEqual(summary["melanoma_sensitivity"]["count"], 190)
        self.assertEqual(summary["melanomas_missed_at_router"], 2)
        self.assertEqual(summary["melanomas_missed_at_specialist"], 8)
        self.assertEqual(scored.tail(2)["prediction"].tolist(), ["MEL", "BCC"])
