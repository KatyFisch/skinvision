from types import SimpleNamespace
from typing import cast
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd

from classifiers.catalog import MILK_DIAGNOSES
from classifiers.compare_clinical import calibration_mask, fit_ensembles


class ClinicalComparisonTest(TestCase):
    def test_calibration_groups_and_holdout_are_excluded_from_every_ensemble_member(self) -> None:
        records = [
            {
                "validation_fold": fold,
                "group_id": f"{fold}-{diagnosis}-{group}",
                "diagnosis_code": diagnosis,
                "split": "train" if fold else "test",
            }
            for fold in range(4)
            for diagnosis in MILK_DIAGNOSES
            for group in range(5)
            for _ in range(2)
        ]
        rows = pd.DataFrame(records)
        features = np.arange(len(rows))[:, None]
        calibration = calibration_mask(rows)
        self.assertTrue(calibration.groupby(rows["group_id"]).nunique().eq(1).all())
        self.assertEqual(set(rows.loc[calibration, "split"]), {"train"})
        self.assertEqual(len(rows.loc[calibration]), 3 * len(MILK_DIAGNOSES) * 2)

        def fit(selected: np.ndarray, labels: np.ndarray) -> SimpleNamespace:
            return SimpleNamespace(rows=selected[:, 0], classes_=np.unique(labels))

        with patch("classifiers.compare_clinical.fit_frozen_head", side_effect=fit):
            ensembles = fit_ensembles(rows, features, calibration)
        for members in ensembles.values():
            self.assertEqual(len(members), 3)
            for fold, member in enumerate(members, start=1):
                seen = cast(SimpleNamespace, member).rows
                training = rows.loc[seen]
                self.assertEqual(set(training["split"]), {"train"})
                self.assertEqual(set(training["validation_fold"]), {1, 2, 3} - {fold})
                self.assertFalse(calibration.loc[seen].any())
