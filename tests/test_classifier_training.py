from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from classifiers.catalog import MILK_DIAGNOSES
from classifiers.dermfm_zero import WEIGHTS_SHA256
from classifiers.export_padded_crop import PaddedCropTorchModel, export_heads
from classifiers.model import TASK_LABELS, DermFMClassifier, Task
from classifiers.score_bands import melanoma_score_band
from classifiers.train import TrainingConfig, TrainingData, calibration_mask, evaluate_model, fit_frozen_head, fit_model, save_model


class ClassifierTrainingTest(TestCase):
    def setUp(self) -> None:
        self.rows = pd.DataFrame(
            [
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
        )
        self.calibration = calibration_mask(self.rows)

    def test_calibration_groups_and_holdout_are_excluded_from_every_ensemble_member(self) -> None:
        rows = self.rows
        calibration = self.calibration
        self.assertTrue(calibration.groupby(rows["group_id"]).nunique().eq(1).all())
        self.assertEqual(set(rows.loc[calibration, "split"]), {"train"})
        self.assertEqual(len(rows.loc[calibration]), 3 * len(MILK_DIAGNOSES) * 2)

        features = np.arange(len(rows), dtype=float)[:, None]
        config = TrainingConfig(Path("data"), "direct_11", torch.device("cpu"))

        with patch("classifiers.train.fit_frozen_head", wraps=fit_frozen_head) as fit:
            model = fit_model(config, TrainingData(rows, calibration, {}), features)

        self.assertEqual(len(model.heads), 3)
        for fold, call in enumerate(fit.call_args_list, start=1):
            seen = call.args[0][:, 0].astype(int)
            training = rows.loc[seen]
            self.assertEqual(set(training["split"]), {"train"})
            self.assertEqual(set(training["validation_fold"]), {1, 2, 3} - {fold})
            self.assertFalse(calibration.loc[seen].any())

    def test_both_tasks_save_and_reload_with_the_same_decision_rule(self) -> None:
        with TemporaryDirectory() as workspace:
            root = Path(workspace)
            checkpoint = root / "unet.pth"
            checkpoint.touch()
            tasks: tuple[Task, ...] = ("padded_crop", "direct_11")
            for task in tasks:
                with self.subTest(task=task):
                    config = TrainingConfig(root, task, torch.device("cpu"), segmentation_model=checkpoint)
                    rows = self.rows.loc[self.rows["diagnosis_code"].isin(TASK_LABELS[task])]
                    data = TrainingData(rows, self.calibration.loc[rows.index], {})
                    features = np.eye(len(MILK_DIAGNOSES))[[MILK_DIAGNOSES.index(label) for label in rows["diagnosis_code"]]]
                    model = fit_model(config, data, features)
                    reports, predictions = evaluate_model(config, data, features, model)
                    provenance: dict = {"encoder": {"sha256": WEIGHTS_SHA256}}
                    if task == "padded_crop":
                        provenance.update(
                            context_transforms={"padded_crop": {"margin_fraction_of_longest_side": 0.25, "minimum_margin_px": 12}},
                            empty_mask_policy="original_photo",
                        )
                    output = save_model(config, data, model, provenance, reports, predictions, root)
                    sample_id = predictions["holdout"].index[-1]
                    sample = features[rows.index.get_loc(sample_id)]
                    photo = Image.new("RGB", (16, 16))
                    demo_results = []
                    with patch(
                        "classifiers.dermfm_zero.load_clinical_encoder", return_value=lambda _, encoded=sample: torch.tensor(encoded[None])
                    ):
                        reloaded = DermFMClassifier(output, root, config.device, expected_task=task)
                        result = reloaded.predict(photo)
                        if task == "padded_crop":
                            exported = root / "heads.pth"
                            export_heads(output, exported)
                            demo_results = [PaddedCropTorchModel(path, root, config.device).predict(photo) for path in (output, exported)]
                    self.assertEqual(result["research_prediction"], predictions["holdout"].at[sample_id, "prediction"])
                    score = result["melanoma_score"]
                    assert isinstance(score, float)
                    np.testing.assert_allclose(score, predictions["holdout"].at[sample_id, "score_MEL"], rtol=1e-6)
                    if task == "padded_crop":
                        self.assertEqual(result["melanoma_score_band"], melanoma_score_band(score, reloaded.cutoff))
                        for demo_result in demo_results:
                            self.assertEqual(demo_result["research_prediction"], result["research_prediction"])
                            self.assertEqual(demo_result["melanoma_score_band"], result["melanoma_score_band"])
                            demo_score = demo_result["melanoma_score"]
                            assert isinstance(demo_score, float)
                            np.testing.assert_allclose(demo_score, score, rtol=1e-6)
                    else:
                        self.assertNotIn("melanoma_score_band", result)
                        competing_scores = np.zeros(len(reloaded.labels))
                        competing_scores[reloaded.labels.index("MEL")] = 0.50
                        competing_scores[reloaded.labels.index("NV")] = 0.30
                        competing_scores[reloaded.labels.index("BCC")] = 0.20
                        with patch.object(reloaded, "score_photo", return_value=competing_scores):
                            self.assertEqual(reloaded.predict_other(photo), "BCC")
