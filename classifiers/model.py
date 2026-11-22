"""Load and run the retained DermFM classifiers."""

import json
from pathlib import Path
from typing import Literal, cast

import joblib
import numpy as np
import torch
from PIL import Image
from sklearn.pipeline import Pipeline

from classifiers import dermfm_zero
from classifiers.catalog import MILK_DIAGNOSES
from classifiers.encoders import load_rgb
from classifiers.lesion_crop import PADDED_CROP_SETTINGS, PHOTO_SIZE, crop_with_skin_margin, frame_photo
from classifiers.score_bands import melanoma_score_band
from classifiers.segmentation import LesionSegmenter

Task = Literal["padded_crop", "direct_11"]
TASK_LABELS = {"padded_crop": ("NV", "MEL"), "direct_11": MILK_DIAGNOSES}


def ensemble_probabilities(heads: list[Pipeline], features: np.ndarray) -> np.ndarray:
    scores = np.mean([head.predict_proba(features) for head in heads], axis=0)
    if not np.isfinite(scores).all():
        raise RuntimeError("The classifier returned non-finite scores")
    return scores


def predict_labels(scores: np.ndarray, labels: tuple[str, ...], cutoff: float) -> np.ndarray:
    melanoma = labels.index("MEL")
    other_scores = scores.copy()
    other_scores[:, melanoma] = -np.inf
    return np.where(scores[:, melanoma] >= cutoff, "MEL", np.asarray(labels)[other_scores.argmax(axis=1)])


def load_classifier_run(run_dir: Path, expected_task: Task | None = None) -> tuple[Task, float, list[Pipeline]]:
    run_dir = run_dir.expanduser().resolve()
    config = json.loads((run_dir / "metrics.json").read_text())
    # The original padded-crop export has no task field.
    task = config.get("task", "padded_crop")
    if task not in TASK_LABELS or (expected_task is not None and task != expected_task):
        raise ValueError(f"Unexpected classifier task: {task}")
    if config.get("encoder", {}).get("sha256") != dermfm_zero.WEIGHTS_SHA256:
        raise ValueError("Expected a model trained with the pinned DermFM-Zero-Open encoder")
    if task == "padded_crop" and (
        config.get("context_transforms", {}).get("padded_crop") != PADDED_CROP_SETTINGS
        or config.get("empty_mask_policy") != "original_photo"
    ):
        raise ValueError("Expected a padded-crop run with the original-photo fallback")

    cutoff = float(config["methods"][task]["cutoff"])
    heads: list[Pipeline] = joblib.load(run_dir / "heads.joblib")[task]
    expected_classes = (False, True) if task == "padded_crop" else TASK_LABELS[task]
    if not 0 <= cutoff <= 1 or len(heads) != 3 or any(tuple(head.classes_) != expected_classes for head in heads):
        raise ValueError("Expected a valid cutoff and three fitted heads with the saved task's labels")
    return task, cutoff, heads


class DermFMClassifier:
    def __init__(
        self,
        run_dir: Path,
        data_dir: Path,
        device: torch.device,
        *,
        expected_task: Task | None = None,
        segmentation_model: Path | None = None,
        encoder: dermfm_zero.DermFMVisionTransformer | None = None,
    ) -> None:
        self.task, self.cutoff, self.heads = load_classifier_run(run_dir, expected_task)
        self.labels = TASK_LABELS[self.task]
        self.device = device
        self.preprocess = dermfm_zero.clinical_preprocess()
        self.encoder = encoder if encoder is not None else dermfm_zero.load_clinical_encoder(data_dir, device)
        self.segmentation_model = segmentation_model
        self.segmenter: LesionSegmenter | None = None

    @torch.inference_mode()
    def score_photo(self, photo: Image.Image) -> np.ndarray:
        """Score a prepared RGB photo. The demo supplies its own padded crop."""
        tensor = cast(torch.Tensor, self.preprocess(photo))
        image = tensor.unsqueeze(0).to(device=self.device, dtype=dermfm_zero.clinical_dtype(self.device))
        features = self.encoder(image).float().cpu().numpy()
        return ensemble_probabilities(self.heads, features)[0]

    def predict(self, photo: Image.Image) -> dict[str, str | float | None]:
        scores = self.score_photo(photo)
        prediction = str(predict_labels(scores[None, :], self.labels, self.cutoff)[0])
        melanoma = float(scores[self.labels.index("MEL")])
        result: dict[str, str | float | None] = {
            "research_prediction": prediction,
            "route": "MEL_OR_NV" if self.task == "padded_crop" else None,
            "melanoma_score": melanoma,
        }
        if self.task == "padded_crop":
            result["melanoma_score_band"] = melanoma_score_band(melanoma, self.cutoff)
        return result

    def predict_other(self, photo: Image.Image) -> str:
        """Pick the strongest diagnosis among the nine classes handled by the other branch."""
        scores = self.score_photo(photo)
        other_indices = (index for index, label in enumerate(self.labels) if label not in {"MEL", "NV"})
        return self.labels[max(other_indices, key=lambda index: scores[index])]

    def predict_file(self, path: Path) -> dict[str, str | float | None]:
        photo = load_rgb(path)
        if self.task == "direct_11":
            max_edge = max(PHOTO_SIZE)
            photo.thumbnail((max_edge, max_edge), Image.Resampling.LANCZOS)
            return self.predict(photo)

        if self.segmenter is None:
            if self.segmentation_model is None:
                raise ValueError("Pass the U-Net checkpoint as segmentation_model to classify a file with the padded-crop model")
            self.segmenter = LesionSegmenter(self.segmentation_model.expanduser(), self.device)
        photo = frame_photo(photo)
        mask = self.segmenter.mask(photo)
        return {
            **self.predict(crop_with_skin_margin(photo, mask)),
            "analysis_input": "padded_crop" if mask.getbbox() is not None else "original_photo",
        }
