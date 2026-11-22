"""Train, calibrate, evaluate, and export the retained clinical-photo classifiers."""

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import cache
from pathlib import Path
from statistics import NormalDist
from tempfile import TemporaryDirectory

import joblib
import numpy as np
import pandas as pd
import segmentation_models_pytorch as smp
import sklearn
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from classifiers import dermfm_zero
from classifiers.catalog import load_catalog
from classifiers.eval_split.create_splits import FULL_MANIFEST_NAME, freeze_splits
from classifiers.lesion_crop import PADDED_CROP_SETTINGS, crop_with_skin_margin, frame_photo
from classifiers.model import TASK_LABELS, Task, ensemble_probabilities, predict_labels
from classifiers.segmentation import SEGMENTATION_SIZE, LesionSegmenter

FOLDS = (1, 2, 3)


def default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


@dataclass(frozen=True)
class TrainingConfig:
    data_dir: Path
    task: Task
    device: torch.device
    catalog: Path | None = None
    segmentation_model: Path | None = None
    batch_size: int = 12
    calibration_fraction: float = 0.20
    calibration_seed: int = 42
    sensitivity_target: float = 0.95
    regularization_c: float = 1.0

    def __post_init__(self) -> None:
        if self.task not in TASK_LABELS:
            raise ValueError(f"Unknown training task: {self.task}")
        if self.task == "padded_crop" and (self.segmentation_model is None or not self.segmentation_model.expanduser().is_file()):
            raise ValueError("Padded-crop training requires a U-Net checkpoint file")
        if self.batch_size < 1 or not math.isfinite(self.regularization_c) or self.regularization_c <= 0:
            raise ValueError("Batch size and regularization C must be positive")
        if not 0 < self.calibration_fraction < 1 or not 0 < self.sensitivity_target <= 1:
            raise ValueError("Use a calibration fraction in (0, 1) and a melanoma sensitivity target in (0, 1]")


@dataclass(frozen=True)
class TrainingData:
    rows: pd.DataFrame
    calibration: pd.Series
    provenance: dict


@dataclass(frozen=True)
class FittedModel:
    heads: list[Pipeline]
    cutoff: float
    training_rows_per_member: dict[int, int]


def calibration_mask(rows: pd.DataFrame, fraction: float = 0.20, seed: int = 42) -> pd.Series:
    """Reserve complete groups, stratified by diagnoses, inside each development fold."""
    development = rows.loc[rows["split"].eq("train")]
    groups = development.groupby("group_id").agg(
        validation_fold=("validation_fold", "first"), diagnoses=("diagnosis_code", lambda codes: tuple(sorted(set(codes))))
    )
    rng = np.random.default_rng(seed)
    reserved = []
    for _, stratum in groups.groupby(["validation_fold", "diagnoses"]):
        count = int(len(stratum) * fraction)
        reserved.extend(rng.choice(stratum.index.to_numpy(), size=count, replace=False))
    return pd.Series(rows["group_id"].isin(reserved), index=rows.index)


def prepare_data(config: TrainingConfig) -> TrainingData:
    data_dir = config.data_dir.expanduser().resolve()
    parent = freeze_splits(data_dir).rename_axis("lesion_id").reset_index()
    provenance = {
        "manifest_sha256": hashlib.sha256((data_dir / "splits" / FULL_MANIFEST_NAME).read_bytes()).hexdigest(),
        "ground_truth": "MILK10k_Training_GroundTruth.csv",
    }
    if config.catalog is not None:
        catalog = load_catalog(config.catalog, data_dir)
        parent = catalog.milk_lesions(parent)
        provenance["catalog_sha256"] = catalog.sha256
    parent = parent.set_index("lesion_id")
    calibration = calibration_mask(parent, config.calibration_fraction, config.calibration_seed)
    rows = parent.loc[parent["diagnosis_code"].isin(TASK_LABELS[config.task])]
    calibration = calibration.loc[rows.index]
    if not (calibration & rows["diagnosis_code"].eq("MEL")).any():
        raise ValueError("Calibration requires at least one reserved melanoma group")
    return TrainingData(rows, calibration, provenance)


def extract_features(config: TrainingConfig, data: TrainingData) -> tuple[np.ndarray, dict]:
    """Apply the task's image preparation and encode frozen DermFM features."""
    data_dir = config.data_dir.expanduser().resolve()
    settings = {}
    prepare_photo: Callable[[Image.Image], Image.Image] | None = None
    if config.task == "padded_crop":
        assert config.segmentation_model is not None
        checkpoint = config.segmentation_model.expanduser().resolve()
        with checkpoint.open("rb") as weights:
            checksum = hashlib.file_digest(weights, "sha256").hexdigest()
        settings = {
            "transform": "padded_crop",
            "segmentation_checkpoint_sha256": checksum,
            "segmentation_input_size": str(SEGMENTATION_SIZE),
            "segmentation_models_pytorch": smp.__version__,
            "pillow": Image.__version__,
            "crop_settings": json.dumps(PADDED_CROP_SETTINGS, sort_keys=True),
            "empty_mask_policy": "original_photo",
        }

        @cache
        def segmenter() -> LesionSegmenter:
            return LesionSegmenter(checkpoint, config.device)

        def prepare_padded_photo(photo: Image.Image) -> Image.Image:
            framed = frame_photo(photo)
            return crop_with_skin_margin(framed, segmenter().mask(framed))

        prepare_photo = prepare_padded_photo

    features, encoder, reused = dermfm_zero.encode_clinical(
        data.rows, data_dir, config.device, config.batch_size, prepare_photo=prepare_photo, image_settings=settings
    )
    if features.ndim != 2 or len(features) != len(data.rows) or not np.isfinite(features).all():
        raise ValueError("Expected a finite feature vector for every selected clinical photo")

    provenance = {
        "encoder": encoder,
        "feature_cache_hit": reused,
        "feature_dimensions": features.shape[1],
        "pretraining_isic_id_overlap": dermfm_zero.pretraining_id_overlap(data.rows, data_dir),
    }
    if config.task == "padded_crop":
        provenance.update(
            context_transforms={"padded_crop": PADDED_CROP_SETTINGS},
            empty_mask_policy="original_photo",
            segmentation_checkpoint_sha256=settings["segmentation_checkpoint_sha256"],
            segmentation_input_size=SEGMENTATION_SIZE,
        )
    return features, provenance


def fit_frozen_head(features: np.ndarray, targets: np.ndarray, regularization_c: float) -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(C=regularization_c, max_iter=2000)).fit(features, targets)


def fit_model(config: TrainingConfig, data: TrainingData, features: np.ndarray) -> FittedModel:
    """Fit three members and choose the cutoff using only reserved development groups."""
    diagnosis = np.asarray(data.rows["diagnosis_code"])
    labels = TASK_LABELS[config.task]
    targets = diagnosis == "MEL" if config.task == "padded_crop" else diagnosis
    expected_classes = (False, True) if config.task == "padded_crop" else labels
    eligible = data.rows["split"].eq("train") & ~data.calibration
    heads, counts = [], {}
    with threadpool_limits(limits=4):
        for fold in FOLDS:
            train = (eligible & data.rows["validation_fold"].ne(fold).fillna(False)).to_numpy(dtype=bool)
            if tuple(np.unique(targets[train])) != expected_classes:
                raise ValueError(f"Training outside fold {fold} is missing a class")
            heads.append(fit_frozen_head(features[train], targets[train], config.regularization_c))
            counts[fold] = int(train.sum())

        calibration_melanomas = (diagnosis == "MEL") & data.calibration.to_numpy(dtype=bool)
        scores = ensemble_probabilities(heads, features[calibration_melanomas])[:, labels.index("MEL")]
    required = math.ceil(len(scores) * config.sensitivity_target)
    return FittedModel(heads, float(np.sort(scores)[-required]), counts)


def rate(count: int, total: int) -> dict:
    if not total:
        return {"count": count, "total": total, "rate": None, "wilson_95_ci": None}
    p = count / total
    z = NormalDist().inv_cdf(0.975)
    denominator = 1 + z**2 / total
    midpoint = (p + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return {"count": count, "total": total, "rate": p, "wilson_95_ci": [max(0, midpoint - half_width), min(1, midpoint + half_width)]}


def evaluate_model(
    config: TrainingConfig, data: TrainingData, features: np.ndarray, model: FittedModel
) -> tuple[dict, dict[str, pd.DataFrame]]:
    labels = TASK_LABELS[config.task]
    reports, predictions = {}, {}
    for split, selected in (("calibration", data.calibration), ("holdout", data.rows["split"].eq("test"))):
        rows = data.rows.loc[selected]
        actual = np.asarray(rows["diagnosis_code"], dtype=str)
        with threadpool_limits(limits=4):
            scores = ensemble_probabilities(model.heads, features[selected.to_numpy(dtype=bool)])
        predicted = predict_labels(scores, labels, model.cutoff)
        melanoma = actual == "MEL"
        flagged = predicted == "MEL"
        reports[split] = {
            "lesions": len(rows),
            "melanoma_sensitivity": rate(int((melanoma & flagged).sum()), int(melanoma.sum())),
            "melanoma_specificity": rate(int((~melanoma & ~flagged).sum()), int((~melanoma).sum())),
            "melanoma_auroc": float(roc_auc_score(melanoma, scores[:, labels.index("MEL")])) if np.unique(melanoma).size == 2 else None,
            "classification_report": classification_report(
                actual,
                predicted,
                labels=list(labels),
                output_dict=True,
                zero_division=0,  # pyright: ignore[reportArgumentType]
            ),
            "confusion_matrix": confusion_matrix(actual, predicted, labels=list(labels)).tolist(),
        }
        predictions[split] = (
            rows[["diagnosis_code", "validation_fold"]]
            .assign(prediction=predicted)
            .join(pd.DataFrame(scores, index=rows.index, columns=[f"score_{label}" for label in labels]))
        )
    return reports, predictions


def save_model(
    config: TrainingConfig,
    data: TrainingData,
    model: FittedModel,
    provenance: dict,
    reports: dict,
    predictions: dict[str, pd.DataFrame],
    output_root: Path,
) -> Path:
    """Publish a new model directory after all its files have been written."""
    output_root = output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = output_root / f"{config.task.replace('_', '-')}-{run_id}"
    metrics = {
        "task": config.task,
        "diagnosis_order": TASK_LABELS[config.task],
        "methods": {config.task: {"cutoff": model.cutoff}},
        "calibration_seed": config.calibration_seed,
        "calibration_fraction": config.calibration_fraction,
        "melanoma_sensitivity_target": config.sensitivity_target,
        "regularization_c": config.regularization_c,
        "scikit_learn": sklearn.__version__,
        "training_rows_per_member": model.training_rows_per_member,
        **data.provenance,
        **provenance,
        **reports,
    }
    with TemporaryDirectory(dir=output_root) as temp:
        staging = Path(temp) / "model"
        staging.mkdir()
        for split, rows in predictions.items():
            rows.to_csv(staging / f"{split}_predictions.csv")
        joblib.dump({config.task: model.heads}, staging / "heads.joblib")
        (staging / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        staging.rename(output)
    return output
