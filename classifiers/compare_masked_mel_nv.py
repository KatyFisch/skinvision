"""Compare masked and unmasked clinical photos for melanoma versus nevus."""

import argparse
import hashlib
import json
import math
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import cv2
import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from classifiers import dermfm_zero
from classifiers.catalog import load_catalog
from classifiers.compare_clinical import CALIBRATION_FRACTION, CALIBRATION_SEED, calibration_mask
from classifiers.eval_split.create_splits import FULL_MANIFEST_NAME
from classifiers.eval_split.prepare_routed_tasks import prepare_routed_tasks
from classifiers.lesion_crop import CROP_MARGIN_FRACTION, PHOTO_SIZE, SKIN_MARGIN_PX, crop_with_skin_margin
from classifiers.single_photo import PIPELINE_TARGET, rate
from classifiers.train_mel_nv import fit_frozen_head
from demo.segmentation import SEGMENTATION_SIZE, LesionSegmenter

BACKGROUND_BRIGHTNESS = 0.5
PhotoTransform = Callable[[Image.Image, Image.Image], Image.Image]


def black_background(photo: Image.Image, mask: Image.Image) -> Image.Image:
    return Image.composite(photo, Image.new("RGB", photo.size), mask)


def dim_background(photo: Image.Image, mask: Image.Image) -> Image.Image:
    diameter = 2 * SKIN_MARGIN_PX + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
    expanded = Image.fromarray(cv2.dilate(np.asarray(mask), kernel))
    dimmed = ImageEnhance.Brightness(photo).enhance(BACKGROUND_BRIGHTNESS)
    return Image.composite(photo, dimmed, expanded)


def prepare_variant_photos(
    rows: pd.DataFrame, data_dir: Path, output: Path, segmenter: LesionSegmenter, variants: dict[str, PhotoTransform]
) -> tuple[dict[str, pd.DataFrame], pd.Series]:
    directories = {name: output / f"{name}_photos" for name in variants}
    for directory in directories.values():
        directory.mkdir()

    foreground = {}
    paths: dict[str, dict[str, str]] = {name: {} for name in variants}
    for count, (lesion_id, clinical_path) in enumerate(rows["clinical_path"].items(), 1):
        with Image.open(data_dir / clinical_path) as source:
            photo = ImageOps.exif_transpose(source).convert("RGB")
        if photo.size != PHOTO_SIZE:
            raise ValueError(f"Expected a 600 × 450 clinical photo for {lesion_id}, got {photo.size}")
        mask = segmenter.mask(photo)
        foreground[lesion_id] = float(np.asarray(mask).mean() / 255)
        if mask.getbbox() is not None:
            for name, transform in variants.items():
                path = directories[name] / f"{lesion_id}.png"
                transform(photo, mask).save(path, compress_level=1)
                paths[name][str(lesion_id)] = str(path.relative_to(data_dir))
        if count == 1 or count % 120 == 0 or count == len(rows):
            print(f"Segmented {count}/{len(rows)} photos", flush=True)
    coverage = pd.Series(foreground, name="mask_foreground_fraction").reindex(rows.index)
    usable = rows.loc[coverage.gt(0)]
    return {name: usable.assign(clinical_path=pd.Series(variant_paths)) for name, variant_paths in paths.items()}, coverage


def score_ensemble(features: pd.DataFrame, target: pd.Series, training: list[pd.Series]) -> tuple[list[Pipeline], pd.Series]:
    heads = [fit_frozen_head(features.loc[mask].to_numpy(), np.asarray(target[mask])) for mask in training]
    valid = features.dropna()
    scores = np.mean([head.predict_proba(valid.to_numpy())[:, 1] for head in heads], axis=0)
    return heads, pd.Series(scores, index=valid.index).reindex(features.index)


def cutoff_at_target(melanoma: np.ndarray, scores: np.ndarray) -> float:
    positives = np.sort(scores[melanoma & np.isfinite(scores)])
    if not len(positives):
        raise ValueError("Cutoff selection requires at least one scored melanoma")
    required = math.ceil(np.count_nonzero(melanoma) * PIPELINE_TARGET)
    return float(positives[-min(required, len(positives))])


def summarize(melanoma: np.ndarray, scores: np.ndarray, cutoff: float) -> dict:
    scored = np.isfinite(scores)
    flagged = scored & (scores >= cutoff)
    negatives_cleared = scored & ~flagged & ~melanoma
    valid_melanomas = melanoma & scored
    required = math.ceil(np.count_nonzero(melanoma) * PIPELINE_TARGET)
    matched = None
    if np.count_nonzero(valid_melanomas) >= required and required:
        test_cutoff = cutoff_at_target(melanoma, scores)
        matched = rate(int(np.count_nonzero(scored & ~melanoma & (scores < test_cutoff))), int(np.count_nonzero(~melanoma)))
    return {
        "melanoma_sensitivity": rate(int(np.count_nonzero(flagged & melanoma)), int(np.count_nonzero(melanoma))),
        "nevus_specificity": rate(int(np.count_nonzero(negatives_cleared)), int(np.count_nonzero(~melanoma))),
        "inconclusive": {"MEL": int(np.count_nonzero(melanoma & ~scored)), "NV": int(np.count_nonzero(~melanoma & ~scored))},
        "matched_95_specificity": matched,
    }


def compare_heads(
    rows: pd.DataFrame,
    features: dict[str, pd.DataFrame],
    calibration: pd.Series,
    coverage: pd.Series,
    *,
    fallback_to_original: bool = False,
) -> tuple[dict, pd.DataFrame, dict[str, list[Pipeline]]]:
    holdout = rows["split"].eq("test")
    target = cast(pd.Series, rows["diagnosis_code"].eq("MEL"))
    usable = coverage.gt(0)
    eligible = ~holdout & ~calibration
    if not fallback_to_original:
        eligible &= usable
    folds = (1, 2, 3)
    training = [cast(pd.Series, eligible & rows["validation_fold"].ne(fold).fillna(False)) for fold in folds]
    if any(target[selected].nunique() != 2 for selected in training):
        raise ValueError("Every training fold needs both melanomas and nevi")
    metrics: dict = {
        "training_rows_per_member": {fold: int(selected.to_numpy().sum()) for fold, selected in zip(folds, training, strict=True)},
        "methods": {},
    }
    predictions = rows[["diagnosis_code", "validation_fold"]].join(coverage)
    heads = {}
    for method, matrix in features.items():
        with threadpool_limits(limits=4):
            heads[method], scores = score_ensemble(matrix, target, training)
        cutoff = cutoff_at_target(np.asarray(target[calibration], dtype=bool), np.asarray(scores[calibration]))
        metrics["methods"][method] = {"cutoff": cutoff}
        predictions[f"{method}_score"] = scores
        predictions[f"{method}_prediction"] = np.where(scores.isna(), "INCONCLUSIVE", np.where(scores.ge(cutoff), "MEL", "NV"))
        for name, split in (("calibration", calibration), ("holdout", holdout)):
            result = summarize(np.asarray(target[split], dtype=bool), np.asarray(scores[split]), cutoff)
            paired = split & usable
            paired_target = np.asarray(target[paired], dtype=bool)
            paired_scores = np.asarray(scores[paired])
            result["on_nonempty_masks"] = summarize(paired_target, paired_scores, cutoff)
            result["on_nonempty_masks"]["roc_auc"] = (
                float(roc_auc_score(paired_target, paired_scores)) if np.unique(paired_target).size == 2 else None
            )
            metrics["methods"][method][name] = result
            sensitivity, specificity = result["melanoma_sensitivity"], result["nevus_specificity"]
            print(
                f"{name} {method}: {sensitivity['count']}/{sensitivity['total']} MEL flagged, "
                f"{specificity['count']}/{specificity['total']} NV cleared",
                flush=True,
            )
    return metrics, predictions, heads


def evaluate(args: argparse.Namespace, device: torch.device) -> None:
    data_dir = args.data_dir.expanduser().resolve()
    checkpoint = args.segmentation_model.expanduser().resolve(strict=True)
    catalog = load_catalog(args.catalog, data_dir)
    assert catalog is not None
    parent, _ = prepare_routed_tasks(data_dir, catalog)
    rows = parent.loc[parent["diagnosis_code"].isin(["MEL", "NV"])]
    calibration = calibration_mask(parent).loc[rows.index]
    holdout = rows["split"].eq("test")
    use_context = args.comparison == "skin-context"
    variants = {"dimmed_background": dim_background, "padded_crop": crop_with_skin_margin} if use_context else {"masked": black_background}

    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_folder = f"{run}-skin-context" if use_context else run
    output = data_dir / "experiments" / "masked-mel-nv" / run_folder
    output.mkdir(parents=True)
    with checkpoint.open("rb") as weights:
        segmentation_sha256 = hashlib.file_digest(weights, "sha256").hexdigest()
    print(f"Run: {output}", flush=True)
    segmenter = LesionSegmenter(checkpoint, device)
    transformed, coverage = prepare_variant_photos(rows, data_dir, output, segmenter, variants)
    del segmenter
    usable = coverage.gt(0)
    coverage.to_csv(output / "mask_coverage.csv")
    print(f"Empty masks: {int((~usable).sum())}/{len(usable)} lesions", flush=True)
    if not usable.any():
        raise ValueError("The U-Net produced no usable masks for this comparison")

    # The full-catalog call can reuse the embeddings from the original clinical comparison.
    unmasked, provenance, reused = dermfm_zero.encode_clinical(parent, data_dir, device, args.batch_size)
    original_features = pd.DataFrame(unmasked, index=parent.index).loc[rows.index]
    features = {"unmasked": original_features}
    for method, inputs in transformed.items():
        print(f"Encoding {method}", flush=True)
        embeddings, _, _ = dermfm_zero.encode_clinical(inputs, data_dir, device, args.batch_size)
        if not np.isfinite(embeddings).all():
            raise ValueError(f"DermFM-Zero-Open produced non-finite {method} embeddings")
        matrix = pd.DataFrame(embeddings, index=inputs.index).reindex(rows.index)
        if use_context:
            matrix.loc[~usable] = original_features.loc[~usable]
        features[method] = matrix
    if not np.isfinite(unmasked).all():
        raise ValueError("DermFM-Zero-Open produced non-finite unmasked embeddings")
    comparison, predictions, heads = compare_heads(rows, features, calibration, coverage, fallback_to_original=use_context)
    metrics = {
        "catalog_sha256": catalog.sha256,
        "manifest_sha256": hashlib.sha256((data_dir / "splits" / FULL_MANIFEST_NAME).read_bytes()).hexdigest(),
        "encoder": provenance,
        "unmasked_feature_cache_hit": reused,
        "segmentation_checkpoint": str(checkpoint),
        "segmentation_checkpoint_sha256": segmentation_sha256,
        "segmentation_input_size": SEGMENTATION_SIZE,
        "comparison": args.comparison,
        "empty_mask_policy": "original_photo" if use_context else "inconclusive",
        "empty_masks": {
            split: int(np.count_nonzero(~usable & selected))
            for split, selected in (("training", ~calibration & ~holdout), ("calibration", calibration), ("holdout", holdout))
        },
        "calibration_seed": CALIBRATION_SEED,
        "calibration_fraction": CALIBRATION_FRACTION,
        "melanoma_sensitivity_target": PIPELINE_TARGET,
        "pretraining_isic_id_overlap": dermfm_zero.pretraining_id_overlap(rows, data_dir),
        **comparison,
    }
    if use_context:
        metrics["context_transforms"] = {
            "dimmed_background": {"skin_margin_px": SKIN_MARGIN_PX, "outside_brightness": BACKGROUND_BRIGHTNESS},
            "padded_crop": {"margin_fraction_of_longest_side": CROP_MARGIN_FRACTION, "minimum_margin_px": SKIN_MARGIN_PX},
        }

    predictions.loc[calibration].to_csv(output / "calibration_predictions.csv")
    predictions.loc[holdout].to_csv(output / "holdout_predictions.csv")
    joblib.dump(heads, output / "heads.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_dir = Path(os.environ.get("MILK10K_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--catalog", type=Path, default=Path("final_data.csv"))
    parser.add_argument("--segmentation-model", type=Path, required=True, help="Path to the demo's U-Net .pth file")
    parser.add_argument("--comparison", choices=["black-background", "skin-context"], default="black-background")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    if args.batch_size < 1 or not args.segmentation_model.expanduser().is_file():
        parser.error("Pass an existing U-Net checkpoint file and a positive batch size")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    evaluate(args, device)


if __name__ == "__main__":
    main()
