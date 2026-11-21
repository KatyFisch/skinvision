"""Compare frozen paired-image encoders on MEL-vs-NV development folds."""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from classifiers.catalog import ImageCatalog, load_catalog
from classifiers.encoders import CHECKPOINTS, encode_pairs

MANIFEST = "milk10k-mel-nv-v1.csv"
TARGET_BY_DIAGNOSIS = {"NV": 0, "MEL": 1}


def load_development(manifest_path: Path, catalog: ImageCatalog | None = None) -> tuple[pd.DataFrame, str]:
    content = manifest_path.read_bytes()
    manifest = pd.read_csv(BytesIO(content), dtype={"validation_fold": "Int64"})
    development = manifest.loc[manifest["split"].eq("train")].sort_values("lesion_id")
    if not manifest["lesion_id"].is_unique or set(development["validation_fold"]) != {1, 2, 3}:
        raise ValueError("Expected unique lesions and training folds 1, 2, and 3 in the frozen manifest")
    partitions = manifest[["group_id", "split", "validation_fold"]].drop_duplicates()
    if partitions["group_id"].isna().any() or not partitions["group_id"].is_unique:
        raise ValueError("Expected each MILK10k group to belong to one frozen split or development fold")
    if not development["target"].eq(development["diagnosis_code"].map(TARGET_BY_DIAGNOSIS)).all():
        raise ValueError("Expected binary labels MEL=1 and NV=0")
    fold_classes = development.groupby("validation_fold")["target"].nunique()
    if not fold_classes.eq(2).all():
        raise ValueError("Each validation fold must include both diagnoses")
    if catalog is not None:
        development = catalog.milk_lesions(development)
    return development, hashlib.sha256(content).hexdigest()


def fit_frozen_head(features: np.ndarray, targets: np.ndarray) -> Pipeline:
    return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(features, targets)


def predict_folds(features: np.ndarray, targets: np.ndarray, folds: np.ndarray) -> np.ndarray:
    scores = np.empty(len(targets))
    for fold in sorted(set(folds)):
        validation = folds == fold
        classifier = fit_frozen_head(features[~validation], targets[~validation])
        scores[validation] = classifier.predict_proba(features[validation])[:, 1]
    return scores


def summarize(targets: np.ndarray, scores: np.ndarray, sensitivity_target: float) -> dict[str, float | int]:
    _, sensitivity, thresholds = roc_curve(targets, scores, drop_intermediate=False)
    first_match = np.flatnonzero(sensitivity >= sensitivity_target)[0]
    threshold = float(thresholds[first_match])
    return summarize_at_threshold(targets, scores, threshold)


def summarize_at_threshold(targets: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float | int]:
    tn, fp, fn, tp = confusion_matrix(targets, scores >= threshold, labels=[0, 1]).ravel()
    return {
        "roc_auc": float(roc_auc_score(targets, scores)),
        "threshold": threshold,
        "sensitivity": float(tp / (tp + fn)),
        "specificity": float(tn / (tn + fp)),
        "melanomas_flagged": int(tp),
        "melanomas_missed": int(fn),
        "nevi_flagged": int(fp),
        "nevi_unflagged": int(tn),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_dir = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("MILK10K_DATA_DIR", default_data_dir)))
    parser.add_argument("--catalog", type=Path, help="Image catalog CSV, relative to --data-dir or absolute")
    parser.add_argument("--encoder", choices=CHECKPOINTS, nargs="+", default=list(CHECKPOINTS))
    parser.add_argument("--sensitivity-target", type=float, default=0.95, help="Exploratory development target (default: 0.95)")
    parser.add_argument("--batch-size", type=int, default=16, help="Number of lesion pairs per encoder batch")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], default=None)
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 < args.sensitivity_target <= 1:
        parser.error("Batch size must be positive and the sensitivity target must be in (0, 1]")

    data_dir = args.data_dir.expanduser().resolve()
    if args.device is None:
        args.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(args.device)
    torch.set_num_threads(min(4, torch.get_num_threads()))

    catalog = load_catalog(args.catalog, data_dir)
    development, manifest_hash = load_development(data_dir / "splits" / MANIFEST, catalog)
    targets = development["target"].to_numpy(dtype=int)
    folds = development["validation_fold"].to_numpy(dtype=int)
    predictions = development[["lesion_id", "diagnosis_code", "target", "validation_fold"]].copy()
    results = {}
    print(f"Comparing on {len(development)} development lesions using {device}", flush=True)
    for name in dict.fromkeys(args.encoder):
        start = perf_counter()
        features, settings, cache_hit = encode_pairs(name, development, data_dir, device, args.batch_size)
        feature_seconds = perf_counter() - start
        scores = predict_folds(features, targets, folds)
        predictions[f"{name}_score"] = scores
        summary = summarize(targets, scores, args.sensitivity_target)
        results[name] = {
            "encoder": settings,
            "feature_seconds": feature_seconds,
            "cache_hit": cache_hit,
            "paired_features": features.shape[1],
            "fold_roc_auc": {str(fold): float(roc_auc_score(targets[folds == fold], scores[folds == fold])) for fold in sorted(set(folds))},
            **summary,
        }
        print(f"{name}: AUC {summary['roc_auc']:.3f}, sensitivity {summary['sensitivity']:.1%}, specificity {summary['specificity']:.1%}")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = data_dir / "experiments" / "mel-nv" / run_id
    output_dir.mkdir(parents=True)
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    settings = {
        "manifest": MANIFEST,
        "manifest_sha256": manifest_hash,
        "classifier": "StandardScaler and LogisticRegression(C=1.0, max_iter=2000), fit within each fold",
        "scikit_learn": sklearn.__version__,
        "batch_size_pairs": args.batch_size,
        "exploratory_sensitivity_target": args.sensitivity_target,
        "results": results,
    }
    if catalog is not None:
        settings["catalog_sha256"] = catalog.sha256
    (output_dir / "metrics.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
