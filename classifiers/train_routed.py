"""Train the MEL-or-NV router and nine-class head on paired MILK10k features."""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

from classifiers.catalog import load_catalog
from classifiers.encoders import encode_pairs
from classifiers.eval_split.create_splits import FULL_MANIFEST_NAME
from classifiers.eval_split.prepare_routed_tasks import OTHER_DIAGNOSIS_TARGETS, prepare_routed_tasks
from classifiers.train_mel_nv import fit_frozen_head

ENCODER = "resnet18"


def summarize_router(rows: pd.DataFrame, scores: np.ndarray, melanoma_sensitivity: float) -> dict:
    melanoma = rows["diagnosis_code"].eq("MEL").to_numpy()
    _, recall, thresholds = roc_curve(melanoma, scores, drop_intermediate=False)
    cutoff = float(thresholds[np.flatnonzero(recall >= melanoma_sensitivity)[0]])
    routed = scores >= cutoff
    diagnoses = {}
    for code in sorted(rows["diagnosis_code"].unique()):
        mask = rows["diagnosis_code"].eq(code).to_numpy()
        diagnoses[code] = {"lesions": int(mask.sum()), "sent_to_specialist": int((mask & routed).sum())}
    other = rows["target"].eq(0).to_numpy()
    return {
        "roc_auc": float(roc_auc_score(rows["target"], scores)),
        "threshold": cutoff,
        "melanoma_routing_sensitivity": float(routed[melanoma].mean()),
        "other_routed_correctly": int((other & ~routed).sum()),
        "other_routing_specificity": float((~routed[other]).mean()),
        "by_diagnosis": diagnoses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("MILK10K_DATA_DIR", default_dir)))
    parser.add_argument("--catalog", type=Path, default=Path("final_data.csv"), help="Relative to --data-dir or absolute")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16, help="Lesion pairs per batch")
    parser.add_argument("--melanoma-sensitivity", type=float, default=0.95, help="Exploratory routing target for melanomas")
    args = parser.parse_args()
    if args.batch_size < 1 or not 0 < args.melanoma_sensitivity <= 1:
        parser.error("Batch size must be positive and melanoma sensitivity must be in (0, 1]")

    data_dir = args.data_dir.expanduser().resolve()
    catalog = load_catalog(args.catalog, data_dir)
    assert catalog is not None
    router, other = prepare_routed_tasks(data_dir, catalog)
    router = router.loc[router["split"].eq("train")]
    other = other.loc[other["split"].eq("train")]
    router_targets = router["target"].to_numpy(dtype=int)
    other_targets = other["target"].to_numpy(dtype=int)
    router_folds = router["validation_fold"].to_numpy(dtype=int)
    other_folds = other["validation_fold"].to_numpy(dtype=int)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    features, encoder, cache_hit = encode_pairs(ENCODER, router, data_dir, device, args.batch_size, full_frame=True, native_resolution=True)
    other_features = features[router.index.get_indexer(other.index)]

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = data_dir / "experiments" / "routed" / run_id
    output_dir.mkdir(parents=True)
    router_scores = np.empty(len(router), dtype=float)
    other_predictions = np.empty(len(other), dtype=int)
    other_majority = np.empty(len(other), dtype=int)
    for fold in sorted(set(router_folds.tolist())):
        route_validation = router_folds == fold
        route_head = fit_frozen_head(features[~route_validation], router_targets[~route_validation])
        router_scores[route_validation] = route_head.predict_proba(features[route_validation])[:, 1]
        joblib.dump(route_head, output_dir / f"router-fold-{fold}.joblib")

        other_validation = other_folds == fold
        training_targets = other_targets[~other_validation]
        other_head = fit_frozen_head(other_features[~other_validation], training_targets)
        other_predictions[other_validation] = other_head.predict(other_features[other_validation])
        other_majority[other_validation] = np.bincount(training_targets).argmax()
        joblib.dump(other_head, output_dir / f"other-fold-{fold}.joblib")
        print(f"Scored development fold {fold} for both tasks", flush=True)

    routing = summarize_router(router, router_scores, args.melanoma_sensitivity)
    class_names = list(OTHER_DIAGNOSIS_TARGETS)
    class_ids = list(OTHER_DIAGNOSIS_TARGETS.values())
    report = classification_report(
        other_targets, other_predictions, labels=class_ids, target_names=class_names, output_dict=True, zero_division=0
    )
    majority = classification_report(
        other_targets, other_majority, labels=class_ids, target_names=class_names, output_dict=True, zero_division=0
    )
    columns = ["diagnosis_code", "target", "validation_fold"]
    router[columns].assign(score=router_scores, sent_to_specialist=router_scores >= routing["threshold"]).to_csv(
        output_dir / "router_validation_predictions.csv"
    )
    other[columns].assign(predicted_target=other_predictions, predicted_diagnosis=np.asarray(class_names)[other_predictions]).to_csv(
        output_dir / "other_validation_predictions.csv"
    )
    manifest = data_dir / "splits" / FULL_MANIFEST_NAME
    metrics = {
        "catalog_sha256": catalog.sha256,
        "parent_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "native_resolution": True,
        "encoder": encoder,
        "feature_cache_hit": cache_hit,
        "scikit_learn": sklearn.__version__,
        "melanoma_sensitivity_target": args.melanoma_sensitivity,
        "router": routing,
        "other_diagnoses": {
            "class_order": class_names,
            "classification_report": report,
            "majority_baseline": majority,
            "confusion_matrix": confusion_matrix(other_targets, other_predictions, labels=class_ids).tolist(),
        },
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Router AUC {routing['roc_auc']:.3f}, melanoma routing sensitivity {routing['melanoma_routing_sensitivity']:.1%}")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
