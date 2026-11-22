"""Compare single-photo melanoma heads on frozen MILK10k clinical embeddings."""

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from threadpoolctl import threadpool_limits

from classifiers import dermfm_zero, panderm
from classifiers.catalog import MILK_DIAGNOSES, load_catalog
from classifiers.encoders import encode_pairs
from classifiers.eval_split.create_splits import FULL_MANIFEST_NAME
from classifiers.eval_split.prepare_routed_tasks import prepare_routed_tasks
from classifiers.single_photo import OTHER_CLASSES, PIPELINE_TARGET, ROUTER_TARGET, Cutoffs, decisions, lock_cutoffs, rate
from classifiers.train_mel_nv import fit_frozen_head

CALIBRATION_FRACTION = 0.20
CALIBRATION_SEED = 42
DIAGNOSES = np.array(MILK_DIAGNOSES)
METHODS = ("cascade", "mel_vs_rest", "direct_11")


def calibration_mask(rows: pd.DataFrame) -> pd.Series:
    """Reserve diagnosis-stratified groups inside each frozen development fold."""
    development = rows.loc[rows["split"].eq("train")]
    groups = development.groupby("group_id").agg(
        validation_fold=("validation_fold", "first"), diagnoses=("diagnosis_code", lambda codes: tuple(sorted(set(codes))))
    )
    rng = np.random.default_rng(CALIBRATION_SEED)
    reserved = []
    for _, stratum in groups.groupby(["validation_fold", "diagnoses"]):
        count = int(len(stratum) * CALIBRATION_FRACTION)
        reserved.extend(rng.choice(stratum.index.to_numpy(), size=count, replace=False))
    return cast(pd.Series, rows["group_id"].isin(reserved))


def fit_ensembles(rows: pd.DataFrame, features: np.ndarray, calibration: pd.Series) -> dict[str, list[Pipeline]]:
    diagnosis = rows["diagnosis_code"].to_numpy()
    eligible = rows["split"].eq("train") & ~calibration
    routed = np.isin(diagnosis, ["MEL", "NV"])
    ensembles: dict[str, list[Pipeline]] = {name: [] for name in ("router", "specialist", "other_diagnoses", "mel_vs_rest", "direct_11")}
    for fold in sorted(rows.loc[eligible, "validation_fold"].unique()):
        train = (eligible & rows["validation_fold"].ne(fold).fillna(False)).to_numpy(dtype=bool)
        targets = {
            "router": (train, routed),
            "specialist": (train & routed, diagnosis == "MEL"),
            "other_diagnoses": (train & ~routed, diagnosis),
            "mel_vs_rest": (train, diagnosis == "MEL"),
            "direct_11": (train, diagnosis),
        }
        for name, (selected, labels) in targets.items():
            head = fit_frozen_head(features[selected], labels[selected])
            expected = {"other_diagnoses": OTHER_CLASSES, "direct_11": DIAGNOSES}.get(name, np.array([False, True]))
            if not np.array_equal(head.classes_, np.sort(expected)):
                raise ValueError(f"Training outside fold {fold} is missing a class for {name}")
            ensembles[name].append(head)
    return ensembles


def ensemble_scores(rows: pd.DataFrame, features: np.ndarray, ensembles: dict[str, list[Pipeline]]) -> pd.DataFrame:
    means = {name: np.mean([member.predict_proba(features) for member in members], axis=0) for name, members in ensembles.items()}
    scores = rows[["diagnosis_code", "validation_fold"]].assign(
        router_score=means["router"][:, 1],
        melanoma_score=means["specialist"][:, 1],
        other_prediction=ensembles["other_diagnoses"][0].classes_[means["other_diagnoses"].argmax(axis=1)],
        mel_vs_rest=means["mel_vs_rest"][:, 1],
    )
    direct = pd.DataFrame(means["direct_11"], index=rows.index, columns=ensembles["direct_11"][0].classes_)
    return scores.join(direct[DIAGNOSES].add_prefix("direct_"))


def sensitivity_cutoff(positive_scores: np.ndarray) -> float:
    return float(np.sort(positive_scores)[-math.ceil(len(positive_scores) * PIPELINE_TARGET)])


def classify(scores: pd.DataFrame, cutoffs: dict[str, float]) -> pd.DataFrame:
    cascade = decisions(scores, Cutoffs(cutoffs["router"], cutoffs["specialist"]))
    direct = scores.filter(items=[f"direct_{label}" for label in DIAGNOSES]).rename(columns=lambda name: name.removeprefix("direct_"))
    direct_flag = direct["MEL"].ge(cutoffs["direct_11"])
    return scores.assign(
        to_specialist=cascade["to_specialist"],
        cascade_prediction=cascade["prediction"],
        mel_vs_rest_flag=scores["mel_vs_rest"].ge(cutoffs["mel_vs_rest"]),
        direct_11_prediction=direct.drop(columns="MEL").idxmax(axis=1).where(~direct_flag, "MEL"),
        direct_11_argmax=direct.idxmax(axis=1),
    )


def diagnosis_report(actual: np.ndarray, predicted: np.ndarray) -> dict:
    report = classification_report(actual, predicted, labels=DIAGNOSES, output_dict=True, zero_division=0)  # pyright: ignore[reportArgumentType]
    assert isinstance(report, dict)
    return report


def summarize(rows: pd.DataFrame) -> dict:
    actual = np.asarray(rows["diagnosis_code"], dtype=str)
    melanoma = actual == "MEL"
    route = np.asarray(rows["to_specialist"], dtype=bool)
    order = {
        "cascade": np.where(route, np.asarray(rows["melanoma_score"]), -1.0),
        "mel_vs_rest": np.asarray(rows["mel_vs_rest"]),
        "direct_11": np.asarray(rows["direct_MEL"]),
    }
    flags = {
        "cascade": np.asarray(rows["cascade_prediction"]) == "MEL",
        "mel_vs_rest": np.asarray(rows["mel_vs_rest_flag"], dtype=bool),
        "direct_11": np.asarray(rows["direct_11_prediction"]) == "MEL",
    }
    metrics = {}
    for method in METHODS:
        flagged = flags[method]
        score = order[method]
        target = sensitivity_cutoff(score[melanoma])
        matched = (score >= target) if target >= 0 else None
        metrics[method] = {
            "sensitivity": rate(int((melanoma & flagged).sum()), int(melanoma.sum())),
            "specificity": rate(int((~melanoma & ~flagged).sum()), int((~melanoma).sum())),
            "roc_auc": float(roc_auc_score(melanoma, score)),
            "average_precision": float(average_precision_score(melanoma, score)),
            "matched_95_specificity": rate(int((~melanoma & ~matched).sum()), int((~melanoma).sum())) if matched is not None else None,
            "false_positives_by_diagnosis": pd.Series(actual[~melanoma & flagged])
            .value_counts()
            .reindex(DIAGNOSES, fill_value=0)
            .to_dict(),
        }
        if method != "mel_vs_rest":
            predicted = np.asarray(rows[f"{method}_prediction"], dtype=str)
            metrics[method]["classification"] = diagnosis_report(actual, predicted)
            metrics[method]["confusion_matrix"] = confusion_matrix(actual, predicted, labels=DIAGNOSES).tolist()
    metrics["cascade"]["melanomas_missed_at_router"] = int((melanoma & ~route).sum())
    metrics["direct_11"]["argmax_classification"] = diagnosis_report(actual, np.asarray(rows["direct_11_argmax"], dtype=str))
    return metrics


def evaluate(args: argparse.Namespace, device: torch.device) -> None:
    data_dir = args.data_dir.expanduser().resolve()
    catalog = load_catalog(args.catalog, data_dir)
    assert catalog is not None
    rows, _ = prepare_routed_tasks(data_dir, catalog)
    calibration = calibration_mask(rows)
    holdout = rows["split"].eq("test")

    pretraining_audit = {}
    if args.encoder == dermfm_zero.NAME:
        pretraining_audit = {"pretraining_isic_id_overlap": dermfm_zero.pretraining_id_overlap(rows, data_dir)}
        features, provenance, cache_hit = dermfm_zero.encode_clinical(rows, data_dir, device, args.batch_size)
    elif args.encoder == panderm.NAME:
        features, provenance, cache_hit = panderm.encode_clinical(rows, data_dir, device, args.batch_size)
    else:
        features, provenance, cache_hit = encode_pairs(
            args.encoder, rows, data_dir, device, args.batch_size, image_columns=["clinical_path"], full_frame=True, native_resolution=True
        )
    if not np.isfinite(features).all():
        raise ValueError("The frozen embeddings contain non-finite values")
    with threadpool_limits(limits=4):
        ensembles = fit_ensembles(rows, features, calibration)
        calibrate = ensemble_scores(rows.loc[calibration], features[calibration], ensembles)
        melanoma = calibrate.loc[calibrate["diagnosis_code"].eq("MEL")]
        cutoffs = {
            **asdict(lock_cutoffs(calibrate)),
            "mel_vs_rest": sensitivity_cutoff(np.asarray(melanoma["mel_vs_rest"])),
            "direct_11": sensitivity_cutoff(np.asarray(melanoma["direct_MEL"])),
        }
        test = ensemble_scores(rows.loc[holdout], features[holdout], ensembles)
    run = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = data_dir / "experiments" / "clinical-comparison" / f"{run}-{args.encoder}"
    output.mkdir(parents=True)
    metrics = {
        "catalog_sha256": catalog.sha256,
        "manifest_sha256": hashlib.sha256((data_dir / "splits" / FULL_MANIFEST_NAME).read_bytes()).hexdigest(),
        "diagnosis_order": MILK_DIAGNOSES,
        "encoder": args.encoder,
        "encoder_provenance": provenance,
        "feature_dimensions": features.shape[1],
        "feature_cache_hit": cache_hit,
        "calibration_seed": CALIBRATION_SEED,
        "calibration_fraction": CALIBRATION_FRACTION,
        "training_rows_per_member": {
            int(fold): int((~calibration & ~holdout & rows["validation_fold"].ne(fold)).sum()) for fold in (1, 2, 3)
        },
        "targets": {"router": ROUTER_TARGET, "end_to_end": PIPELINE_TARGET},
        "cutoffs": cutoffs,
        **pretraining_audit,
    }
    for name, scores in (("calibration", calibrate), ("holdout", test)):
        scored = classify(scores, cutoffs)
        metrics[name] = {"lesions": len(scored), "methods": summarize(scored)}
        scored.to_csv(output / f"{name}_predictions.csv")
        for method, result in metrics[name]["methods"].items():
            sensitivity, specificity = result["sensitivity"], result["specificity"]
            print(
                f"{name} {method}: {sensitivity['count']}/{sensitivity['total']} melanomas flagged, "
                f"{specificity['rate']:.1%} specificity, AUROC {result['roc_auc']:.3f}",
                flush=True,
            )
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    joblib.dump(ensembles, output / "heads.joblib")
    print(f"Results: {output}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_dir = Path(os.environ.get("MILK10K_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    parser.add_argument("--data-dir", type=Path, default=default_data_dir)
    parser.add_argument("--catalog", type=Path, default=Path("final_data.csv"))
    parser.add_argument("--encoder", choices=["resnet18", panderm.NAME, dermfm_zero.NAME], required=True)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    evaluate(args, device)


if __name__ == "__main__":
    main()
