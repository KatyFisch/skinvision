"""Evaluate the single-photo cascade or run a saved cascade on a local image."""

import argparse
import hashlib
import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from classifiers.catalog import MILK_DIAGNOSES, load_catalog
from classifiers.encoders import build_preprocess, load_image
from classifiers.eval_split.create_splits import FULL_MANIFEST_NAME
from classifiers.eval_split.prepare_routed_tasks import OTHER_DIAGNOSIS_TARGETS, prepare_routed_tasks
from classifiers.finetune_mel_nv import LesionImages, LesionResNet18, checkpoint_path
from classifiers.finetune_routed import RoutedResNet18
from classifiers.train_mel_nv import MANIFEST
from classifiers.train_routed import ENCODER

PHOTO_EDGE = 600
CLINICAL_VIEW = ["clinical_path"]
OTHER_CLASSES = np.array(list(OTHER_DIAGNOSIS_TARGETS))
ROUTER_TARGET = 0.99
PIPELINE_TARGET = 0.95
NORMALIZE = build_preprocess(ENCODER, full_frame=True, native_resolution=True)


@dataclass(frozen=True)
class Cutoffs:
    router: float
    specialist: float


def prepare_photo(image: Image.Image) -> torch.Tensor:
    """Match the training scale while keeping the complete frame and aspect ratio."""
    image.thumbnail((PHOTO_EDGE, PHOTO_EDGE), Image.Resampling.LANCZOS)
    return NORMALIZE(image)


def lock_cutoffs(scores: pd.DataFrame) -> Cutoffs:
    melanoma = scores.loc[scores["diagnosis_code"].eq("MEL")]
    if melanoma.empty:
        raise ValueError("Cutoff selection requires development melanomas")
    router_count = math.ceil(len(melanoma) * ROUTER_TARGET)
    final_count = math.ceil(len(melanoma) * PIPELINE_TARGET)
    router = float(melanoma["router_score"].nlargest(router_count).iloc[-1])
    reached = melanoma.loc[melanoma["router_score"].ge(router), "melanoma_score"]
    specialist = float(reached.nlargest(final_count).iloc[-1])
    return Cutoffs(router, specialist)


def decisions(scores: pd.DataFrame, cutoffs: Cutoffs) -> pd.DataFrame:
    to_specialist = scores["router_score"].ge(cutoffs.router)
    specialist = np.where(scores["melanoma_score"].ge(cutoffs.specialist), "MEL", "NV")
    return scores.assign(to_specialist=to_specialist, prediction=np.where(to_specialist, specialist, scores["other_prediction"]))


def rate(count: int, total: int) -> dict:
    if not total:
        return {"count": count, "total": total, "rate": None, "wilson_95_ci": None}
    p = count / total
    z = NormalDist().inv_cdf(0.975)
    denominator = 1 + z**2 / total
    midpoint = (p + z**2 / (2 * total)) / denominator
    half_width = z * math.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return {"count": count, "total": total, "rate": p, "wilson_95_ci": [max(0, midpoint - half_width), min(1, midpoint + half_width)]}


def summarize_pipeline(scored: pd.DataFrame) -> dict:
    actual = scored["diagnosis_code"]
    final = scored["prediction"]
    routed = scored["to_specialist"]
    melanoma = actual.eq("MEL")
    nevus = actual.eq("NV")
    other = ~(melanoma | nevus)
    detected = final.eq("MEL")
    return {
        "lesions": len(scored),
        "melanoma_sensitivity": rate(int((melanoma & detected).sum()), int(melanoma.sum())),
        "melanomas_missed_at_router": int((melanoma & ~routed).sum()),
        "melanomas_missed_at_specialist": int((melanoma & routed & ~detected).sum()),
        "melanoma_flag_specificity": rate(int((~melanoma & ~detected).sum()), int((~melanoma).sum())),
        "nevi_flagged_melanoma": rate(int((nevus & detected).sum()), int(nevus.sum())),
        "other_flagged_melanoma": rate(int((other & detected).sum()), int(other.sum())),
        "other_routed_to_specialist": rate(int((other & routed).sum()), int(other.sum())),
        "classification_report": classification_report(actual, final, labels=MILK_DIAGNOSES, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, final, labels=MILK_DIAGNOSES).tolist(),
    }


def load_models(weights: list[dict[str, str]], base: Path, device: torch.device) -> tuple[list[RoutedResNet18], list[LesionResNet18]]:
    def encoder() -> nn.Module:
        model = resnet18()
        model.fc = nn.Identity()
        return model

    routed, specialist = [], []
    for paths in weights:
        routing = RoutedResNet18(encoder(), views=1)
        detail = LesionResNet18(encoder(), views=1)
        routing.load_state_dict(torch.load(base / paths["routed"], map_location="cpu", weights_only=True))
        detail.load_state_dict(torch.load(base / paths["specialist"], map_location="cpu", weights_only=True))
        routed.append(routing.eval().to(device))
        specialist.append(detail.eval().to(device))
    return routed, specialist


@torch.inference_mode()
def routing_scores(models: list[RoutedResNet18], photos: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    fold_outputs = [model(photos) for model in models]
    route = torch.stack([router.sigmoid() for router, _ in fold_outputs]).mean(dim=0)
    other = torch.stack([diagnosis.softmax(dim=1) for _, diagnosis in fold_outputs]).mean(dim=0)
    return route.cpu().numpy(), OTHER_CLASSES[other.argmax(dim=1).cpu().numpy()]


@torch.inference_mode()
def specialist_scores(models: list[LesionResNet18], photos: torch.Tensor) -> np.ndarray:
    return torch.stack([model(photos).sigmoid() for model in models]).mean(dim=0).cpu().numpy()


def score_rows(
    rows: pd.DataFrame, models: tuple[list[RoutedResNet18], list[LesionResNet18]], data_dir: Path, device: torch.device, batch_size: int
) -> pd.DataFrame:
    batches = DataLoader(LesionImages(rows, data_dir, prepare_photo, CLINICAL_VIEW), batch_size=batch_size)
    route, other, melanoma = [], [], []
    for photos, _ in batches:
        photos = photos.to(device)
        router_scores, other_predictions = routing_scores(models[0], photos)
        route.append(router_scores)
        other.append(other_predictions)
        melanoma.append(specialist_scores(models[1], photos))
    return rows[["diagnosis_code", "validation_fold"]].assign(
        router_score=np.concatenate(route), melanoma_score=np.concatenate(melanoma), other_prediction=np.concatenate(other)
    )


def evaluate(args: argparse.Namespace, device: torch.device) -> None:
    data_dir = args.data_dir.expanduser().resolve()
    catalog = load_catalog(args.catalog, data_dir)
    assert catalog is not None
    routed_run = data_dir / args.routed_run.expanduser()
    specialist_run = data_dir / args.specialist_run.expanduser()
    for run, manifest, hash_field in (
        (routed_run, FULL_MANIFEST_NAME, "parent_manifest_sha256"),
        (specialist_run, MANIFEST, "manifest_sha256"),
    ):
        metadata = json.loads((run / "metrics.json").read_text())
        fingerprint = hashlib.sha256((data_dir / "splits" / manifest).read_bytes()).hexdigest()
        if metadata.get("catalog_sha256") != catalog.sha256 or metadata.get(hash_field) != fingerprint:
            raise ValueError(f"Catalog or split differs from the run in {run}")
        if metadata.get("image_columns") != CLINICAL_VIEW or not metadata.get("native_resolution"):
            raise ValueError(f"Expected an original-resolution clinical-only run in {run}")
        if run == routed_run and metadata["class_order"] != OTHER_CLASSES.tolist():
            raise ValueError("The routed run uses a different nine-class label order")

    rows, _ = prepare_routed_tasks(data_dir, catalog)
    development = rows.loc[rows["split"].eq("train")]
    folds = sorted(development["validation_fold"].unique())
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output = data_dir / "experiments" / "single-photo" / run_id
    output.mkdir(parents=True)
    paths = [
        {
            "routed": os.path.relpath((routed_run / f"fold-{fold}-refit.pt").resolve(strict=True), output),
            "specialist": os.path.relpath(checkpoint_path(specialist_run, fold, "refit").resolve(strict=True), output),
        }
        for fold in folds
    ]
    routed, specialist = load_models(paths, output, device)
    print(f"Scoring {len(development)} development photos with their held-out fold models", flush=True)
    calibration = pd.concat(
        score_rows(development.loc[development["validation_fold"].eq(fold)], ([routing], [detail]), data_dir, device, args.batch_size)
        for fold, routing, detail in zip(folds, routed, specialist, strict=True)
    ).sort_index()
    cutoffs = lock_cutoffs(calibration)
    config = {"weights": paths, "cutoffs": asdict(cutoffs)}
    (output / "model.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    print(f"Locked cutoffs in {output / 'model.json'}", flush=True)

    holdout = rows.loc[rows["split"].eq("test")]
    print(f"Scoring {len(holdout)} holdout photos with the three-fold ensemble", flush=True)
    test = score_rows(holdout, (routed, specialist), data_dir, device, args.batch_size)
    metrics = {
        "catalog_sha256": catalog.sha256,
        "diagnosis_order": MILK_DIAGNOSES,
        "development_targets": {"router_melanoma_sensitivity": ROUTER_TARGET, "end_to_end_melanoma_sensitivity": PIPELINE_TARGET},
        "cutoffs": asdict(cutoffs),
    }
    for name, scores in (("development", calibration), ("holdout", test)):
        scored = decisions(scores, cutoffs)
        scored.to_csv(output / f"{name}_predictions.csv")
        summary = summarize_pipeline(scored)
        metrics[name] = summary
        mel = summary["melanoma_sensitivity"]
        print(f"{name}: {mel['count']}/{mel['total']} melanomas flagged end to end", flush=True)
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output}", flush=True)


def predict_photo(model_path: Path, photo: Path, device: torch.device) -> dict:
    model_path = model_path.expanduser().resolve()
    config = json.loads(model_path.read_text())
    cutoffs = Cutoffs(**config["cutoffs"])
    image = load_image(photo.expanduser(), prepare_photo).unsqueeze(0).to(device)
    routed, specialist = load_models(config["weights"], model_path.parent, device)
    route, other = routing_scores(routed, image)
    melanoma = specialist_scores(specialist, image) if route[0] >= cutoffs.router else [float("nan")]
    scored = decisions(pd.DataFrame({"router_score": route, "melanoma_score": melanoma, "other_prediction": other}), cutoffs).iloc[0]
    return {"research_prediction": scored["prediction"], "route": "MEL_OR_NV" if scored["to_specialist"] else "OTHER"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("evaluate", help="Lock cutoffs on development folds and score the holdout")
    default_data_dir = Path(os.environ.get("MILK10K_DATA_DIR", Path(__file__).resolve().parents[1] / "data"))
    evaluation.add_argument("--data-dir", type=Path, default=default_data_dir)
    evaluation.add_argument("--catalog", type=Path, default=Path("final_data.csv"))
    evaluation.add_argument("--routed-run", type=Path, required=True, help="Clinical routed run, relative to --data-dir or absolute")
    evaluation.add_argument("--specialist-run", type=Path, default=Path("models/clinical-mel-nv"))
    evaluation.add_argument("--batch-size", type=int, default=16)
    prediction = commands.add_parser("predict", help="Run the locked cascade locally on a curated example")
    prediction.add_argument("--model", type=Path, required=True, help="Path to model.json from evaluation")
    prediction.add_argument("photo", type=Path, help="Path to a JPEG or PNG")
    for command in (evaluation, prediction):
        command.add_argument("--device", choices=["cpu", "mps", "cuda"])

    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.command == "evaluate":
        if args.batch_size < 1:
            parser.error("Batch size must be positive")
        evaluate(args, device)
    else:
        try:
            result = predict_photo(args.model, args.photo, device)
        except OSError as error:
            prediction.error(str(error))
        print(json.dumps(result))


if __name__ == "__main__":
    main()
