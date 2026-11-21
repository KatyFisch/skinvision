"""Measure whether ISIC 2017 training data improve a dermoscopy-only MEL/NV classifier."""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import torch
import torchvision
from torch.utils.data import DataLoader

from classifiers.catalog import ImageCatalog, load_catalog
from classifiers.encoders import CHECKPOINTS, build_encoder, load_checkpoint
from classifiers.eval_split.split_isic2017 import MANIFEST as ISIC_MANIFEST
from classifiers.finetune_mel_nv import (
    BLOCK_LR,
    ENCODER,
    EPOCHS,
    HEAD_LR,
    LesionImages,
    LesionResNet18,
    encode_images,
    predict,
    train_last_block,
)
from classifiers.train_mel_nv import MANIFEST as MILK_MANIFEST
from classifiers.train_mel_nv import TARGET_BY_DIAGNOSIS, fit_frozen_head, load_development, summarize, summarize_at_threshold

SEED = 42
BATCH_SIZE = 16
TARGET_SENSITIVITY = 0.95
RECIPES = ("milk_only", "milk_plus_isic")


def load_isic(data_dir: Path) -> tuple[pd.DataFrame, str]:
    content = (data_dir / "splits" / ISIC_MANIFEST).read_bytes()
    manifest = pd.read_csv(BytesIO(content))
    usable = manifest.loc[manifest.split.isin(["train", "validation", "test"])].copy()
    partitions = manifest[["group_id", "split"]].drop_duplicates()
    if not manifest.image_id.is_unique or partitions.group_id.isna().any() or not partitions.group_id.is_unique:
        raise ValueError("Expected unique image IDs and duplicate groups confined to one ISIC split")
    if not usable.target.eq(usable.diagnosis_code.map(TARGET_BY_DIAGNOSIS)).fillna(False).all():
        raise ValueError("Expected binary labels MEL=1 and NV=0 in the ISIC manifest")
    if set(usable.split) != {"train", "validation", "test"} or not usable.groupby("split").target.nunique().eq(2).all():
        raise ValueError("Expected training, validation, and test with both diagnoses in each ISIC split")
    return usable, hashlib.sha256(content).hexdigest()


def save_json(path: Path, content: dict) -> None:
    path.write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")


def run_training(data_dir: Path, device: torch.device, catalog: ImageCatalog | None = None) -> None:
    milk, milk_digest = load_development(data_dir / "splits" / MILK_MANIFEST, catalog)
    isic, isic_digest = load_isic(data_dir)
    isic = isic.loc[isic.split.ne("test")]
    if catalog is not None:
        isic = catalog.isic_images(isic)
    milk = milk.rename(columns={"lesion_id": "image_id", "dermoscopic_path": "image_path"}).assign(source="MILK10k")
    isic = isic.assign(source="ISIC2017")
    rows = pd.concat([milk, isic], ignore_index=True)
    is_milk = rows.source.eq("MILK10k").to_numpy()
    milk_folds = rows.validation_fold.fillna(0).to_numpy(dtype=int)
    isic_training = (~is_milk) & rows.split.eq("train").to_numpy()
    isic_validation = (~is_milk) & rows.split.eq("validation").to_numpy()
    targets = rows.target.to_numpy(dtype=int)
    folds = sorted(set(milk_folds[is_milk].tolist()))
    weights, checksum = CHECKPOINTS[ENCODER]
    pretrained = load_checkpoint(weights.url, checksum, data_dir)
    preprocess = weights.transforms()
    images = LesionImages(rows, data_dir, preprocess, ["image_path"])
    torch.manual_seed(SEED)
    initial = LesionResNet18(build_encoder(ENCODER, pretrained, device), views=1).to(device)
    features = encode_images(initial, DataLoader(images, batch_size=BATCH_SIZE), device)
    del initial

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    run_dir = data_dir / "experiments" / "mel-nv" / f"{run_id}-isic2017-dermoscopy"
    run_dir.mkdir(parents=True)
    validation_results = {}
    predictions = rows.loc[is_milk | isic_validation, ["source", "image_id", "target", "validation_fold"]].copy()
    for recipe in RECIPES:
        combined = recipe == "milk_plus_isic"
        milk_scores = np.empty(int(is_milk.sum()))
        isic_scores = []
        training_counts = {}
        losses = {}
        for fold in folds:
            milk_training = is_milk & (milk_folds != fold)
            training = milk_training | isic_training if combined else milk_training
            milk_validation = is_milk & (milk_folds == fold)
            scoring = training | milk_validation | isic_validation
            training_counts[fold] = int(training.sum())
            print(f"{recipe}, fold {fold}: {training_counts[fold]} training images", flush=True)
            head = fit_frozen_head(features[training], targets[training])
            model = LesionResNet18(build_encoder(ENCODER, pretrained, device), views=1)
            model.load_frozen_head(head)
            model.to(device)
            generator = torch.Generator().manual_seed(SEED + fold)
            batches = DataLoader(
                LesionImages(rows.loc[training], data_dir, preprocess, ["image_path"]),
                batch_size=BATCH_SIZE,
                shuffle=True,
                generator=generator,
            )
            losses[fold] = list(train_last_block(model, batches, device, EPOCHS, HEAD_LR, BLOCK_LR))
            scoring_batches = DataLoader(LesionImages(rows.loc[scoring], data_dir, preprocess, ["image_path"]), batch_size=BATCH_SIZE)
            tuned_features = encode_images(model, scoring_batches, device)
            refit = fit_frozen_head(tuned_features[training[scoring]], targets[training])
            model.load_frozen_head(refit)
            with torch.inference_mode():
                scores = model.logits_from_embeddings(torch.from_numpy(tuned_features).to(device)).sigmoid().cpu().numpy()
            milk_scores[milk_folds[is_milk] == fold] = scores[milk_validation[scoring]]
            isic_scores.append(scores[isic_validation[scoring]])
            torch.save({name: value.cpu() for name, value in model.state_dict().items()}, run_dir / f"{recipe}-fold-{fold}.pt")

        isic_mean = np.mean(isic_scores, axis=0, dtype=np.float64)
        predictions[f"{recipe}_score"] = np.concatenate((milk_scores, isic_mean))
        validation_results[recipe] = {
            "milk_development": summarize(targets[is_milk], milk_scores, TARGET_SENSITIVITY),
            "isic_validation": summarize(targets[isic_validation], isic_mean, TARGET_SENSITIVITY),
            "training_images_by_fold": training_counts,
            "training_loss_by_fold": losses,
        }

    predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    save_json(
        run_dir / "validation_metrics.json",
        {
            "milk_manifest_sha256": milk_digest,
            "isic_manifest_sha256": isic_digest,
            **({"catalog_sha256": catalog.sha256} if catalog is not None else {}),
            "checkpoint_sha256": checksum,
            "preprocess": str(preprocess),
            "versions": {"torch": torch.__version__, "torchvision": torchvision.__version__, "scikit_learn": sklearn.__version__},
            "device": str(device),
            "seed": SEED,
            "batch_size": BATCH_SIZE,
            "epochs": EPOCHS,
            "head_lr": HEAD_LR,
            "block_lr": BLOCK_LR,
            "folds": folds,
            "sensitivity_target": TARGET_SENSITIVITY,
            "results": validation_results,
        },
    )
    print(f"Validation results and locked cutoffs: {run_dir}")


def run_test(
    data_dir: Path, run_dir: Path, device: torch.device, *, include_milk_test: bool = False, catalog_path: Path | None = None
) -> None:
    settings = json.loads((run_dir / "validation_metrics.json").read_text())
    catalog = load_catalog(catalog_path, data_dir, expected_sha256=settings.get("catalog_sha256"))
    isic, isic_digest = load_isic(data_dir)
    if isic_digest != settings["isic_manifest_sha256"]:
        raise ValueError("ISIC manifest differs from the version used in training")
    weights, checksum = CHECKPOINTS[ENCODER]
    if checksum != settings["checkpoint_sha256"] or str(weights.transforms()) != settings["preprocess"]:
        raise ValueError("The pretrained checkpoint configuration differs from the validation run")
    pretrained = load_checkpoint(weights.url, checksum, data_dir)
    test = isic.loc[isic.split.eq("test")].assign(source="ISIC2017")
    if catalog is not None:
        test = catalog.isic_images(test)
    if include_milk_test:
        milk_content = (data_dir / "splits" / MILK_MANIFEST).read_bytes()
        if hashlib.sha256(milk_content).hexdigest() != settings["milk_manifest_sha256"]:
            raise ValueError("MILK10k manifest differs from the version used in training")
        milk = pd.read_csv(BytesIO(milk_content))
        milk_test = milk.loc[milk.split.eq("test")]
        if catalog is not None:
            milk_test = catalog.milk_lesions(milk_test)
        milk_test = milk_test.rename(columns={"lesion_id": "image_id", "dermoscopic_path": "image_path"})
        test = pd.concat([test, milk_test.assign(source="MILK10k")], ignore_index=True)
    batches = DataLoader(LesionImages(test, data_dir, weights.transforms(), ["image_path"]), batch_size=settings["batch_size"])
    model = LesionResNet18(build_encoder(ENCODER, pretrained, device), views=1).to(device)
    targets = test.target.to_numpy(dtype=int)
    predictions = test[["image_id", "target", "group_id"]].copy()
    if include_milk_test:
        predictions.insert(0, "source", test.source)
    subsets = {"pooled": np.ones(len(test), dtype=bool), **{source: test.source.eq(source).to_numpy() for source in test.source.unique()}}
    results = {}
    for recipe in RECIPES:
        split_label = "ISIC and MILK10k test splits" if include_milk_test else "ISIC test split"
        print(f"Scoring {split_label}: {recipe}", flush=True)
        fold_scores = []
        for fold in settings["folds"]:
            model.load_state_dict(torch.load(run_dir / f"{recipe}-fold-{fold}.pt", map_location="cpu", weights_only=True))
            fold_scores.append(predict(model, batches, device))
        scores = np.mean(fold_scores, axis=0, dtype=np.float64)
        cutoff = settings["results"][recipe]["isic_validation"]["threshold"]
        summary = summarize_at_threshold(targets, scores, cutoff)
        if include_milk_test:
            results[recipe] = {
                "locked_cutoffs": {name: summarize_at_threshold(targets[mask], scores[mask], cutoff) for name, mask in subsets.items()},
                "exploratory_matched_sensitivity": {
                    name: summarize(targets[mask], scores[mask], settings["sensitivity_target"]) for name, mask in subsets.items()
                },
            }
        else:
            results[recipe] = summary
        print(
            f"  AUC {summary['roc_auc']:.3f}, sensitivity {summary['sensitivity']:.1%}, specificity {summary['specificity']:.1%}",
            flush=True,
        )
        predictions[f"{recipe}_score"] = scores
    prefix = "combined_test" if include_milk_test else "isic_test"
    predictions.to_csv(run_dir / f"{prefix}_predictions.csv", index=False)
    report = {"isic_manifest_sha256": isic_digest, "results": results}
    if catalog is not None:
        report["catalog_sha256"] = catalog.sha256
    if include_milk_test:
        report.update(milk_manifest_sha256=settings["milk_manifest_sha256"], sensitivity_target=settings["sensitivity_target"])
    save_json(run_dir / f"{prefix}_metrics.json", report)
    print(f"Test results: {run_dir / f'{prefix}_metrics.json'}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("MILK10K_DATA_DIR", default_dir)))
    parser.add_argument("--catalog", type=Path, help="Image catalog CSV, relative to --data-dir or absolute")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--test-run", type=Path, help="Apply saved checkpoints and validation cutoffs from this run to the ISIC test split")
    parser.add_argument("--include-milk-test", action="store_true", help="Also score the local MILK10k holdout and report pooled results")
    args = parser.parse_args()
    if args.include_milk_test and not args.test_run:
        parser.error("--include-milk-test requires --test-run")
    data_dir = args.data_dir.expanduser().resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    if args.test_run:
        run_test(
            data_dir, args.test_run.expanduser().resolve(), device, include_milk_test=args.include_milk_test, catalog_path=args.catalog
        )
    else:
        run_training(data_dir, device, load_catalog(args.catalog, data_dir))


if __name__ == "__main__":
    main()
