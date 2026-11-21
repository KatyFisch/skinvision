"""Train an augmented clinical MEL-vs-NV model and compare it to the saved baseline."""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageEnhance, ImageOps
from torch.utils.data import DataLoader

from classifiers.encoders import CHECKPOINTS, build_encoder, encode_pairs, load_checkpoint
from classifiers.finetune_mel_nv import ENCODER, LesionImages, LesionResNet18, checkpoint_path, predict, train_last_block
from classifiers.train_mel_nv import MANIFEST, TARGET_BY_DIAGNOSIS, fit_frozen_head, load_development, summarize, summarize_at_threshold

IMAGE_COLUMNS = ["clinical_path"]
VARIANT = "fine_tuned"
BRIGHTNESS_RANGE = (0.9, 1.1)
CONTRAST_RANGE = (0.9, 1.1)
FLIP_PROBABILITY = 0.5


def create_assets(development: pd.DataFrame, data_dir: Path, asset_dir: Path, epochs: int, seed: int) -> list[pd.DataFrame]:
    rng = np.random.default_rng(seed)
    source_columns = ["lesion_id", "group_id", "validation_fold", "diagnosis_code", "target", "clinical_path"]
    epoch_rows = []
    for epoch in range(1, epochs + 1):
        epoch_dir = asset_dir / f"epoch-{epoch}"
        epoch_dir.mkdir(parents=True)
        records = []
        for lesion_id, group_id, fold, diagnosis_code, target, source_path in development[source_columns].itertuples(
            index=False, name=None
        ):
            brightness = rng.uniform(*BRIGHTNESS_RANGE)
            contrast = rng.uniform(*CONTRAST_RANGE)
            flipped = rng.random() < FLIP_PROBABILITY
            path = epoch_dir / f"{lesion_id}.png"
            with Image.open(data_dir / source_path) as source:
                image = ImageOps.exif_transpose(source).convert("RGB")
                image = ImageEnhance.Brightness(image).enhance(brightness)
                image = ImageEnhance.Contrast(image).enhance(contrast)
                if flipped:
                    image = ImageOps.mirror(image)
                image.save(path, compress_level=1)
            records.append(
                {
                    "lesion_id": lesion_id,
                    "group_id": group_id,
                    "validation_fold": fold,
                    "diagnosis_code": diagnosis_code,
                    "target": target,
                    "source_path": source_path,
                    "clinical_path": path.relative_to(data_dir).as_posix(),
                    "epoch": epoch,
                    "brightness": brightness,
                    "contrast": contrast,
                    "horizontal_flip": flipped,
                }
            )
        epoch_rows.append(pd.DataFrame(records, index=development.index))
        print(f"Saved {len(development)} augmented photos for epoch {epoch}/{epochs}", flush=True)
    pd.concat(epoch_rows, ignore_index=True).to_csv(asset_dir / "assets.csv", index=False)
    return epoch_rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parents[2] / "data"
    parser.add_argument("baseline_run", type=Path, help="Saved clinical-only fine-tuning run with epoch-five checkpoints")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("MILK10K_DATA_DIR", default_dir)))
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], help="Defaults to the device used by the baseline")
    args = parser.parse_args()

    data_dir = args.data_dir.expanduser().resolve()
    baseline_dir = args.baseline_run.expanduser().resolve()
    baseline = json.loads((baseline_dir / "metrics.json").read_text())
    weights, checksum = CHECKPOINTS[ENCODER]
    preprocess = weights.transforms()
    if (
        baseline["manifest"] != MANIFEST
        or baseline["image_columns"] != IMAGE_COLUMNS
        or baseline["pass_2"]["checkpoint_epochs"].get(VARIANT) != 5
        or baseline["encoder"]["sha256"] != checksum
        or baseline["encoder"]["preprocessing"] != str(preprocess)
        or "catalog_sha256" in baseline
    ):
        raise ValueError("Expected the original manifest-backed, five-epoch clinical ResNet-18 baseline")
    device = torch.device(args.device or baseline["encoder"]["device"])
    torch.set_num_threads(min(4, torch.get_num_threads()))
    batch_size = baseline["batch_size_pairs"]
    sensitivity_target = baseline["exploratory_sensitivity_target"]
    seed = baseline["seed"]
    recipe = baseline["pass_2"]
    epochs = recipe["epochs"]

    manifest_path = data_dir / "splits" / MANIFEST
    development, manifest_hash = load_development(manifest_path)
    if manifest_hash != baseline["manifest_sha256"]:
        raise ValueError("The frozen split differs from the source run")
    prediction_columns = ["lesion_id", "diagnosis_code", "target", "validation_fold"]
    published = pd.read_csv(baseline_dir / "validation_predictions.csv")
    development_predictions = (
        development[prediction_columns]
        .merge(published[[*prediction_columns, f"{VARIANT}_score"]], on=prediction_columns, validate="one_to_one")
        .rename(columns={f"{VARIANT}_score": "baseline_score"})
    )
    if len(development_predictions) != len(development) or len(published) != len(development):
        raise ValueError("Baseline predictions do not match the frozen development lesions and folds")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    asset_dir = data_dir / "augmented" / "milk10k-clinical-mel-nv" / run_id
    run_dir = data_dir / "experiments" / "mel-nv" / f"{run_id}-resnet18-clinical-augmentation"
    run_dir.mkdir(parents=True)
    print(f"Saving training images to {asset_dir}", flush=True)
    epoch_rows = create_assets(development, data_dir, asset_dir, epochs, seed)
    pretrained = load_checkpoint(weights.url, checksum, data_dir)
    features, encoder_settings, _ = encode_pairs(ENCODER, development, data_dir, device, batch_size, image_columns=IMAGE_COLUMNS)
    torch.manual_seed(seed)
    targets = development["target"].to_numpy(dtype=int)
    folds = development["validation_fold"].to_numpy(dtype=int)
    fold_numbers = sorted(set(folds.tolist()))
    development_scores = np.empty(len(development))
    losses = {}

    for fold in fold_numbers:
        validation = folds == fold
        print(f"Training fold {fold} on {(~validation).sum()} lesions per epoch", flush=True)
        head = fit_frozen_head(features[~validation], targets[~validation])
        model = LesionResNet18(build_encoder(ENCODER, pretrained, device), views=1)
        model.load_frozen_head(head)
        model.to(device)
        generator = torch.Generator().manual_seed(seed + fold)

        training_batches = [
            DataLoader(
                LesionImages(rows.loc[~validation], data_dir, preprocess, IMAGE_COLUMNS),
                batch_size=batch_size,
                shuffle=True,
                generator=generator,
            )
            for rows in epoch_rows
        ]
        losses[fold] = list(train_last_block(model, training_batches, device, epochs, recipe["head_lr"], recipe["block_lr"]))
        validation_images = LesionImages(development.loc[validation], data_dir, preprocess, IMAGE_COLUMNS)
        development_scores[validation] = predict(model, DataLoader(validation_images, batch_size=batch_size), device)
        torch.save({name: value.cpu() for name, value in model.state_dict().items()}, checkpoint_path(run_dir, fold, VARIANT))

    development_predictions["augmented_score"] = development_scores
    development_results = {
        "baseline": summarize_at_threshold(
            targets, development_predictions["baseline_score"].to_numpy(), baseline["results"][VARIANT]["threshold"]
        ),
        "augmented": summarize(targets, development_scores, sensitivity_target),
    }
    development_predictions.to_csv(run_dir / "validation_predictions.csv", index=False)
    validation_report = {
        "baseline_run": baseline_dir.name,
        "checkpoint_variant": VARIANT,
        "manifest": MANIFEST,
        "manifest_sha256": manifest_hash,
        "assets": asset_dir.relative_to(data_dir).as_posix(),
        "augmentation": {"brightness": BRIGHTNESS_RANGE, "contrast": CONTRAST_RANGE, "horizontal_flip_probability": FLIP_PROBABILITY},
        "encoder": encoder_settings,
        "seed": seed,
        "batch_size": batch_size,
        "epochs": epochs,
        "head_lr": recipe["head_lr"],
        "block_lr": recipe["block_lr"],
        "sensitivity_target": sensitivity_target,
        "training_loss_by_fold": losses,
        "results": development_results,
    }
    (run_dir / "validation_metrics.json").write_text(json.dumps(validation_report, indent=2) + "\n", encoding="utf-8")

    # Read and score the original test photos after both operating cutoffs have been saved.
    content = manifest_path.read_bytes()
    if hashlib.sha256(content).hexdigest() != manifest_hash:
        raise ValueError("The frozen split changed during training")
    manifest = pd.read_csv(BytesIO(content))
    test = manifest.loc[manifest["split"].eq("test")].sort_values("lesion_id")
    if set(test["target"]) != {0, 1} or not test["target"].eq(test["diagnosis_code"].map(TARGET_BY_DIAGNOSIS)).all():
        raise ValueError("Expected both diagnoses and matching binary labels in the test holdout")
    test_batches = DataLoader(LesionImages(test, data_dir, preprocess, IMAGE_COLUMNS), batch_size=batch_size)
    test_predictions = test[["lesion_id", "group_id", "diagnosis_code", "target"]].copy()
    test_targets = test["target"].to_numpy(dtype=int)
    test_results = {}
    for name, source in (("baseline", baseline_dir), ("augmented", run_dir)):
        print(f"Scoring {len(test)} original holdout photos with the {name} fold models", flush=True)
        fold_scores = []
        for fold in fold_numbers:
            model.load_state_dict(torch.load(checkpoint_path(source, fold, VARIANT), map_location="cpu", weights_only=True))
            fold_scores.append(predict(model, test_batches, device))
        scores = np.mean(fold_scores, axis=0, dtype=np.float64)
        test_predictions[f"{name}_score"] = scores
        test_results[name] = {
            "development_cutoff": summarize_at_threshold(test_targets, scores, development_results[name]["threshold"]),
            "exploratory_matched_sensitivity": summarize(test_targets, scores, sensitivity_target),
        }
    test_predictions.to_csv(run_dir / "test_predictions.csv", index=False)
    test_report = {"manifest_sha256": manifest_hash, "sensitivity_target": sensitivity_target, "results": test_results}
    (run_dir / "test_metrics.json").write_text(json.dumps(test_report, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {run_dir}", flush=True)


if __name__ == "__main__":
    main()
