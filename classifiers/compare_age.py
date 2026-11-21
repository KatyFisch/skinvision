"""Compare image-only and image-plus-age heads on a frozen checkpoint from a fine-tuning run."""

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from classifiers.encoders import CHECKPOINTS, build_encoder, encode_pairs, load_checkpoint
from classifiers.finetune_mel_nv import ENCODER, LesionImages, LesionResNet18, checkpoint_path, encode_images
from classifiers.train_mel_nv import MANIFEST, fit_frozen_head, load_development, summarize

METADATA_FILE = "MILK10k_Training_Metadata.csv"
AGE_BANDS = ["under 50", "50–64", "65+", "unknown"]


def load_ages(data_dir: Path, development: pd.DataFrame) -> np.ndarray:
    metadata = pd.read_csv(data_dir / METADATA_FILE, usecols=["lesion_id", "image_type", "age_approx"])
    clinical_ages = metadata.loc[metadata.image_type.eq("clinical: close-up")].set_index("lesion_id")["age_approx"]
    if not clinical_ages.index.is_unique or not development.lesion_id.isin(clinical_ages.index).all():
        raise ValueError("Expected one clinical metadata row for each development lesion")
    return clinical_ages.loc[development.lesion_id].to_numpy(dtype=float)


def encode_checkpoint(checkpoint: Path, pretrained: Path, batches: DataLoader, device: torch.device) -> np.ndarray:
    model = LesionResNet18(build_encoder(ENCODER, pretrained, torch.device("cpu")))
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    return encode_images(model.to(device), batches, device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_dir = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("run_dir", type=Path, help="Fine-tuning run directory containing metrics.json and fold checkpoints")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("MILK10K_DATA_DIR", default_data_dir)))
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"], help="Defaults to the device from the source run")
    parser.add_argument("--variant", help="Defaults to the source run's selected variant")
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    data_dir = args.data_dir.expanduser().resolve()
    settings = json.loads((run_dir / "metrics.json").read_text())
    selected = args.variant or settings["selected_variant"]
    if selected not in settings["results"]:
        parser.error(f"Unknown variant {selected!r} for this run")
    device = torch.device(args.device or settings["encoder"]["device"])
    batch_size = settings["batch_size_pairs"]
    sensitivity_target = settings["exploratory_sensitivity_target"]
    torch.set_num_threads(min(4, torch.get_num_threads()))

    development, manifest_hash = load_development(data_dir / "splits" / MANIFEST)
    if manifest_hash != settings["manifest_sha256"]:
        raise ValueError("The development manifest differs from the source run")
    weights, checksum = CHECKPOINTS[ENCODER]
    if checksum != settings["encoder"]["sha256"]:
        raise ValueError("The pretrained ResNet checkpoint differs from the source run")

    ages = load_ages(data_dir, development)
    known_age = np.isfinite(ages)
    age_groups = pd.cut(ages, [-np.inf, 50, 65, np.inf], right=False, labels=AGE_BANDS[:3]).astype("string").fillna("unknown")
    targets = development["target"].to_numpy(dtype=int)
    folds = development["validation_fold"].to_numpy(dtype=int)
    fold_numbers = sorted(set(folds.tolist()))
    scores = {name: np.empty(len(development)) for name in ("image_only", "image_and_age")}

    frozen_features = None
    if selected == "frozen":
        frozen_features, _, _ = encode_pairs(ENCODER, development, data_dir, device, batch_size)
    else:
        pretrained = load_checkpoint(weights.url, checksum, data_dir)
        batches = DataLoader(LesionImages(development, data_dir, weights.transforms()), batch_size=batch_size)

    print(f"Comparing age heads using {selected} embeddings on {device}", flush=True)
    for fold in fold_numbers:
        validation = folds == fold
        training = ~validation
        print(f"  Fold {fold}", flush=True)
        features = frozen_features
        if features is None:
            features = encode_checkpoint(checkpoint_path(run_dir, fold, selected), pretrained, batches, device)
        images = fit_frozen_head(features[training], targets[training])
        scores["image_only"][validation] = images.predict_proba(features[validation])[:, 1]

        median_age = np.median(ages[training & known_age])
        with_age = np.column_stack((features, np.where(known_age, ages, median_age)))
        images_and_age = fit_frozen_head(with_age[training], targets[training])
        scores["image_and_age"][validation] = scores["image_only"][validation]
        use_age = validation & known_age
        if use_age.any():
            scores["image_and_age"][use_age] = images_and_age.predict_proba(with_age[use_age])[:, 1]

    predictions = development[["lesion_id", "diagnosis_code", "target", "validation_fold"]].copy()
    results = {}
    for name, probabilities in scores.items():
        predictions[f"{name}_score"] = probabilities
        summary = summarize(targets, probabilities, sensitivity_target)
        flagged = probabilities >= summary["threshold"]
        by_age_group = {}
        for age_group in AGE_BANDS:
            group = age_groups == age_group
            melanomas = group & (targets == 1)
            nevi = group & (targets == 0)
            by_age_group[age_group] = {
                "melanomas": int(melanomas.sum()),
                "melanomas_missed": int((melanomas & ~flagged).sum()),
                "nevi": int(nevi.sum()),
                "nevi_flagged": int((nevi & flagged).sum()),
            }
        results[name] = {
            **summary,
            "by_age_group": by_age_group,
            "by_fold": {fold: summarize(targets[folds == fold], probabilities[folds == fold], sensitivity_target) for fold in fold_numbers},
        }
        print(f"{name}: AUC {summary['roc_auc']:.3f}, sensitivity {summary['sensitivity']:.1%}, specificity {summary['specificity']:.1%}")

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = run_dir / f"age-{run_id}"
    output_dir.mkdir()
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    report = {
        "manifest_sha256": manifest_hash,
        "source_variant": selected,
        "device": str(device),
        "exploratory_sensitivity_target": sensitivity_target,
        "missing_ages": int((~known_age).sum()),
        "results": results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
