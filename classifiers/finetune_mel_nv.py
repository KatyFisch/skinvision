"""Fit a paired ResNet-18 head, then fine-tune it with ResNet's last block."""

import argparse
import json
import math
import os
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import sklearn
import torch
from PIL import Image
from sklearn.pipeline import Pipeline
from torch import nn
from torch.utils.data import DataLoader, Dataset

from classifiers.encoders import CHECKPOINTS, IMAGE_COLUMNS, build_encoder, encode_pairs, load_checkpoint, load_image
from classifiers.train_mel_nv import MANIFEST, fit_frozen_head, load_development, summarize

ENCODER = "resnet18"
EPOCHS = 5
HEAD_LR = 1e-3
BLOCK_LR = 1e-4


class LesionImages(Dataset):
    def __init__(
        self,
        rows: pd.DataFrame,
        data_dir: Path,
        preprocess: Callable[[Image.Image], torch.Tensor],
        image_columns: list[str] = IMAGE_COLUMNS,
    ):
        self.paths = rows[image_columns].to_numpy()
        self.targets = rows["target"].to_numpy(dtype=np.float32)
        self.data_dir = data_dir
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple:
        views = tuple(load_image(self.data_dir / path, self.preprocess) for path in self.paths[index])
        return (*views, self.targets[index])


class LesionResNet18(nn.Module):
    mean: torch.Tensor
    scale: torch.Tensor

    def __init__(self, encoder: nn.Module, views: int = 2):
        super().__init__()
        self.encoder = encoder
        feature_size = views * 512
        self.head = nn.Linear(feature_size, 1)
        self.register_buffer("mean", torch.zeros(feature_size))
        self.register_buffer("scale", torch.ones(feature_size))

    @torch.no_grad()
    def load_frozen_head(self, fitted: Pipeline) -> None:
        scaler = fitted.named_steps["standardscaler"]
        classifier = fitted.named_steps["logisticregression"]
        self.mean.copy_(torch.from_numpy(scaler.mean_))
        self.scale.copy_(torch.from_numpy(scaler.scale_))
        self.head.weight.copy_(torch.from_numpy(classifier.coef_))
        self.head.bias.copy_(torch.from_numpy(classifier.intercept_))

    def logits_from_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        return self.head((embeddings - self.mean) / self.scale).squeeze(-1)

    def embeddings(self, *views: torch.Tensor) -> torch.Tensor:
        encoded = self.encoder(torch.cat(views, dim=0)).chunk(len(views), dim=0)
        return torch.cat(encoded, dim=1)

    def forward(self, *views: torch.Tensor) -> torch.Tensor:
        return self.logits_from_embeddings(self.embeddings(*views))


def train_last_block(
    model: LesionResNet18, batches: DataLoader, device: torch.device, epochs: int, head_lr: float, block_lr: float
) -> Iterator[float]:
    last_block = model.encoder.get_submodule("layer4").requires_grad_(True)
    optimizer = torch.optim.AdamW([{"params": model.head.parameters(), "lr": head_lr}, {"params": last_block.parameters(), "lr": block_lr}])
    criterion = nn.BCEWithLogitsLoss()
    for epoch in range(1, epochs + 1):
        model.train()
        # Keep the pretrained BatchNorm running statistics fixed in both views.
        model.encoder.eval()
        loss_sum = 0.0
        samples = 0
        for *views, targets in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(*(view.to(device) for view in views))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item() * len(targets)
            samples += len(targets)
        epoch_loss = loss_sum / samples
        if not math.isfinite(epoch_loss):
            raise RuntimeError(f"Training loss was non-finite at epoch {epoch}")
        print(f"  Pass 2 epoch {epoch}/{epochs}: train loss {epoch_loss:.4f}", flush=True)
        yield epoch_loss


@torch.inference_mode()
def predict(model: LesionResNet18, batches: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores = [model(*(view.to(device) for view in views)).sigmoid().cpu().numpy() for *views, _ in batches]
    return np.concatenate(scores)


@torch.inference_mode()
def encode_images(model: LesionResNet18, batches: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    return np.concatenate([model.embeddings(*(view.to(device) for view in views)).cpu().numpy() for *views, _ in batches])


def checkpoint_path(run_dir: Path, fold: int, variant: str) -> Path:
    suffix = "" if variant == "fine_tuned" else f"-{variant.replace('_', '-')}"
    return run_dir / f"fold-{fold}{suffix}.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_dir = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("MILK10K_DATA_DIR", default_data_dir)))
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16, help="Lesion pairs per batch")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Fixed number of epochs for pass 2")
    parser.add_argument("--head-lr", type=float, default=HEAD_LR, help="Pass 2 learning rate for the classifier head")
    parser.add_argument("--block-lr", type=float, default=BLOCK_LR, help="Pass 2 learning rate for ResNet's last block")
    parser.add_argument("--sensitivity-target", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if min(args.batch_size, args.epochs) < 1 or not (0 < args.sensitivity_target <= 1):
        parser.error("Batch size and epochs must be positive and sensitivity target must be in (0, 1]")
    if not (math.isfinite(args.head_lr) and 0 < args.block_lr < args.head_lr):
        parser.error("Learning rates must be finite and positive, with block-lr below head-lr")

    data_dir = args.data_dir.expanduser().resolve()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(args.seed)
    start = perf_counter()

    development, manifest_hash = load_development(data_dir / "splits" / MANIFEST)
    targets = development["target"].to_numpy(dtype=int)
    folds = development["validation_fold"].to_numpy(dtype=int)
    features, encoder_settings, cache_hit = encode_pairs(ENCODER, development, data_dir, device, args.batch_size)
    weights, checksum = CHECKPOINTS[ENCODER]
    checkpoint = load_checkpoint(weights.url, checksum, data_dir)
    preprocess = weights.transforms()
    print(f"Training on development folds of {len(development)} lesions using {device}", flush=True)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    output_dir = data_dir / "experiments" / "mel-nv" / f"{run_id}-resnet18-finetune"
    output_dir.mkdir(parents=True)
    epoch_variants = {
        epoch: "fine_tuned" if epoch == args.epochs else f"epoch_{epoch}" for epoch in (1, 2, args.epochs) if epoch <= args.epochs
    }
    scores_by_variant = {name: np.empty(len(development)) for name in ("frozen", *epoch_variants.values())}
    fold_losses = {}
    for fold in sorted(set(folds.tolist())):
        validation = folds == fold
        training = ~validation
        print(f"Fold {fold}, pass 1: fitting logistic regression on frozen embeddings", flush=True)
        head = fit_frozen_head(features[training], targets[training])
        scores_by_variant["frozen"][validation] = head.predict_proba(features[validation])[:, 1]
        model = LesionResNet18(build_encoder(ENCODER, checkpoint, device))
        model.load_frozen_head(head)
        model.to(device)

        train_pairs = LesionImages(development.loc[training], data_dir, preprocess)
        validation_pairs = LesionImages(development.loc[validation], data_dir, preprocess)
        generator = torch.Generator().manual_seed(args.seed + fold)
        train_batches = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True, generator=generator)
        validation_batches = DataLoader(validation_pairs, batch_size=args.batch_size)
        fold_losses[fold] = []
        for epoch, loss in enumerate(train_last_block(model, train_batches, device, args.epochs, args.head_lr, args.block_lr), start=1):
            fold_losses[fold].append(loss)
            if epoch in epoch_variants:
                variant = epoch_variants[epoch]
                scores_by_variant[variant][validation] = predict(model, validation_batches, device)
                torch.save({name: value.cpu() for name, value in model.state_dict().items()}, checkpoint_path(output_dir, fold, variant))

    predictions = development[["lesion_id", "diagnosis_code", "target", "validation_fold"]].copy()
    summaries = {name: summarize(targets, scores, args.sensitivity_target) for name, scores in scores_by_variant.items()}
    results = {}
    for name, scores in scores_by_variant.items():
        predictions[f"{name}_score"] = scores
        summary = summaries[name]
        results[name] = {
            **summary,
            "nearby_sensitivity": {target: summarize(targets, scores, target) for target in (0.94, 0.95, 0.96, 0.97, 0.98)},
            "by_fold": {
                fold: summarize(targets[folds == fold], scores[folds == fold], args.sensitivity_target) for fold in sorted(fold_losses)
            },
        }
        print(f"{name}: AUC {summary['roc_auc']:.3f}, sensitivity {summary['sensitivity']:.1%}, specificity {summary['specificity']:.1%}")

    selected_variant = max(summaries, key=lambda name: summaries[name]["specificity"])
    print(f"Selected at the development sensitivity target: {selected_variant}")
    predictions.to_csv(output_dir / "validation_predictions.csv", index=False)
    settings = {
        "manifest": MANIFEST,
        "manifest_sha256": manifest_hash,
        "encoder": encoder_settings,
        "feature_cache_hit": cache_hit,
        "scikit_learn": sklearn.__version__,
        "seed": args.seed,
        "batch_size_pairs": args.batch_size,
        "exploratory_sensitivity_target": args.sensitivity_target,
        "selected_variant": selected_variant,
        "pass_1": "StandardScaler and LogisticRegression(C=1.0, max_iter=2000) on frozen paired features",
        "pass_2": {
            "trainable": ["head", "encoder.layer4"],
            "batch_norm_running_statistics": "frozen",
            "optimizer": "AdamW",
            "head_lr": args.head_lr,
            "block_lr": args.block_lr,
            "epochs": args.epochs,
            "checkpoint_epochs": {variant: epoch for epoch, variant in epoch_variants.items()},
            "training_loss_by_fold": fold_losses,
        },
        "elapsed_seconds": perf_counter() - start,
        "results": results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
