"""Fit a paired ResNet-18 head, then fine-tune it with ResNet's last block."""

import argparse
import json
import math
import os
from collections.abc import Callable
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


class LesionPairs(Dataset):
    def __init__(self, rows: pd.DataFrame, data_dir: Path, preprocess: Callable[[Image.Image], torch.Tensor]):
        self.paths = rows[IMAGE_COLUMNS].to_numpy()
        self.targets = rows["target"].to_numpy(dtype=np.float32)
        self.data_dir = data_dir
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, np.float32]:
        clinical, dermoscopy = self.paths[index]
        return (
            load_image(self.data_dir / clinical, self.preprocess),
            load_image(self.data_dir / dermoscopy, self.preprocess),
            self.targets[index],
        )


class PairedResNet18(nn.Module):
    mean: torch.Tensor
    scale: torch.Tensor

    def __init__(self, encoder: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(1024, 1)
        self.register_buffer("mean", torch.zeros(1024))
        self.register_buffer("scale", torch.ones(1024))

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

    def forward(self, clinical: torch.Tensor, dermoscopy: torch.Tensor) -> torch.Tensor:
        views = self.encoder(torch.cat((clinical, dermoscopy), dim=0)).chunk(2, dim=0)
        return self.logits_from_embeddings(torch.cat(views, dim=1))


def train_last_block(
    model: PairedResNet18, batches: DataLoader, device: torch.device, epochs: int, head_lr: float, block_lr: float
) -> list[float]:
    model.train()
    # Keep the pretrained BatchNorm running statistics fixed in both views.
    model.encoder.eval()
    last_block = model.encoder.get_submodule("layer4").requires_grad_(True)
    optimizer = torch.optim.AdamW([{"params": model.head.parameters(), "lr": head_lr}, {"params": last_block.parameters(), "lr": block_lr}])
    criterion = nn.BCEWithLogitsLoss()
    epoch_losses = []
    for epoch in range(1, epochs + 1):
        loss_sum = 0.0
        samples = 0
        for clinical, dermoscopy, targets in batches:
            optimizer.zero_grad(set_to_none=True)
            logits = model(clinical.to(device), dermoscopy.to(device))
            loss = criterion(logits, targets.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.detach().item() * len(targets)
            samples += len(targets)
        epoch_loss = loss_sum / samples
        if not math.isfinite(epoch_loss):
            raise RuntimeError(f"Training loss was non-finite at epoch {epoch}")
        epoch_losses.append(epoch_loss)
        print(f"  Pass 2 epoch {epoch}/{epochs}: train loss {epoch_loss:.4f}", flush=True)
    return epoch_losses


@torch.inference_mode()
def predict(model: PairedResNet18, batches: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    scores = [model(clinical.to(device), dermoscopy.to(device)).sigmoid().cpu().numpy() for clinical, dermoscopy, _ in batches]
    return np.concatenate(scores)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data_dir = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("MILK10K_DATA_DIR", default_data_dir)))
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--batch-size", type=int, default=16, help="Lesion pairs per batch")
    parser.add_argument("--epochs", type=int, default=5, help="Fixed number of epochs for pass 2")
    parser.add_argument("--head-lr", type=float, default=1e-3, help="Pass 2 learning rate for the classifier head")
    parser.add_argument("--block-lr", type=float, default=1e-4, help="Pass 2 learning rate for ResNet's last block")
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
    frozen_scores = np.empty(len(development))
    fine_tuned_scores = np.empty(len(development))
    fold_losses = {}
    for fold in sorted(set(folds.tolist())):
        validation = folds == fold
        training = ~validation
        print(f"Fold {fold}, pass 1: fitting logistic regression on frozen embeddings", flush=True)
        head = fit_frozen_head(features[training], targets[training])
        frozen_scores[validation] = head.predict_proba(features[validation])[:, 1]
        model = PairedResNet18(build_encoder(ENCODER, checkpoint, device))
        model.load_frozen_head(head)
        model.to(device)

        train_pairs = LesionPairs(development.loc[training], data_dir, preprocess)
        validation_pairs = LesionPairs(development.loc[validation], data_dir, preprocess)
        generator = torch.Generator().manual_seed(args.seed + fold)
        train_batches = DataLoader(train_pairs, batch_size=args.batch_size, shuffle=True, generator=generator)
        validation_batches = DataLoader(validation_pairs, batch_size=args.batch_size)
        fold_losses[fold] = train_last_block(model, train_batches, device, args.epochs, args.head_lr, args.block_lr)
        fine_tuned_scores[validation] = predict(model, validation_batches, device)
        torch.save({name: value.cpu() for name, value in model.state_dict().items()}, output_dir / f"fold-{fold}.pt")

    predictions = development[["lesion_id", "diagnosis_code", "target", "validation_fold"]].copy()
    results = {}
    for name, scores in [("frozen", frozen_scores), ("fine_tuned", fine_tuned_scores)]:
        predictions[f"{name}_score"] = scores
        summary = summarize(targets, scores, args.sensitivity_target)
        results[name] = {
            **summary,
            "by_fold": {
                fold: summarize(targets[folds == fold], scores[folds == fold], args.sensitivity_target) for fold in sorted(fold_losses)
            },
        }
        print(f"{name}: AUC {summary['roc_auc']:.3f}, sensitivity {summary['sensitivity']:.1%}, specificity {summary['specificity']:.1%}")

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
        "pass_1": "StandardScaler and LogisticRegression(C=1.0, max_iter=2000) on frozen paired features",
        "pass_2": {
            "trainable": ["head", "encoder.layer4"],
            "batch_norm_running_statistics": "frozen",
            "optimizer": "AdamW",
            "head_lr": args.head_lr,
            "block_lr": args.block_lr,
            "epochs": args.epochs,
            "training_loss_by_fold": fold_losses,
        },
        "elapsed_seconds": perf_counter() - start,
        "results": results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output_dir}")


if __name__ == "__main__":
    main()
