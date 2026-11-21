"""Fine-tune a shared ResNet-18 for the router and nine-class diagnosis head."""

import argparse
import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pandas as pd
import torch
import torchvision
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from torch import nn
from torch.nn import functional
from torch.utils.data import DataLoader

from classifiers.catalog import load_catalog
from classifiers.encoders import (
    CHECKPOINTS,
    IMAGE_COLUMNS,
    build_encoder,
    build_preprocess,
    encode_pairs,
    feature_cache_path,
    load_checkpoint,
)
from classifiers.eval_split.create_splits import FULL_MANIFEST_NAME
from classifiers.eval_split.prepare_routed_tasks import OTHER_DIAGNOSIS_TARGETS, prepare_routed_tasks
from classifiers.finetune_mel_nv import BLOCK_LR, EPOCHS, HEAD_LR, LesionImages
from classifiers.train_mel_nv import fit_frozen_head
from classifiers.train_routed import ENCODER, summarize_router

FROZEN_LAYERS = ("conv1", "bn1", "relu", "maxpool", "layer1", "layer2", "layer3")
VARIANTS = ("frozen", "fine_tuned", "refit")
SEED = 42


@torch.inference_mode()
def cache_frozen_activations(
    rows: pd.DataFrame, data_dir: Path, device: torch.device, batch_size: int, image_columns: list[str] = IMAGE_COLUMNS
) -> np.ndarray:
    """Run the frozen part once per photo and memory-map its outputs across folds."""
    weights, checksum = CHECKPOINTS[ENCODER]
    preprocess = build_preprocess(ENCODER, full_frame=True, native_resolution=True)
    settings = {
        "checkpoint_sha256": checksum,
        "preprocessing": str(preprocess),
        "layer": FROZEN_LAYERS[-1],
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
    }
    path = feature_cache_path(rows, data_dir, f"{ENCODER}-layer3-native", settings, image_columns=image_columns)
    if path.is_file():
        return np.load(path, mmap_mode="r", allow_pickle=False)

    checkpoint = load_checkpoint(weights.url, checksum, data_dir)
    encoder = build_encoder(ENCODER, checkpoint, device)
    trunk = nn.Sequential(*(encoder.get_submodule(name) for name in FROZEN_LAYERS)).eval()
    batches = DataLoader(LesionImages(rows, data_dir, preprocess, image_columns), batch_size=batch_size)
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete_on_close=False) as temporary:
        temporary.close()
        cache = None
        start = 0
        for batch_index, (*views, _) in enumerate(batches):
            encoded = trunk(torch.cat(views).to(device)).cpu().numpy()
            pairs = np.stack(np.split(encoded, len(views)), axis=1)
            if cache is None:
                cache = np.lib.format.open_memmap(temporary.name, mode="w+", dtype=pairs.dtype, shape=(len(rows), *pairs.shape[1:]))
            stop = start + len(pairs)
            cache[start:stop] = pairs
            start = stop
            if batch_index == 0 or (batch_index + 1) % 10 == 0 or stop == len(rows):
                print(f"Cached frozen layers for {stop}/{len(rows)} lesions", flush=True)
        assert cache is not None and start == len(rows)
        cache.flush()
        del cache
        Path(temporary.name).replace(path)
    return np.load(path, mmap_mode="r", allow_pickle=False)


class StandardizedHead(nn.Module):
    mean: torch.Tensor
    scale: torch.Tensor

    def __init__(self, classes: int, features: int) -> None:
        super().__init__()
        self.linear = nn.Linear(features, classes)
        self.register_buffer("mean", torch.zeros(features))
        self.register_buffer("scale", torch.ones(features))

    @torch.no_grad()
    def load_fit(self, fitted: Pipeline) -> None:
        scaler = fitted.named_steps["standardscaler"]
        head = fitted.named_steps["logisticregression"]
        self.mean.copy_(torch.from_numpy(scaler.mean_))
        self.scale.copy_(torch.from_numpy(scaler.scale_))
        self.linear.weight.copy_(torch.from_numpy(head.coef_))
        self.linear.bias.copy_(torch.from_numpy(head.intercept_))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear((features - self.mean) / self.scale)


class RoutedResNet18(nn.Module):
    def __init__(self, encoder: nn.Module, views: int = 2) -> None:
        super().__init__()
        self.encoder = encoder
        self.router = StandardizedHead(1, 512 * views)
        self.other = StandardizedHead(len(OTHER_DIAGNOSIS_TARGETS), 512 * views)

    def embed_cached(self, pairs: torch.Tensor) -> torch.Tensor:
        maps = self.encoder.get_submodule("layer4")(pairs.flatten(0, 1))
        return self.encoder.get_submodule("avgpool")(maps).reshape(len(pairs), -1)

    def classify(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.router(features).squeeze(-1), self.other(features)

    def forward(self, clinical: torch.Tensor, dermoscopy: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        if dermoscopy is None:
            return self.classify(self.encoder(clinical))
        clinical_features, dermoscopy_features = self.encoder(torch.cat((clinical, dermoscopy))).chunk(2)
        return self.classify(torch.cat((clinical_features, dermoscopy_features), dim=1))


def train_fold(
    model: RoutedResNet18,
    activation_cache: np.ndarray,
    training: np.ndarray,
    targets: torch.Tensor,
    other_targets: torch.Tensor,
    device: torch.device,
    batch_size: int,
    epochs: int,
    seed: int,
) -> list[dict[str, float]]:
    model.train()
    # Fixed running statistics keep the two views and the frozen cache consistent.
    model.encoder.eval()
    block = model.encoder.get_submodule("layer4").requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [{"params": [*model.router.parameters(), *model.other.parameters()], "lr": HEAD_LR}, {"params": block.parameters(), "lr": BLOCK_LR}]
    )
    rng = np.random.default_rng(seed)
    history = []
    for epoch in range(1, epochs + 1):
        totals = torch.zeros(2, device=device)
        shuffled = rng.permutation(training)
        for start in range(0, len(shuffled), batch_size):
            idx = shuffled[start : start + batch_size]
            pairs = torch.from_numpy(activation_cache[idx]).to(device)
            route_target = targets[idx].to(device)
            diagnosis_target = other_targets[idx].to(device)
            optimizer.zero_grad(set_to_none=True)
            router_logits, diagnosis_logits = model.classify(model.embed_cached(pairs))
            router_loss = functional.binary_cross_entropy_with_logits(router_logits, route_target, reduction="sum")
            diagnosis_loss = functional.cross_entropy(diagnosis_logits, diagnosis_target, ignore_index=-1, reduction="sum")
            other_count = diagnosis_target.ne(-1).sum().clamp_min(1)
            (router_loss / len(idx) + diagnosis_loss / other_count).backward()
            optimizer.step()
            totals += torch.stack((router_loss.detach(), diagnosis_loss.detach()))
        losses = totals.cpu().numpy()
        if not np.isfinite(losses).all():
            raise RuntimeError(f"Non-finite loss after epoch {epoch}")
        report = {"router": float(losses[0] / len(training)), "other": float(losses[1] / other_targets[training].ne(-1).sum().item())}
        history.append(report)
        print(f"  Epoch {epoch}/{epochs}: router loss {report['router']:.4f}, other loss {report['other']:.4f}", flush=True)
    return history


@torch.inference_mode()
def tuned_embeddings(model: RoutedResNet18, activation_cache: np.ndarray, batch_size: int, device: torch.device) -> np.ndarray:
    model.eval()
    return np.concatenate(
        [
            model.embed_cached(torch.from_numpy(activation_cache[start : start + batch_size].copy()).to(device)).cpu().numpy()
            for start in range(0, len(activation_cache), batch_size)
        ]
    )


@torch.inference_mode()
def predict_heads(model: RoutedResNet18, features: np.ndarray, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    router, other = model.classify(torch.from_numpy(features).to(device))
    return router.sigmoid().cpu().numpy(), other.argmax(dim=1).cpu().numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    default_dir = Path(__file__).resolve().parents[1] / "data"
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("MILK10K_DATA_DIR", default_dir)))
    parser.add_argument("--catalog", type=Path, default=Path("final_data.csv"), help="Relative to --data-dir or absolute")
    parser.add_argument("--device", choices=["cpu", "mps", "cuda"])
    parser.add_argument("--clinical-only", action="store_true", help="Train from one clinical photo per lesion")
    parser.add_argument("--batch-size", type=int, default=32, help="Lesions per batch")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--melanoma-sensitivity", type=float, default=0.95)
    args = parser.parse_args()
    if min(args.batch_size, args.epochs) < 1 or not 0 < args.melanoma_sensitivity <= 1:
        parser.error("Batch size and epochs must be positive and melanoma sensitivity must be in (0, 1]")

    data_dir = args.data_dir.expanduser().resolve()
    catalog = load_catalog(args.catalog, data_dir)
    assert catalog is not None
    router, _ = prepare_routed_tasks(data_dir, catalog)
    development = router.loc[router["split"].eq("train")]
    folds = development["validation_fold"].to_numpy(dtype=int)
    targets = development["target"].to_numpy(dtype=int)
    other_targets = development["diagnosis_code"].map(OTHER_DIAGNOSIS_TARGETS).fillna(-1).to_numpy(dtype=int)
    is_other = other_targets != -1
    route_tensor = torch.from_numpy(targets.astype(np.float32))
    other_tensor = torch.from_numpy(other_targets)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(SEED)
    image_columns = ["clinical_path"] if args.clinical_only else IMAGE_COLUMNS
    activation_cache = cache_frozen_activations(development, data_dir, device, args.batch_size, image_columns)
    frozen, encoder_settings, _ = encode_pairs(
        ENCODER, development, data_dir, device, args.batch_size, image_columns=image_columns, full_frame=True, native_resolution=True
    )
    weights, checksum = CHECKPOINTS[ENCODER]
    checkpoint = load_checkpoint(weights.url, checksum, data_dir)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    experiment = "routed-finetune-clinical" if args.clinical_only else "routed-finetune"
    output_dir = data_dir / "experiments" / experiment / run_id
    output_dir.mkdir(parents=True)
    print(f"Run directory: {output_dir}", flush=True)
    scores = {name: np.empty(len(development)) for name in VARIANTS}
    predictions = {name: np.empty(len(development), dtype=int) for name in VARIANTS}
    loss_history = {}

    for fold in sorted(set(folds.tolist())):
        validation = folds == fold
        training = ~validation
        other_training = training & is_other
        router_fit = fit_frozen_head(frozen[training], targets[training])
        other_fit = fit_frozen_head(frozen[other_training], other_targets[other_training])
        scores["frozen"][validation] = router_fit.predict_proba(frozen[validation])[:, 1]
        predictions["frozen"][validation] = other_fit.predict(frozen[validation])
        model = RoutedResNet18(build_encoder(ENCODER, checkpoint, device), views=len(image_columns)).to(device)
        model.router.load_fit(router_fit)
        model.other.load_fit(other_fit)
        print(f"Fine-tuning fold {fold}", flush=True)
        loss_history[fold] = train_fold(
            model, activation_cache, np.flatnonzero(training), route_tensor, other_tensor, device, args.batch_size, args.epochs, SEED + fold
        )

        features = tuned_embeddings(model, activation_cache, args.batch_size, device)
        for variant in ("fine_tuned", "refit"):
            if variant == "refit":
                model.router.load_fit(fit_frozen_head(features[training], targets[training]))
                model.other.load_fit(fit_frozen_head(features[other_training], other_targets[other_training]))
            scores[variant][validation], predictions[variant][validation] = predict_heads(model, features[validation], device)
            torch.save({name: value.cpu() for name, value in model.state_dict().items()}, output_dir / f"fold-{fold}-{variant}.pt")

    class_ids = list(OTHER_DIAGNOSIS_TARGETS.values())
    class_names = list(OTHER_DIAGNOSIS_TARGETS)
    routing_scores = development[["diagnosis_code", "target", "validation_fold"]].copy()
    diagnosis_scores = routing_scores.loc[is_other].copy()
    diagnosis_scores["target"] = other_targets[is_other]
    results = {}
    for variant in VARIANTS:
        routing = summarize_router(development, scores[variant], args.melanoma_sensitivity)
        report = classification_report(
            other_targets[is_other],
            predictions[variant][is_other],
            labels=class_ids,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        )
        routing_scores[f"{variant}_score"] = scores[variant]
        routing_scores[f"{variant}_sent_to_specialist"] = scores[variant] >= routing["threshold"]
        diagnosis_scores[f"{variant}_diagnosis"] = np.asarray(class_names)[predictions[variant][is_other]]
        results[variant] = {
            "router": routing,
            "other_diagnoses": {
                "classification_report": report,
                "confusion_matrix": confusion_matrix(other_targets[is_other], predictions[variant][is_other], labels=class_ids).tolist(),
            },
        }
        print(f"{variant}: router AUC {routing['roc_auc']:.3f}, nine-class macro F1 {report['macro avg']['f1-score']:.3f}", flush=True)
    routing_scores.to_csv(output_dir / "router_validation_predictions.csv")
    diagnosis_scores.to_csv(output_dir / "other_validation_predictions.csv")
    settings = {
        "catalog_sha256": catalog.sha256,
        "parent_manifest_sha256": hashlib.sha256((data_dir / "splits" / FULL_MANIFEST_NAME).read_bytes()).hexdigest(),
        "native_resolution": True,
        "image_columns": image_columns,
        "encoder": encoder_settings,
        "seed": SEED,
        "epochs": args.epochs,
        "batch_size_pairs": args.batch_size,
        "head_lr": HEAD_LR,
        "block_lr": BLOCK_LR,
        "class_order": class_names,
        "melanoma_sensitivity_target": args.melanoma_sensitivity,
        "loss_by_fold": loss_history,
        "results": results,
    }
    (output_dir / "metrics.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    print(f"Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
