"""Export the fitted MEL-vs-NV heads as a PyTorch checkpoint."""

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from classifiers import dermfm_zero
from classifiers.model import load_classifier_run

FORMAT = "skinvision.padded-crop.v1"


def export_heads(run_dir: Path, output: Path) -> None:
    _, cutoff, heads = load_classifier_run(run_dir, expected_task="padded_crop")
    scalers = [head.named_steps["standardscaler"] for head in heads]
    regressions = [head.named_steps["logisticregression"] for head in heads]
    checkpoint = {
        "format": FORMAT,
        "encoder_sha256": dermfm_zero.WEIGHTS_SHA256,
        "cutoff": cutoff,
        "mean": torch.from_numpy(np.stack([scaler.mean_ for scaler in scalers])),
        "scale": torch.from_numpy(np.stack([scaler.scale_ for scaler in scalers])),
        "weight": torch.from_numpy(np.stack([regression.coef_[0] for regression in regressions])),
        "bias": torch.from_numpy(np.concatenate([regression.intercept_ for regression in regressions])),
    }
    torch.save(checkpoint, output)


class PaddedCropTorchModel(nn.Module):
    """Score normalized crops with a frozen encoder and differentiable heads."""

    mean: torch.Tensor
    scale: torch.Tensor
    weight: torch.Tensor
    bias: torch.Tensor

    def __init__(self, checkpoint_path: Path, data_dir: Path, device: torch.device) -> None:
        super().__init__()
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint["format"] != FORMAT or checkpoint["encoder_sha256"] != dermfm_zero.WEIGHTS_SHA256:
            raise ValueError("Expected exported padded-crop heads for the pinned DermFM-Zero-Open encoder")
        self.cutoff: float = checkpoint["cutoff"]
        self.encoder = dermfm_zero.load_clinical_encoder(data_dir, device)
        # Keep the scaler's double precision on a backend that supports it.
        head_device = torch.device("cpu") if device.type == "mps" else device
        for name in ("mean", "scale", "weight", "bias"):
            self.register_buffer(name, checkpoint[name].to(head_device))

    def score_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        features = embeddings.to(device=self.mean.device, dtype=torch.float32)
        # Match StandardScaler's two in-place operations on float32 features.
        centered = (features[:, None, :] - self.mean).float()
        standardized = (centered / self.scale).float()
        logits = (standardized * self.weight).sum(dim=-1) + self.bias
        return logits.sigmoid().mean(dim=-1)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return scores in NV, MEL order for normalized NCHW input."""
        melanoma = self.score_embeddings(self.encoder(images))
        scores = torch.stack((1 - melanoma, melanoma), dim=1)
        return scores.float().to(images.device) if images.device.type == "mps" else scores


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path, help="Saved padded-crop run containing metrics.json and heads.joblib")
    parser.add_argument("output", type=Path, help="Destination .pth for the fitted heads and cutoff")
    args = parser.parse_args()
    export_heads(args.run_dir, args.output)


if __name__ == "__main__":
    main()
