"""Run existing single-photo cascade bundles."""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision.models import resnet18

from classifiers.catalog import MILK_DIAGNOSES
from classifiers.encoders import imagenet_normalize, load_rgb
from classifiers.score_bands import melanoma_score_band

PHOTO_EDGE = 600
OTHER_CLASSES = np.array(sorted(set(MILK_DIAGNOSES) - {"MEL", "NV"}))


@dataclass(frozen=True)
class Cutoffs:
    router: float
    specialist: float


def prepare_photo(image: Image.Image) -> torch.Tensor:
    """Match the training scale while keeping the complete frame and aspect ratio."""
    image.thumbnail((PHOTO_EDGE, PHOTO_EDGE), Image.Resampling.LANCZOS)
    return imagenet_normalize(image)


# Preserve the module and buffer names stored in existing state dictionaries.
class LesionResNet18(nn.Module):
    mean: torch.Tensor
    scale: torch.Tensor

    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(512, 1)
        self.register_buffer("mean", torch.zeros(512))
        self.register_buffer("scale", torch.ones(512))

    def forward(self, clinical: torch.Tensor) -> torch.Tensor:
        return self.head((self.encoder(clinical) - self.mean) / self.scale).squeeze(-1)


class StandardizedHead(nn.Module):
    mean: torch.Tensor
    scale: torch.Tensor

    def __init__(self, classes: int) -> None:
        super().__init__()
        self.linear = nn.Linear(512, classes)
        self.register_buffer("mean", torch.zeros(512))
        self.register_buffer("scale", torch.ones(512))

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.linear((features - self.mean) / self.scale)


class RoutedResNet18(nn.Module):
    def __init__(self, encoder: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.router = StandardizedHead(1)
        self.other = StandardizedHead(len(OTHER_CLASSES))

    def forward(self, clinical: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(clinical)
        return self.router(features).squeeze(-1), self.other(features)


def load_models(weights: list[dict[str, str]], base: Path, device: torch.device) -> tuple[list[RoutedResNet18], list[LesionResNet18]]:
    def encoder() -> nn.Module:
        model = resnet18()
        model.set_submodule("fc", nn.Identity())
        return model

    routed, specialist = [], []
    for paths in weights:
        routing = RoutedResNet18(encoder())
        detail = LesionResNet18(encoder())
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


class PhotoClassifier:
    def __init__(self, model_path: Path, device: torch.device) -> None:
        model_path = model_path.expanduser().resolve()
        config = json.loads(model_path.read_text())
        self.cutoffs = Cutoffs(**config["cutoffs"])
        self.routed, self.specialist = load_models(config["weights"], model_path.parent, device)
        self.device = device

    def predict(self, photo: Image.Image) -> dict[str, str | float | None]:
        image = prepare_photo(ImageOps.exif_transpose(photo).convert("RGB")).unsqueeze(0).to(self.device)
        route, other = routing_scores(self.routed, image)
        melanoma = float(specialist_scores(self.specialist, image)[0]) if route[0] >= self.cutoffs.router else None
        if melanoma is not None and not math.isfinite(melanoma):
            raise RuntimeError("The specialist returned an invalid melanoma score")
        if melanoma is None:
            return {"research_prediction": str(other[0]), "route": "OTHER", "melanoma_score": None, "melanoma_score_band": None}
        return {
            "research_prediction": "MEL" if melanoma >= self.cutoffs.specialist else "NV",
            "route": "MEL_OR_NV",
            "melanoma_score": melanoma,
            "melanoma_score_band": melanoma_score_band(melanoma, self.cutoffs.specialist),
        }


def predict_photo(model_path: Path, photo: Path, device: torch.device) -> dict[str, str | float | None]:
    return PhotoClassifier(model_path, device).predict(load_rgb(photo))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    prediction = parser.add_subparsers(dest="command", required=True).add_parser("predict", help="Run a saved cascade on a curated photo")
    prediction.add_argument("--model", type=Path, required=True, help="Path to a saved cascade model.json")
    prediction.add_argument("photo", type=Path)
    prediction.add_argument("--device", choices=["cpu", "mps", "cuda"])
    args = parser.parse_args()
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"))
    torch.set_num_threads(min(4, torch.get_num_threads()))
    try:
        result = predict_photo(args.model, args.photo, device)
    except OSError as error:
        prediction.error(str(error))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
