"""Decode clinical photos and cache frozen image embeddings."""

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import cast

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights

FEATURE_CACHE_VERSION = 1

# The U-Net and legacy cascade both use the ResNet-18 RGB normalization.
_imagenet = ResNet18_Weights.IMAGENET1K_V1.transforms()
_imagenet_normalize = transforms.Compose(
    [transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32), transforms.Normalize(_imagenet.mean, _imagenet.std)]
)


def imagenet_normalize(photo: Image.Image) -> torch.Tensor:
    return cast(torch.Tensor, _imagenet_normalize(photo))


def load_rgb(path: Path) -> Image.Image:
    with Image.open(path.expanduser()) as photo:
        return ImageOps.exif_transpose(photo).convert("RGB")


def cached_embeddings(
    name: str,
    rows: pd.DataFrame,
    data_dir: Path,
    device: torch.device,
    batch_size: int,
    preprocess: Callable[[Image.Image], torch.Tensor],
    load_encoder: Callable[[], nn.Module],
    settings: dict[str, str],
    *,
    dtype: torch.dtype,
) -> tuple[np.ndarray, dict[str, str], bool]:
    paths = rows["clinical_path"]
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for relative_path in paths:
        image_stat = (data_dir / relative_path).stat()
        digest.update(f"{relative_path}\0{image_stat.st_size}\0{image_stat.st_mtime_ns}\n".encode())
    cache_path = data_dir / "experiments" / "mel-nv" / "features" / f"{name}-{digest.hexdigest()}.npy"
    if cache_path.is_file():
        return np.load(cache_path, allow_pickle=False), settings, True

    encoder = load_encoder()
    feature_batches = []
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            batch = paths.iloc[start : start + batch_size]
            images = [preprocess(load_rgb(data_dir / path)) for path in batch]
            embeddings = encoder(torch.stack(images).to(device=device, dtype=dtype)).float().cpu().numpy()
            feature_batches.append(embeddings)
            if start == 0 or (start // batch_size + 1) % 10 == 0 or start + batch_size >= len(rows):
                print(f"{name}: encoded {min(start + batch_size, len(rows))}/{len(rows)} lesions", flush=True)

    features = np.concatenate(feature_batches)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=cache_path.parent, delete_on_close=False) as temp:
        np.save(temp, features, allow_pickle=False)
        temp.close()
        Path(temp.name).replace(cache_path)
    return features, settings, False
