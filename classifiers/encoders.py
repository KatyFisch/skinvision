"""Extract paired lesion features with frozen Torchvision encoders."""

import hashlib
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
import torchvision
from PIL import Image, ImageOps
from torch import nn
from torchvision import transforms
from torchvision.models import ConvNeXt_Tiny_Weights, ResNet18_Weights, convnext_tiny, resnet18

CHECKPOINTS = {
    "resnet18": (ResNet18_Weights.IMAGENET1K_V1, "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"),
    "convnext_tiny": (ConvNeXt_Tiny_Weights.IMAGENET1K_V1, "983f1562536e84ff750a1576fb08e54de751dbf2e17c0d8a4a13704341fdcd3d"),
}
FEATURE_CACHE_VERSION = 1
IMAGE_COLUMNS = ["clinical_path", "dermoscopic_path"]


def build_preprocess(name: str, *, full_frame: bool = False) -> Callable[[Image.Image], torch.Tensor]:
    standard = CHECKPOINTS[name][0].transforms()
    if not full_frame:
        return standard
    return transforms.Compose(
        [
            transforms.Resize(standard.resize_size, interpolation=standard.interpolation, antialias=standard.antialias),
            transforms.PILToTensor(),
            transforms.ConvertImageDtype(torch.float32),
            transforms.Normalize(standard.mean, standard.std),
        ]
    )


def load_image(path: Path, preprocess: Callable[[Image.Image], torch.Tensor]) -> torch.Tensor:
    with Image.open(path) as photo:
        return preprocess(ImageOps.exif_transpose(photo).convert("RGB"))


def load_checkpoint(url: str, sha256: str, data_dir: Path) -> Path:
    path = data_dir / "checkpoints" / "torchvision" / Path(urlparse(url).path).name
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        torch.hub.download_url_to_file(url, str(path), hash_prefix=sha256)
    with path.open("rb") as checkpoint:
        digest = hashlib.file_digest(checkpoint, "sha256").hexdigest()
    if digest != sha256:
        raise ValueError(f"Checksum mismatch for {path}. Remove the damaged file and rerun to download the official checkpoint.")
    return path


def build_encoder(name: str, checkpoint: Path, device: torch.device) -> nn.Module:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if name == "resnet18":
        model = resnet18()
        model.load_state_dict(state)
        model.fc = nn.Identity()
    else:
        model = convnext_tiny()
        model.load_state_dict(state)
        model.classifier[2] = nn.Identity()
    return model.eval().requires_grad_(False).to(device)


def feature_cache_path(
    pairs: pd.DataFrame, data_dir: Path, name: str, settings: dict[str, str], *, image_columns: Sequence[str] = IMAGE_COLUMNS
) -> Path:
    digest = hashlib.sha256(json.dumps(settings, sort_keys=True).encode())
    for column in image_columns:
        for relative_path in pairs[column]:
            image_stat = (data_dir / relative_path).stat()
            digest.update(f"{relative_path}\0{image_stat.st_size}\0{image_stat.st_mtime_ns}\n".encode())
    return data_dir / "experiments" / "mel-nv" / "features" / f"{name}-{digest.hexdigest()}.npy"


def encode_pairs(
    name: str,
    pairs: pd.DataFrame,
    data_dir: Path,
    device: torch.device,
    batch_size: int,
    *,
    image_columns: Sequence[str] = IMAGE_COLUMNS,
    full_frame: bool = False,
) -> tuple[np.ndarray, dict[str, str], bool]:
    """Extract and concatenate the selected image views per lesion."""
    weights, sha256 = CHECKPOINTS[name]
    checkpoint = load_checkpoint(weights.url, sha256, data_dir)
    preprocess = build_preprocess(name, full_frame=full_frame)
    settings = {
        "checkpoint": str(weights),
        "url": weights.url,
        "sha256": sha256,
        "preprocessing": str(preprocess),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "device": str(device),
        "feature_cache_version": str(FEATURE_CACHE_VERSION),
    }
    if tuple(image_columns) != tuple(IMAGE_COLUMNS):
        settings["image_columns"] = ",".join(image_columns)
    cache_path = feature_cache_path(pairs, data_dir, name, settings, image_columns=image_columns)
    if cache_path.is_file():
        return np.load(cache_path, allow_pickle=False), settings, True

    encoder = build_encoder(name, checkpoint, device)
    feature_batches = []
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs.iloc[start : start + batch_size]
            images = [load_image(data_dir / relative_path, preprocess) for column in image_columns for relative_path in batch[column]]
            embeddings = encoder(torch.stack(images).to(device)).cpu().numpy()
            feature_batches.append(np.concatenate(np.split(embeddings, len(image_columns)), axis=1))
            if start == 0 or (start // batch_size + 1) % 10 == 0 or start + batch_size >= len(pairs):
                print(f"{name}: encoded {min(start + batch_size, len(pairs))}/{len(pairs)} lesions", flush=True)

    features = np.concatenate(feature_batches)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=cache_path.parent, delete_on_close=False) as temp:
        np.save(temp, features, allow_pickle=False)
        temp.close()
        Path(temp.name).replace(cache_path)
    return features, settings, False
