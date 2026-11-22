"""Frozen image embeddings from the gated DermFM-Zero-Open research checkpoint."""

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torchvision
from huggingface_hub import hf_hub_download
from timm.layers.pos_embed import resample_abs_pos_embed
from timm.models.vision_transformer import VisionTransformer
from torchvision import transforms

from classifiers.encoders import FEATURE_CACHE_VERSION, cached_embeddings
from classifiers.panderm import convert_attention_weights

NAME = "dermfm_zero_open"
REPOSITORY = "redlessone/DermFM-Zero"
REVISION = "2039b405065ff5ead7e58c04ec5604bd1e4a7769"
WEIGHTS_FILENAME = "open_clip_pytorch_model.bin"
WEIGHTS_SHA256 = "83f498e3ee5e0963a14356b2d0facde640a2f108972d12437705a805f30d2918"
ISIC_IDS_FILENAME = "pretrain_image_lists/isic_image_ids.txt"
ISIC_IDS_SHA256 = "fe92cbf950b3ac4b01c07e03b5af1e71fadff4b32ba7d9c7334b7770a6474f51"
SOURCE = "https://github.com/SiyuanYan1/DermFM-Zero/tree/c9f3d6a149ab048ae8a98ea4701e23b271e7368a"


class DermFMVisionTransformer(VisionTransformer):
    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        # The authors' PanDermVisualWrapper uses bicubic interpolation without antialiasing.
        batch, height, width, channels = x.shape
        assert self.pos_embed is not None and self.cls_token is not None
        positions = resample_abs_pos_embed(self.pos_embed, [height, width], old_size=[14, 14], antialias=False)
        tokens = torch.cat((self.cls_token.expand(batch, -1, -1), x.reshape(batch, height * width, channels)), dim=1)
        return self.pos_drop(tokens + positions.to(dtype=x.dtype))


def checkpoint_dir(data_dir: Path) -> Path:
    return data_dir / "checkpoints" / "dermfm-zero" / REVISION


def download(filename: str, data_dir: Path) -> Path:
    local_dir = checkpoint_dir(data_dir)
    path = local_dir / filename
    if not path.is_file():
        hf_hub_download(repo_id=REPOSITORY, filename=filename, revision=REVISION, local_dir=local_dir)
    return path


def pretraining_id_overlap(rows: pd.DataFrame, data_dir: Path) -> dict:
    content = download(ISIC_IDS_FILENAME, data_dir).read_bytes()
    if hashlib.sha256(content).hexdigest() != ISIC_IDS_SHA256:
        raise ValueError("The published DermFM-Zero-Open pretraining image list has an unexpected checksum")
    listed_ids = set(content.decode("utf-8").splitlines())
    matches = {
        view: np.fromiter((Path(path).stem in listed_ids for path in rows[f"{view}_path"]), dtype=bool, count=len(rows))
        for view in ("clinical", "dermoscopic")
    }
    assignment = rows["split"].to_numpy()
    return {
        "listed_isic_images": len(listed_ids),
        "source": f"{REPOSITORY}@{REVISION}/{ISIC_IDS_FILENAME}",
        "sha256": ISIC_IDS_SHA256,
        "exact_matches_by_split": {
            split: {view: int(np.count_nonzero((assignment == name) & matched)) for view, matched in matches.items()}
            for split, name in (("development", "train"), ("holdout", "test"))
        },
    }


def build_encoder(checkpoint: Path, device: torch.device, dtype: torch.dtype) -> DermFMVisionTransformer:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    state = state.get("state_dict", state)
    vision = {
        key.removeprefix("module.").removeprefix("visual.trunk."): value
        for key, value in state.items()
        if key.removeprefix("module.").startswith("visual.trunk.")
    }
    # The release's linear probes use the pooled 1,024 values before the text-aligned projection.
    vision.pop("head.weight")
    vision.pop("head.bias")
    vision["fc_norm.weight"] = vision.pop("norm.weight")
    vision["fc_norm.bias"] = vision.pop("norm.bias")
    convert_attention_weights(vision, depth=24)
    encoder = DermFMVisionTransformer(
        img_size=224,
        patch_size=16,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        qkv_bias=True,
        num_classes=0,
        global_pool="avg",
        fc_norm=True,
        init_values=0.1,
        dynamic_img_size=True,
        dynamic_img_pad=True,
        weight_init="skip",
    )
    encoder.load_state_dict(vision, strict=True, assign=True)
    return encoder.eval().requires_grad_(False).to(device=device, dtype=dtype)


def clinical_preprocess() -> transforms.Compose:
    return transforms.Compose(
        [transforms.PILToTensor(), transforms.ConvertImageDtype(torch.float32), transforms.Normalize((0.5,) * 3, (0.5,) * 3)]
    )


def clinical_dtype(device: torch.device) -> torch.dtype:
    return torch.float16 if device.type == "mps" else torch.float32


def load_clinical_encoder(data_dir: Path, device: torch.device) -> DermFMVisionTransformer:
    checkpoint = download(WEIGHTS_FILENAME, data_dir)
    with checkpoint.open("rb") as weights:
        digest = hashlib.file_digest(weights, "sha256").hexdigest()
    if digest != WEIGHTS_SHA256:
        raise ValueError(f"DermFM-Zero-Open checkpoint checksum mismatch for {checkpoint}")
    return build_encoder(checkpoint, device, clinical_dtype(device))


def encode_clinical(rows: pd.DataFrame, data_dir: Path, device: torch.device, batch_size: int) -> tuple[np.ndarray, dict[str, str], bool]:
    preprocess = clinical_preprocess()
    dtype = clinical_dtype(device)
    settings = {
        "checkpoint": f"{REPOSITORY}@{REVISION}/{WEIGHTS_FILENAME}",
        "source": SOURCE,
        "license": "CC BY-NC-ND 4.0",
        "sha256": WEIGHTS_SHA256,
        "preprocessing": f"{preprocess}; native full frame, normalized zero padding to 16-pixel patches, bicubic positional interpolation",
        "feature": "mean patch embeddings followed by LayerNorm, before vision-language projection",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "device": str(device),
        "dtype": str(dtype),
        "feature_cache_version": str(FEATURE_CACHE_VERSION),
        "image_columns": "clinical_path",
    }

    return cached_embeddings(
        NAME,
        rows,
        data_dir,
        device,
        batch_size,
        ["clinical_path"],
        preprocess,
        lambda: load_clinical_encoder(data_dir, device),
        settings,
        dtype=dtype,
    )
