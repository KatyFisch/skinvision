"""Frozen PanDerm Base features for native-resolution clinical photos."""

from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torchvision
from torch import nn

from classifiers.encoders import FEATURE_CACHE_VERSION, build_preprocess, cached_embeddings, load_checkpoint

NAME = "panderm_base"
SOURCE = "https://github.com/SiyuanYan1/PanDerm#1-download-panderm-pre-trained-weights"
WEIGHTS_URL = "https://drive.usercontent.google.com/download?id=17J4MjsZu3gdBP6xAQi_NMDVvH65a00HB&export=download&confirm=t"
WEIGHTS_SHA256 = "be1e0fb108b3bc58721cb5195f136c948160799438f222acf1fd142230ac1ff1"
WEIGHTS_FILENAME = "panderm_bb_data6_checkpoint-499.pth"


def build_panderm(checkpoint: Path, device: torch.device, dtype: torch.dtype) -> nn.Module:
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    encoder = timm.create_model(
        "vit_base_patch16_224", num_classes=0, init_values=0.1, dynamic_img_size=True, dynamic_img_pad=True, weight_init="skip"
    )
    convert_attention_weights(state, depth=12)
    encoder.load_state_dict(state, strict=True)
    return encoder.eval().requires_grad_(False).to(device=device, dtype=dtype)


def convert_attention_weights(state: dict[str, torch.Tensor], *, depth: int) -> None:
    """Map PanDerm's separate query/value biases and residual scales to timm."""
    for index in range(depth):
        prefix = f"blocks.{index}"
        query = state.pop(f"{prefix}.attn.q_bias")
        value = state.pop(f"{prefix}.attn.v_bias")
        state[f"{prefix}.attn.qkv.bias"] = torch.cat((query, torch.zeros_like(value), value))
        state[f"{prefix}.ls1.gamma"] = state.pop(f"{prefix}.gamma_1")
        state[f"{prefix}.ls2.gamma"] = state.pop(f"{prefix}.gamma_2")


def encode_clinical(rows: pd.DataFrame, data_dir: Path, device: torch.device, batch_size: int) -> tuple[np.ndarray, dict[str, str], bool]:
    preprocess = build_preprocess("resnet18", full_frame=True, native_resolution=True)
    dtype = torch.float16 if device.type == "mps" else torch.float32
    settings = {
        "checkpoint": WEIGHTS_FILENAME,
        "source": SOURCE,
        "license": "CC BY-NC-ND 4.0",
        "url": WEIGHTS_URL,
        "sha256": WEIGHTS_SHA256,
        "preprocessing": f"{preprocess}; normalized zero padding to 16-pixel patches, interpolated position embeddings",
        "feature": "normalized final CLS token",
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "timm": timm.__version__,
        "device": str(device),
        "dtype": str(dtype),
        "feature_cache_version": str(FEATURE_CACHE_VERSION),
        "image_columns": "clinical_path",
    }

    def load_encoder() -> nn.Module:
        checkpoint = load_checkpoint(WEIGHTS_URL, WEIGHTS_SHA256, data_dir, filename=WEIGHTS_FILENAME, publisher="panderm")
        return build_panderm(checkpoint, device, dtype)

    return cached_embeddings(NAME, rows, data_dir, device, batch_size, ["clinical_path"], preprocess, load_encoder, settings, dtype=dtype)
