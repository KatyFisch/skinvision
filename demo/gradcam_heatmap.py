"""
Generate Grad-CAM heatmaps for the demo and the explanation notebook.

The returned heatmap is a 2D NumPy array with values between 0 and 1.
No plot or heatmap overlay is created.

Example
-------
from demo.gradcam_heatmap import load_model, generate_heatmap

model = load_model("../models/mel-vs-nv-full.pth")
heatmap = generate_heatmap("path/to/new_image.jpg", model)

print(heatmap.shape)  # (image_height, image_width)
print(heatmap.min(), heatmap.max())
"""

from base64 import b64encode
from collections.abc import Callable
from io import BytesIO
from pathlib import Path
from typing import cast

import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from torch import nn
from torchvision import transforms

from classifiers.export_padded_crop import PaddedCropTorchModel

HEAT_THRESHOLD = 0.30
MAX_HEATMAP_ALPHA = round(0.75 * 255)


class GradCAMWrapper(nn.Module):
    """Expose the first model output in the format expected by Grad-CAM."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0].reshape(-1, 1)


def tensor_heatmap(
    model: nn.Module,
    image: torch.Tensor,
    target_layer: nn.Module,
    target_category: int,
    reshape_transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
) -> np.ndarray:
    # Grad-CAM accepts callable targets even though its annotation says Module.
    targets = cast(list[nn.Module], [ClassifierOutputTarget(target_category)])
    maps = None
    with torch.enable_grad(), GradCAM(model=model, target_layers=[target_layer], reshape_transform=reshape_transform) as cam:
        maps = cam(input_tensor=image.requires_grad_(True), targets=targets)
    if maps is None:
        raise RuntimeError("Grad-CAM did not return a heatmap")
    heat = np.asarray(maps[0], dtype=np.float32)
    if not np.isfinite(heat).all():
        raise RuntimeError("Grad-CAM returned non-finite values")
    return np.clip(heat, 0.0, 1.0)


def load_model(model_path, device=None):
    """
    Load the complete model checkpoint used in the notebook.

    Note: weights_only=False is required because the notebook saves/loads
    the complete model object. Only load checkpoints you trust.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    model = torch.load(Path(model_path), map_location=device, weights_only=False)
    model = model.to(device)
    model.eval()
    return model


def generate_heatmap(image, model, target_category=0, device=None):
    """
    Generate a Grad-CAM heatmap for one image.

    Parameters
    ----------
    image : str, pathlib.Path, or PIL.Image.Image
        Input image. It is converted to RGB.
    model : torch.nn.Module
        Model loaded with load_model().
    target_category : int, default=0
        Class index passed to ClassifierOutputTarget. The notebook uses 0.
    device : str or torch.device, optional
        Defaults to the model's device, or CUDA/CPU if it cannot be inferred.

    Returns
    -------
    numpy.ndarray
        2D float32 heatmap with values between 0 and 1.
    """
    if device is None:
        try:
            device = next(model.parameters()).device
        except StopIteration:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Match the notebook preprocessing: RGB -> PILToTensor -> float32.
    if isinstance(image, (str, Path)):
        with Image.open(image) as img:
            pil_image = img.convert("RGB")
            pil_image.load()
    elif isinstance(image, Image.Image):
        pil_image = image.convert("RGB")
    else:
        raise TypeError("image must be a file path or a PIL.Image.Image")

    input_tensor = transforms.PILToTensor()(pil_image).unsqueeze(0)
    input_tensor = input_tensor.to(device=device, dtype=torch.float32)

    model = model.to(device)
    model.eval()
    cam_model = GradCAMWrapper(model)

    # The notebook explains model(x)[0], i.e. the first fold/output.
    # Therefore use the corresponding layer in the first fold only.
    try:
        first_fold_name = next(iter(model.folds.named_children()))[0]
    except (AttributeError, StopIteration):
        raise AttributeError("Expected the model to have a non-empty 'folds' module, as in the notebook.")

    target_layer = model.get_submodule(f"folds.{first_fold_name}.encoder.layer4.1.conv2")

    return tensor_heatmap(cam_model, input_tensor, target_layer, target_category)


def padded_crop_heatmap(photo: Image.Image, classifier: PaddedCropTorchModel, prediction: str) -> np.ndarray:
    encoder = classifier.encoder
    image = classifier.image_tensor(photo)
    patch_height, patch_width = encoder.patch_embed.patch_size
    height, width = image.shape[-2:]
    padded = nn.functional.pad(image, (0, -width % patch_width, 0, -height % patch_height))
    rows, columns = padded.shape[-2] // patch_height, padded.shape[-1] // patch_width

    def reshape(tokens: torch.Tensor) -> torch.Tensor:
        patches = tokens[:, encoder.num_prefix_tokens :]
        return patches.reshape(tokens.shape[0], rows, columns, tokens.shape[-1]).permute(0, 3, 1, 2).float()

    layer = encoder.get_submodule(f"blocks.{len(encoder.blocks) - 1}.norm1")
    heat = tensor_heatmap(classifier, padded, layer, classifier.labels.index(prediction), reshape)
    return heat[:height, :width]


def heatmap_overlay(heatmap: np.ndarray, mask: Image.Image, box: tuple[int, int, int, int]) -> str | None:
    left, top, right, bottom = box
    aligned = Image.new("F", mask.size)
    resized = Image.fromarray(heatmap).resize((right - left, bottom - top), Image.Resampling.BILINEAR)
    aligned.paste(resized, (left, top))
    strength = np.clip((np.asarray(aligned) - HEAT_THRESHOLD) / (1 - HEAT_THRESHOLD), 0, 1)
    strength = np.where(np.asarray(mask) > 0, strength, 0)

    alpha = np.rint(np.sqrt(strength) * MAX_HEATMAP_ALPHA).astype(np.uint8)
    if not alpha.any():
        return None
    green = np.rint(100 + 139 * strength).astype(np.uint8)
    blue = np.rint(75 + 62 * strength).astype(np.uint8)
    rgba = np.stack((np.full_like(alpha, 255), green, blue, alpha), axis=-1)
    with BytesIO() as buffer:
        Image.fromarray(rgba).save(buffer, format="PNG")
        return b64encode(buffer.getvalue()).decode("ascii")
