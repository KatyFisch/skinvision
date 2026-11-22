"""
Generate a Grad-CAM heatmap for a new image using the trained model
from explain_classification.ipynb.

The returned heatmap is a 2D NumPy array with values between 0 and 1.
No plot or heatmap overlay is created.

Example
-------
from gradcam import load_model, generate_heatmap

model = load_model("../models/mel-vs-nv-full.pth")
heatmap = generate_heatmap("path/to/new_image.jpg", model)

print(heatmap.shape)  # (image_height, image_width)
print(heatmap.min(), heatmap.max())
"""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget


class GradCAMWrapper(nn.Module):
    """Expose the first model output in the format expected by Grad-CAM."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)[0].reshape(-1, 1)


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

    cam = GradCAM(model=cam_model, target_layers=[target_layer])

    grayscale_cam = cam(input_tensor=input_tensor, targets=[ClassifierOutputTarget(target_category)])

    # Grad-CAM returns (batch, height, width); return the first image.
    heatmap = np.asarray(grayscale_cam[0], dtype=np.float32)
    return np.clip(heatmap, 0.0, 1.0)
