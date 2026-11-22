"""Load the lesion U-Net and produce a mask aligned with the clinical photo."""

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import segmentation_models_pytorch as smp
import torch
from PIL import Image
from segmentation_models_pytorch.base.heads import SegmentationHead
from segmentation_models_pytorch.base.modules import Activation, Attention, Conv2dReLU
from segmentation_models_pytorch.decoders.unet.decoder import UnetDecoder, UnetDecoderBlock
from segmentation_models_pytorch.encoders.resnet import ResNetEncoder
from torch import nn
from torchvision.models.resnet import BasicBlock

from classifiers.encoders import imagenet_normalize

SEGMENTATION_SIZE = (512, 384)

# The notebook also exports a serialized U-Net. Limit that loader to the
# exact module classes used by this checkpoint, then use its state dict.
UNET_CLASSES = [
    smp.Unet,
    SegmentationHead,
    Activation,
    Attention,
    Conv2dReLU,
    UnetDecoder,
    UnetDecoderBlock,
    ResNetEncoder,
    BasicBlock,
    nn.ReLU,
    nn.BatchNorm2d,
    nn.ModuleList,
    nn.Sequential,
    nn.Conv2d,
    nn.Identity,
    nn.MaxPool2d,
]


def load_checkpoint(path: Path) -> nn.Module | dict[str, torch.Tensor]:
    with torch.serialization.safe_globals(UNET_CLASSES):
        if path.is_file():
            return torch.load(path, map_location="cpu", weights_only=True)

        # A .pth opened as a ZIP may arrive as an extracted folder on macOS.
        with BytesIO() as packed:
            with ZipFile(packed, "w", strict_timestamps=False) as archive:
                for name in ("data.pkl", "byteorder", "version", ".format_version", ".storage_alignment", ".data/serialization_id"):
                    if (path / name).is_file():
                        archive.write(path / name, f"checkpoint/{name}")
                for storage in (path / "data").iterdir():
                    if storage.name.isdecimal():
                        archive.write(storage, f"checkpoint/data/{storage.name}")
            packed.seek(0)
            return torch.load(packed, map_location="cpu", weights_only=True)


class LesionSegmenter:
    def __init__(self, checkpoint: Path, device: torch.device) -> None:
        saved = load_checkpoint(checkpoint)
        model = smp.Unet(encoder_name="resnet18", encoder_weights=None, in_channels=3, classes=1)
        model.load_state_dict(saved.state_dict() if isinstance(saved, nn.Module) else saved)
        self.model = model.eval().to(device)
        self.device = device

    @torch.inference_mode()
    def mask(self, photo: Image.Image) -> Image.Image:
        resized = photo.resize(SEGMENTATION_SIZE, Image.Resampling.BILINEAR)
        tensor = imagenet_normalize(resized).unsqueeze(0).to(self.device)
        logits = self.model(tensor)
        aligned = nn.functional.interpolate(logits, size=(photo.height, photo.width), mode="bilinear", align_corners=False)
        foreground = aligned[0, 0].gt(0).to(torch.uint8).mul(255).cpu().numpy()
        return Image.fromarray(foreground)
