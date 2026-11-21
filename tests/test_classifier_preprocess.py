from unittest import TestCase

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional

from classifiers.encoders import CHECKPOINTS, build_preprocess


class ImagePreprocessTest(TestCase):
    def test_full_frame_leaves_center_pixels_identical(self) -> None:
        pixels = np.random.default_rng(42).integers(0, 256, size=(450, 600, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        standard = build_preprocess("resnet18")(image)
        full_frame = build_preprocess("resnet18", full_frame=True)(image)

        self.assertEqual(tuple(full_frame.shape), (3, 256, 341))
        self.assertEqual(tuple(standard.shape), (3, 224, 224))
        torch.testing.assert_close(functional.center_crop(full_frame, [224, 224]), standard, rtol=0, atol=0)

    def test_native_resolution_preserves_pixels_before_normalization(self) -> None:
        pixels = np.random.default_rng(42).integers(0, 256, size=(450, 600, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        native = build_preprocess("resnet18", full_frame=True, native_resolution=True)(image)
        standard = CHECKPOINTS["resnet18"][0].transforms()
        unnormalized = native * torch.tensor(standard.std)[:, None, None] + torch.tensor(standard.mean)[:, None, None]

        self.assertEqual(tuple(native.shape), (3, 450, 600))
        torch.testing.assert_close((unnormalized * 255).round().to(torch.uint8), functional.pil_to_tensor(image), rtol=0, atol=0)
