from typing import cast
from unittest import TestCase

import numpy as np
import torch
from PIL import Image
from torchvision.models import ResNet18_Weights
from torchvision.transforms import functional

from classifiers.dermfm_zero import clinical_preprocess
from classifiers.encoders import imagenet_normalize


class ImagePreprocessTest(TestCase):
    def test_native_resolution_preserves_pixels_before_normalization(self) -> None:
        pixels = np.random.default_rng(42).integers(0, 256, size=(450, 600, 3), dtype=np.uint8)
        image = Image.fromarray(pixels)
        standard = ResNet18_Weights.IMAGENET1K_V1.transforms()
        for name, preprocess, mean, std in (
            ("ImageNet", imagenet_normalize, standard.mean, standard.std),
            ("DermFM", clinical_preprocess(), (0.5,) * 3, (0.5,) * 3),
        ):
            with self.subTest(encoder=name):
                native = cast(torch.Tensor, preprocess(image))
                unnormalized = native * torch.tensor(std)[:, None, None] + torch.tensor(mean)[:, None, None]
                self.assertEqual(tuple(native.shape), (3, 450, 600))
                torch.testing.assert_close((unnormalized * 255).round().to(torch.uint8), functional.pil_to_tensor(image), rtol=0, atol=0)
