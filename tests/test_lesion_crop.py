from unittest import TestCase

import numpy as np
from PIL import Image

from classifiers.lesion_crop import crop_with_skin_margin


class LesionCropTest(TestCase):
    def test_crop_keeps_an_edge_lesion_in_frame_without_stretching(self) -> None:
        photo = Image.new("RGB", (600, 450), (180, 180, 180))
        mask = Image.new("L", photo.size)
        bounds = (580, 440, 600, 450)
        photo.paste((255, 0, 0), bounds)
        mask.paste(255, bounds)

        crop = crop_with_skin_margin(photo, mask)
        pixels = np.asarray(crop)
        y, x = np.where((pixels[:, :, 0] > 220) & (pixels[:, :, 1] < 90))
        width, height = x.max() - x.min() + 1, y.max() - y.min() + 1

        self.assertEqual(crop.size, photo.size)
        self.assertEqual((x.max(), y.max()), (599, 449))
        self.assertAlmostEqual(width / height, 2, delta=0.05)
        self.assertEqual(crop.getpixel((0, 0)), (180, 180, 180))
