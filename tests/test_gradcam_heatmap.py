from base64 import b64decode
from io import BytesIO
from unittest import TestCase

import numpy as np
from PIL import Image

from demo.gradcam_heatmap import HEAT_COLORS, MAX_HEATMAP_ALPHA, heatmap_overlay


class HeatmapOverlayTest(TestCase):
    def test_projects_heat_into_crop_and_clips_to_mask(self) -> None:
        heat = np.array([[0.17, 1.0], [0.8, 0.25]], dtype=np.float32)
        mask = Image.new("L", (10, 8))
        mask.paste(255, (4, 3, 6, 5))
        mask.putpixel((4, 4), 0)

        encoded = heatmap_overlay(heat, mask, (4, 3, 6, 5))
        assert encoded is not None
        with Image.open(BytesIO(b64decode(encoded))) as overlay:
            alpha = np.asarray(overlay.getchannel("A"))
            self.assertEqual(overlay.getpixel((5, 3)), (*HEAT_COLORS[-1], MAX_HEATMAP_ALPHA))
        self.assertEqual(alpha.shape, (8, 10))
        self.assertEqual(alpha[3, 4], 0)
        self.assertEqual(alpha[3, 5], MAX_HEATMAP_ALPHA)
        self.assertEqual(alpha[4, 4], 0)
        self.assertGreater(alpha[4, 5], 0)
        self.assertLess(alpha[4, 5], MAX_HEATMAP_ALPHA)
        self.assertFalse(alpha[np.asarray(mask) == 0].any())

    def test_omits_overlay_when_heat_is_below_threshold(self) -> None:
        heat = np.full((2, 2), 0.17, dtype=np.float32)
        self.assertIsNone(heatmap_overlay(heat, Image.new("L", (2, 2), 255), (0, 0, 2, 2)))

    def test_outside_hotspot_does_not_change_the_display_scale(self) -> None:
        heat = np.array([[0.25, 0.58], [0.62, 0.0]], dtype=np.float32)
        mask = Image.new("L", (2, 2), 255)
        mask.putpixel((1, 1), 0)
        expected = heatmap_overlay(heat, mask, (0, 0, 2, 2))
        heat[1, 1] = 1.0
        self.assertEqual(heatmap_overlay(heat, mask, (0, 0, 2, 2)), expected)
