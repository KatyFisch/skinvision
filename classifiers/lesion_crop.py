"""Fit the mask's bounding box and surrounding skin into the clinical frame."""

import math

from PIL import Image

PHOTO_SIZE = (600, 450)
SKIN_MARGIN_PX = 12
CROP_MARGIN_FRACTION = 0.25


def crop_with_skin_margin(photo: Image.Image, mask: Image.Image) -> Image.Image:
    bounds = mask.getbbox()
    if bounds is None:
        return photo
    left, top, right, bottom = bounds
    margin = max(SKIN_MARGIN_PX, round(max(right - left, bottom - top) * CROP_MARGIN_FRACTION))
    left, top = max(0, left - margin), max(0, top - margin)
    right, bottom = min(photo.width, right + margin), min(photo.height, bottom + margin)

    # Enlarge the padded box to 4:3 and shift it inside the frame before resizing.
    units = math.ceil(max((right - left) / 4, (bottom - top) / 3))
    width, height = 4 * units, 3 * units
    x = min(max(0, (left + right - width) // 2), photo.width - width)
    y = min(max(0, (top + bottom - height) // 2), photo.height - height)
    return photo.crop((x, y, x + width, y + height)).resize(PHOTO_SIZE, Image.Resampling.LANCZOS)
