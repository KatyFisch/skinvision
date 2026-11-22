"""Fit the mask's bounding box and surrounding skin into the clinical frame."""

import math

from PIL import Image, ImageOps

PHOTO_SIZE = (600, 450)
SKIN_MARGIN_PX = 12
CROP_MARGIN_FRACTION = 0.25
PADDED_CROP_SETTINGS = {"margin_fraction_of_longest_side": CROP_MARGIN_FRACTION, "minimum_margin_px": SKIN_MARGIN_PX}


def frame_photo(photo: Image.Image) -> Image.Image:
    if photo.size == PHOTO_SIZE:
        return photo
    return ImageOps.fit(photo, PHOTO_SIZE, Image.Resampling.LANCZOS)


def skin_margin_box(photo: Image.Image, mask: Image.Image) -> tuple[int, int, int, int] | None:
    bounds = mask.getbbox()
    if bounds is None:
        return None
    left, top, right, bottom = bounds
    margin = max(SKIN_MARGIN_PX, round(max(right - left, bottom - top) * CROP_MARGIN_FRACTION))
    left, top = max(0, left - margin), max(0, top - margin)
    right, bottom = min(photo.width, right + margin), min(photo.height, bottom + margin)

    # Enlarge the padded box to 4:3 and shift it inside the frame before resizing.
    units = math.ceil(max((right - left) / 4, (bottom - top) / 3))
    width, height = 4 * units, 3 * units
    x = min(max(0, (left + right - width) // 2), photo.width - width)
    y = min(max(0, (top + bottom - height) // 2), photo.height - height)
    return (x, y, x + width, y + height)


def crop_with_skin_margin(photo: Image.Image, mask: Image.Image) -> Image.Image:
    box = skin_margin_box(photo, mask)
    return photo if box is None else photo.crop(box).resize(PHOTO_SIZE, Image.Resampling.LANCZOS)
