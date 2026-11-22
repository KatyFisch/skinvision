"""Render the lesion mask and outline for the photo demo."""

from base64 import b64encode
from io import BytesIO

import numpy as np
from PIL import Image
from skimage.measure import approximate_polygon, find_contours

BACKGROUND_ALPHA = round(255 * 0.85)


def overlay_png(mask: Image.Image) -> str:
    overlay = Image.new("RGBA", mask.size)
    overlay.putalpha(mask.point(lambda value: 0 if value else BACKGROUND_ALPHA))
    with BytesIO() as buffer:
        overlay.save(buffer, format="PNG")
        return b64encode(buffer.getvalue()).decode("ascii")


def outline_path(mask: Image.Image) -> str:
    # Padding closes contours that touch the crop edge.
    padded = np.pad(np.asarray(mask), 1)
    paths = []
    for contour in find_contours(padded, level=127.5, fully_connected="high"):
        polygon = approximate_polygon(contour, tolerance=0.35)
        points = " ".join(f"{column - 0.5:g},{row - 0.5:g}" for row, column in polygon[:-1])
        paths.append(f"M{points}Z")
    return "".join(paths)
