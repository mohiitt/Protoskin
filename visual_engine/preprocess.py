"""Image conditioning for ControlNet. Person A owns this file.

Canny is the MVP/default control mode: it works well on clean line
drawings, wireframes, and product-outline sketches and is what the
hackathon demo path uses. Depth preprocessing is optional/out of scope
until Canny is proven reliable (see the implementation plan, section 7).
"""

from __future__ import annotations

import numpy as np
from PIL import Image

# Conservative default thresholds tuned for clean sketches/wireframes on
# a plain background. Lower thresholds pick up more (often noisy) detail;
# raise them if a busy/textured input produces a cluttered edge map.
_DEFAULT_LOW_THRESHOLD = 100
_DEFAULT_HIGH_THRESHOLD = 200

# SDXL and its ControlNet variants expect image dimensions that are
# multiples of 8; round up so both the source and the edge map stay
# pixel-aligned with what the pipeline will generate.
_SIZE_MULTIPLE = 8


def _round_up(value: int, multiple: int) -> int:
    return max(multiple, ((value + multiple - 1) // multiple) * multiple)


def load_and_resize(image_path: str, max_side: int = 1024) -> Image.Image:
    """Load an image and resize it to SDXL-friendly dimensions.

    Keeps aspect ratio, caps the longer side at ``max_side``, and rounds
    both dimensions to a multiple of 8.
    """
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    scale = min(1.0, max_side / max(width, height))
    new_width = _round_up(max(1, round(width * scale)), _SIZE_MULTIPLE)
    new_height = _round_up(max(1, round(height * scale)), _SIZE_MULTIPLE)
    if (new_width, new_height) != (width, height):
        image = image.resize((new_width, new_height), Image.LANCZOS)
    return image


def make_canny(
    image_path: str,
    low_threshold: int = _DEFAULT_LOW_THRESHOLD,
    high_threshold: int = _DEFAULT_HIGH_THRESHOLD,
    max_side: int = 1024,
) -> Image.Image:
    """Return a Canny edge map as a 3-channel PIL image for ControlNet.

    Imports OpenCV lazily so importing this module never requires the ML
    dependencies unless a real edge map is actually requested.
    """
    import cv2

    image = load_and_resize(image_path, max_side=max_side)
    array = np.array(image)
    edges = cv2.Canny(array, low_threshold, high_threshold)
    edges_rgb = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges_rgb)
