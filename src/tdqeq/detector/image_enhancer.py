import numpy as np
from PIL import Image, ImageEnhance

from tdqeq.types import PageBundle

# Tuning constants — adjust here only, never scattered across files
BRIGHTNESS_FACTOR = 1.05
CONTRAST_FACTOR = 1.8


def enhance(page: PageBundle) -> np.ndarray:
    """
    Enhance brightness and contrast of a page image for YOLO detection.

    Takes the RGB numpy array from a PageBundle, applies brightness and
    contrast enhancement, and returns an enhanced numpy array.

    Args:
        page: PageBundle containing the rendered page image (RGB np.ndarray)

    Returns:
        Enhanced image as np.ndarray (HxWx3, RGB, uint8)
    """
    pil_image = Image.fromarray(page.image)

    pil_image = ImageEnhance.Contrast(pil_image).enhance(CONTRAST_FACTOR)
    pil_image = ImageEnhance.Brightness(pil_image).enhance(BRIGHTNESS_FACTOR)

    return np.array(pil_image)
