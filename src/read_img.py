"""Medical image reading utilities.

Supports DICOM (.dcm), JPEG, and PNG formats.
Each function returns a tuple (numpy array, PIL Image) for downstream
processing and UI display.
"""

import logging

import cv2
import numpy as np
import pydicom
from PIL import Image

logger = logging.getLogger(__name__)


def read_dicom(path: str) -> tuple[np.ndarray, Image.Image]:
    """Read and preprocess a DICOM file.

    Args:
        path: Path to the .dcm file.

    Returns:
        Tuple of (RGB numpy array uint8, PIL Image for display).

    Raises:
        FileNotFoundError: If the file does not exist.
        OSError: If the DICOM file is corrupted or unreadable.
    """
    logger.info("Reading DICOM: %s", path)
    img = pydicom.dcmread(path)
    img_array = img.pixel_array
    logger.info(
        "DICOM shape=%s dtype=%s min=%s max=%s",
        img_array.shape,
        img_array.dtype,
        img_array.min(),
        img_array.max(),
    )

    range_val = max(img_array.max() - img_array.min(), 1)
    img_normalized = (img_array.astype(float) - img_array.min()) / range_val * 255
    img_normalized = np.clip(img_normalized, 0, 255).astype(np.uint8)
    img2show = Image.fromarray(img_normalized)

    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / max(img2.max(), 1)) * 255.0
    img2 = np.uint8(img2)
    img_rgb = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    return img_rgb, img2show


def read_jpg(path: str) -> tuple[np.ndarray, Image.Image]:
    """Read a standard image file (JPEG, PNG, etc.).

    Args:
        path: Path to the image file.

    Returns:
        Tuple of (numpy array uint8, PIL Image for display).

    Raises:
        ValueError: If the file cannot be read (cv2 returns None).
    """
    logger.info("Reading image: %s", path)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen en '{path}'.")
    img_array = np.asarray(img)
    logger.info("Image shape=%s dtype=%s", img_array.shape, img_array.dtype)

    img_rgb = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img2show = Image.fromarray(img_rgb)

    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / max(img2.max(), 1)) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show
