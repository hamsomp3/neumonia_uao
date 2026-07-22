"""Medical image reading utilities.

Supports DICOM (.dcm), JPEG, and PNG formats.
Each function returns a tuple (numpy array, PIL Image) for downstream
processing and UI display.
"""

import cv2
import numpy as np
import pydicom
from PIL import Image


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
    img = pydicom.dcmread(path)
    img_array = img.pixel_array
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
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
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"No se pudo leer la imagen en '{path}'.")
    img_array = np.asarray(img)
    img2show = Image.fromarray(img_array)
    img2 = img_array.astype(float)
    img2 = (np.maximum(img2, 0) / img2.max()) * 255.0
    img2 = np.uint8(img2)
    return img2, img2show
