"""Preprocessing pipeline for the CNN model.

Transforms a raw RGB image array into a normalized tensor
ready for model inference: resize, grayscale, CLAHE, normalise,
and expand dimensions to match the expected (1, 512, 512, 1) shape.
"""

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def preprocess(array: np.ndarray) -> np.ndarray:
    """Preprocess a raw image array for model inference.

    Pipeline:
        1. Resize to 512x512.
        2. Convert BGR to grayscale.
        3. Apply CLAHE (clipLimit=2.0, tileGridSize=(4,4)).
        4. Normalise pixel values to [0, 1].
        5. Expand dimensions to shape (1, 512, 512, 1).

    Args:
        array: Input RGB image array (H, W, 3).

    Returns:
        Tensor of shape (1, 512, 512, 1) with dtype float.

    Raises:
        TypeError: If input is not a valid image array.
        cv2.error: If OpenCV operations fail.
    """
    logger.info("Preprocess start, input shape=%s dtype=%s", array.shape, array.dtype)
    array = cv2.resize(array, (512, 512))
    array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    array = clahe.apply(array)
    array = array / 255
    array = np.expand_dims(array, axis=-1)
    array = np.expand_dims(array, axis=0)
    logger.info("Preprocess output shape=%s", array.shape)
    return array
