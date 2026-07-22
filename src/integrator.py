"""Pipeline orchestrator.

Coordinates the full prediction pipeline by chaining the image reader,
preprocessor, model loader, and Grad-CAM generator into a single call.
"""

import logging

import numpy as np

from grad_cam import grad_cam
from load_model import model_fun
from preprocess_img import preprocess

logger = logging.getLogger(__name__)


def predict(array: np.ndarray) -> tuple[str, float, np.ndarray]:
    """Run the full prediction pipeline on an image array.

    Args:
        array: Input RGB image array (H, W, 3).

    Returns:
        Tuple of:
        - label: one of "bacteriana", "normal", or "viral".
        - probability: confidence percentage (0-100).
        - heatmap: RGB heatmap-overlaid image (512, 512, 3) as uint8.
    """
    logger.info("Predict start, input shape=%s dtype=%s", array.shape, array.dtype)
    batch_array_img = preprocess(array)
    model = model_fun()
    preds = model.predict(batch_array_img, verbose=0)
    prediction = np.argmax(preds)
    proba = np.max(preds) * 100
    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"
    logger.info("Prediction: label=%s probability=%.2f%%", label, proba)
    heatmap = grad_cam(array, img=batch_array_img, model=model)
    logger.info("Heatmap shape=%s dtype=%s", heatmap.shape, heatmap.dtype)
    return (label, proba, heatmap)
