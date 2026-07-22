"""Pipeline orchestrator.

Coordinates the full prediction pipeline by chaining the image reader,
preprocessor, model loader, and Grad-CAM generator into a single call.
"""

import numpy as np

from grad_cam import grad_cam
from load_model import model_fun
from preprocess_img import preprocess


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
    batch_array_img = preprocess(array)
    model = model_fun()
    prediction = np.argmax(model.predict(batch_array_img))
    proba = np.max(model.predict(batch_array_img)) * 100
    label = ""
    if prediction == 0:
        label = "bacteriana"
    if prediction == 1:
        label = "normal"
    if prediction == 2:
        label = "viral"
    heatmap = grad_cam(array)
    return (label, proba, heatmap)
