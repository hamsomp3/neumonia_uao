"""Grad-CAM heatmap generation.

Computes a class-discriminative localisation heatmap by looking at the
gradients of the target class flowing into the last convolutional layer.
"""

import cv2
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()
tf.compat.v1.experimental.output_all_intermediates(True)
from tensorflow.keras import backend as K  # noqa: E402

from load_model import model_fun  # noqa: E402
from preprocess_img import preprocess  # noqa: E402


def grad_cam(array: np.ndarray) -> np.ndarray:
    """Generate a Grad-CAM heatmap overlaid on the input image.

    Args:
        array: Input RGB image array (H, W, 3).

    Returns:
        RGB heatmap-overlaid image of shape (512, 512, 3) as uint8.
    """
    img = preprocess(array)
    model = model_fun()
    preds = model.predict(img)
    argmax = np.argmax(preds[0])
    output = model.output[:, argmax]
    last_conv_layer = model.get_layer("conv10_thisone")
    grads = K.gradients(output, last_conv_layer.output)[0]
    pooled_grads = K.mean(grads, axis=(0, 1, 2))
    iterate = K.function([model.input], [pooled_grads, last_conv_layer.output[0]])
    pooled_grads_value, conv_layer_output_value = iterate(img)
    for filters in range(64):
        conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[2]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)
    return superimposed_img[:, :, ::-1]
