"""Grad-CAM heatmap generation.

Computes a class-discriminative localisation heatmap using eager-mode
tf.GradientTape (TF2-native). No graph-mode hacks needed.
"""

import logging

import cv2
import numpy as np
import tensorflow as tf

from load_model import model_fun  # noqa: E402
from preprocess_img import preprocess  # noqa: E402

logger = logging.getLogger(__name__)


def grad_cam(
    array: np.ndarray,
    img: np.ndarray | None = None,
    model: tf.keras.Model | None = None,
) -> np.ndarray:
    """Generate a Grad-CAM heatmap overlaid on the input image.

    Args:
        array: Input RGB image array (H, W, 3).
        img: Preprocessed tensor (1, 512, 512, 1). If omitted, computed
             from *array* via ``preprocess``.
        model: Loaded Keras model. If omitted, loaded from disk.

    Returns:
        RGB heatmap-overlaid image of shape (512, 512, 3) as uint8.
    """
    if img is None:
        img = preprocess(array)
    if model is None:
        model = model_fun()
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)

    preds = model.predict(img_tensor, verbose=0)
    argmax = np.argmax(preds[0])
    logger.info("Grad-CAM argmax class=%s", argmax)

    last_conv_layer = model.get_layer("conv10_thisone")
    grad_model = tf.keras.Model(
        inputs=model.inputs,
        outputs=[last_conv_layer.output] + model.outputs,
    )

    with tf.GradientTape() as tape:
        conv_output, preds_output = grad_model(img_tensor, training=False)
        class_output = preds_output[:, argmax]

    grads = tape.gradient(class_output, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    logger.info("Grad-CAM grads shape=%s pooled_grads shape=%s", grads.shape, pooled_grads.shape)

    conv_layer_output_value = np.array(conv_output[0])
    pooled_grads_value = np.asarray(pooled_grads)

    n_filters = conv_layer_output_value.shape[-1]
    for filters in range(n_filters):
        conv_layer_output_value[:, :, filters] *= pooled_grads_value[filters]
    heatmap = np.mean(conv_layer_output_value, axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap) + 1e-8
    heatmap = cv2.resize(heatmap, (512, 512))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    img2 = cv2.resize(array, (512, 512))
    hif = 0.8
    transparency = heatmap * hif
    transparency = transparency.astype(np.uint8)
    superimposed_img = cv2.add(transparency, img2)
    superimposed_img = superimposed_img.astype(np.uint8)
    logger.info("Grad-CAM completed")
    return superimposed_img[:, :, ::-1]
