"""Model loading and validation.

Loads the pre-trained CNN from disk and validates its integrity.
"""

import logging

import tensorflow as tf

logger = logging.getLogger(__name__)

MODEL_PATH = "models/conv_MLP_84.h5"


def model_fun() -> tf.keras.Model:
    """Load and return the pre-trained CNN model.

    Returns:
        Loaded tf.keras.Model instance.

    Raises:
        FileNotFoundError: If the model file is not found at MODEL_PATH.
    """
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        logger.info("Model loaded from %s", MODEL_PATH)
        return model
    except OSError as exc:
        logger.error("Failed to load model from %s: %s", MODEL_PATH, exc)
        raise FileNotFoundError(
            f"No se encontró el modelo '{MODEL_PATH}'. "
            "Asegúrate de que el archivo .h5 está en el directorio del proyecto."
        ) from exc
