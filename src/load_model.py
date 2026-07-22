"""Model loading and validation.

Loads the pre-trained CNN from disk and validates its integrity.
"""

import tensorflow as tf

MODEL_PATH = "models/conv_MLP_84.h5"


def model_fun() -> tf.keras.Model:
    """Load and return the pre-trained CNN model.

    Returns:
        Loaded tf.keras.Model instance.

    Raises:
        FileNotFoundError: If the model file is not found at MODEL_PATH.
    """
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except OSError as exc:
        raise FileNotFoundError(
            f"No se encontró el modelo '{MODEL_PATH}'. "
            "Asegúrate de que el archivo .h5 está en el directorio del proyecto."
        ) from exc
