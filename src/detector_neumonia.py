#!/usr/bin/env python
"""Backward-compatible facade for the pneumonia detection pipeline.

Re-exports all functions from the refactored modules so that existing
code and tests can still ``import detector_neumonia as dn`` and call
``dn.preprocess``, ``dn.predict``, etc. without changes.
"""

import logging

from grad_cam import grad_cam  # noqa: F401
from load_model import model_fun  # noqa: F401
from preprocess_img import preprocess  # noqa: F401
from read_img import read_dicom as read_dicom_file  # noqa: F401
from read_img import read_jpg as read_jpg_file  # noqa: F401

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logger.info("Starting application")
    from gui.controller import main as gui_main

    gui_main()
    return 0


if __name__ == "__main__":
    main()
