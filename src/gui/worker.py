"""Background task runner with thread-safe UI callbacks.

Uses ``ThreadPoolExecutor`` to offload heavy operations (model inference,
image I/O) off the main thread, and schedules result callbacks via
``root.after()`` so they run safely on the Tkinter event loop.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

logger = logging.getLogger(__name__)


class AsyncWorker:
    """Execute *task* in a background thread and notify the UI on completion.

    Args:
        max_workers: Maximum thread pool size (default 2).
    """

    def __init__(self, max_workers: int = 2) -> None:
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def run(
        self,
        task: Callable[[], Any],
        on_done: Callable[[Any], None] | None = None,
        on_error: Callable[[Exception], None] | None = None,
    ) -> None:
        """Submit *task* to the thread pool.

        Args:
            task: Zero-argument callable to run in the background.
            on_done: Called on the main thread with the result on success.
            on_error: Called on the main thread with the exception on failure.
        """
        import tkinter as tk

        def _wrapper() -> None:
            try:
                result = task()
                if on_done is not None:
                    try:
                        root = tk._default_root
                        if root is not None:
                            root.after(0, on_done, result)
                    except Exception:
                        logger.exception("Failed to schedule on_done callback")
            except Exception as exc:
                logger.exception("Background task failed")
                if on_error is not None:
                    try:
                        root = tk._default_root
                        if root is not None:
                            root.after(0, on_error, exc)
                    except Exception:
                        logger.exception("Failed to schedule on_error callback")

        self._executor.submit(_wrapper)
