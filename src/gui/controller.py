"""Main application controller.

Wires together the domain modules (read_img, preprocess_img, load_model,
integrator, grad_cam) with the GUI views (views.py) and background task
executor (worker.py).  Maintains the current application state and
provides the callback handlers for all UI actions.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import customtkinter as ctk
import numpy as np
from PIL import Image

from gui.theme import configure_ctk
from gui.views import (
    ControlFrame,
    HeaderFrame,
    HeatmapPreviewFrame,
    ImagePreviewFrame,
    PdfDialog,
    ResultsFrame,
    StatusBar,
)
from gui.worker import AsyncWorker

logger = logging.getLogger(__name__)

_IS_DARK = False


def _toggle_dark() -> None:
    global _IS_DARK
    _IS_DARK = not _IS_DARK
    mode = "Dark" if _IS_DARK else "Light"
    ctk.set_appearance_mode(mode)


def _theme_icon() -> str:
    return "🌙" if not _IS_DARK else "☀"


class AppController:
    """Orchestrates the full GUI application.

    Args:
        root: The CustomTkinter root window.
    """

    def __init__(self, root: ctk.CTk) -> None:
        self.root = root
        self._worker = AsyncWorker()

        # ── application state ─────────────────────────────────────
        self._current_array: np.ndarray | None = None  # RGB uint8 for inference
        self._current_pil: Image.Image | None = None  # PIL for display
        self._heatmap_pil: Image.Image | None = None
        self._current_heatmap: np.ndarray | None = None  # array for PDF
        self._current_label: str | None = None
        self._current_proba: float | None = None

        self._build_ui()
        self._bind_shortcuts()
        logger.info("Application started")

    # ── UI construction ──────────────────────────────────────────

    def _build_ui(self) -> None:
        self.root.title("Diagnóstico de Neumonía")
        self.root.geometry("1000x680")
        self.root.minsize(800, 560)
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        self._header = HeaderFrame(self.root, on_toggle_theme=self._on_toggle_theme)
        self._header.grid(row=0, column=0, sticky="ew")

        preview_frames = ctk.CTkFrame(self.root)
        preview_frames.grid(row=1, column=0, padx=16, pady=(8, 0), sticky="ew")
        preview_frames.grid_columnconfigure((0, 1), weight=1)

        self._img_preview = ImagePreviewFrame(preview_frames)
        self._img_preview.grid(row=0, column=0, padx=(0, 8), sticky="nsew")

        self._heatmap_preview = HeatmapPreviewFrame(preview_frames)
        self._heatmap_preview.grid(row=0, column=1, padx=(8, 0), sticky="nsew")

        self._results = ResultsFrame(self.root)
        self._results.grid(row=2, column=0, padx=16, pady=(8, 0), sticky="ew")

        callbacks = {
            "load": self._on_load,
            "predict": self._on_predict,
            "save": self._on_save,
            "pdf": self._on_pdf,
            "delete": self._on_delete,
        }
        self._controls = ControlFrame(self.root, callbacks)
        self._controls.grid(row=3, column=0, padx=16, pady=(8, 0), sticky="ew")

        self._status = StatusBar(self.root)
        self._status.grid(row=4, column=0, padx=16, pady=(4, 12), sticky="ew")

    def _bind_shortcuts(self) -> None:
        self.root.bind("<Control-o>", lambda _: self._on_load())
        self.root.bind("<Control-p>", lambda _: self._on_predict())
        self.root.bind("<Control-s>", lambda _: self._on_save())
        self.root.bind("<Control-d>", lambda _: self._on_pdf())
        self.root.bind("<Delete>", lambda _: self._on_delete())
        self.root.bind("<BackSpace>", lambda _: self._on_delete())

    # ── Theme ────────────────────────────────────────────────────

    def _on_toggle_theme(self) -> None:
        _toggle_dark()
        self._header.set_theme_icon(_theme_icon())
        self._status.set_message(f"Modo {'oscuro' if _IS_DARK else 'claro'} activado")

    # ── Load image ───────────────────────────────────────────────

    def _on_load(self) -> None:
        from tkinter.filedialog import askopenfilename

        path = askopenfilename(
            title="Seleccionar imagen radiográfica",
            initialdir="images",
            filetypes=[
                ("Imágenes", "*.dcm *.jpg *.jpeg *.png"),
                ("DICOM", "*.dcm"),
                ("JPEG", "*.jpg *.jpeg"),
                ("PNG", "*.png"),
            ],
        )
        if not path:
            return

        def _task() -> tuple[np.ndarray, Image.Image, str]:
            from read_img import read_dicom, read_jpg

            _, ext = os.path.splitext(path)
            if ext.lower() in (".jpg", ".jpeg", ".png"):
                _array, _pil = read_jpg(path)
            else:
                _array, _pil = read_dicom(path)
            return _array, _pil, path

        def _done(result: tuple[np.ndarray, Image.Image, str]) -> None:
            self._current_array, self._current_pil, _ = result
            self._img_preview.show(self._current_pil, (320, 320))
            self._controls.enable_predict()
            self._status.set_message(f"Imagen cargada: {os.path.basename(result[2])}")
            self._on_clear_results()

        def _error(exc: Exception) -> None:
            from tkinter.messagebox import showerror

            logger.exception("Error loading image")
            showerror(title="Error de carga", message=str(exc))
            self._status.set_message("Error al cargar imagen")

        self._status.start_progress("Cargando imagen...")
        self._worker.run(_task, _done, _error)

    # ── Predict ──────────────────────────────────────────────────

    def _on_predict(self) -> None:
        if self._current_array is None:
            return

        array_copy = self._current_array.copy()

        def _task() -> tuple[str, float, np.ndarray]:
            from integrator import predict

            _label, _proba, _heatmap = predict(array_copy)
            return _label, _proba, _heatmap

        def _done(result: tuple[str, float, np.ndarray]) -> None:
            self._current_label, self._current_proba, self._current_heatmap = result
            heatmap_pil = Image.fromarray(self._current_heatmap)
            self._heatmap_pil = heatmap_pil
            self._heatmap_preview.show(self._heatmap_pil, (320, 320))
            self._results.show(result[0], result[1])
            self._status.set_message(f"Diagnóstico: {result[0]} ({result[1]:.2f}%)")

        def _error(exc: Exception) -> None:
            logging.getLogger(__name__).exception("Prediction failed")
            from tkinter.messagebox import showerror

            showerror(title="Error de predicción", message=str(exc))
            self._status.set_message("Error durante la predicción")

        self._status.start_progress("Ejecutando modelo...")
        self._controls.disable_all()
        self._worker.run(_task, _done, lambda exc: (_error(exc), self._controls.enable_all()))
        # Re-enable controls when prediction completes
        original_done = _done

        def _wrap_done(result: Any) -> None:
            original_done(result)
            self._controls.enable_all()

        # We rewire via the closure trick: use a wrapper
        def _done_wrapper(result: tuple[str, float, np.ndarray]) -> None:
            _done(result)
            self._controls.enable_all()

        self._worker.run(_task, _done_wrapper, _error)

    # ── Save CSV ─────────────────────────────────────────────────

    def _on_save(self) -> None:
        from tkinter.filedialog import asksaveasfilename
        from tkinter.messagebox import showerror, showinfo

        if self._current_label is None:
            showerror(title="Error", message="No hay resultados para guardar.")
            return

        path = asksaveasfilename(
            title="Guardar resultados",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv")],
        )
        if not path:
            return

        try:
            self._save_csv(path)
            showinfo(title="Guardado", message=f"Resultados guardados en:\n{path}")
            self._status.set_message(f"Resultados guardados: {os.path.basename(path)}")
        except Exception:
            logger.exception("Failed to save CSV")
            showerror(title="Error", message="No se pudo guardar el archivo.")

    def _save_csv(self, path: str) -> None:
        import csv

        patient_id = self._results.get_patient_id()
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Paciente", "Diagnóstico", "Probabilidad (%)"])
            writer.writerow([patient_id, self._current_label, f"{self._current_proba:.2f}"])

    # ── PDF ──────────────────────────────────────────────────────

    def _on_pdf(self) -> None:
        from tkinter.messagebox import showerror

        if self._current_heatmap is None or self._current_array is None:
            showerror(title="Error", message="No hay resultados para generar PDF.")
            return

        PdfDialog(
            self.root,
            label=self._current_label or "",
            proba=self._current_proba or 0.0,
            heatmap=self._current_heatmap,
            original=self._current_array,
            patient_id=self._results.get_patient_id(),
        )

    # ── Delete ───────────────────────────────────────────────────

    def _on_delete(self) -> None:
        self._current_array = None
        self._current_pil = None
        self._heatmap_pil = None
        self._current_heatmap = None
        self._current_label = None
        self._current_proba = None
        self._img_preview.show(None)
        self._heatmap_preview.show(None)
        self._results.clear()
        self._results.reset_id()
        self._status.set_message("Datos borrados")
        logger.info("Application state cleared")

    # ── Helpers ──────────────────────────────────────────────────

    def _on_clear_results(self) -> None:
        self._current_label = None
        self._current_proba = None
        self._current_heatmap = None
        self._heatmap_pil = None
        self._heatmap_preview.show(None)
        self._results.clear()


def main() -> None:
    """Application entry point."""
    configure_ctk()

    try:
        root = ctk.CTk()
        root.option_add("*Dialog.msg.font", "Helvetica 12")
        AppController(root)
        root.mainloop()
    except KeyboardInterrupt:
        logger.info("Application terminated by user")
    except Exception:
        logger.exception("Unhandled application error")
        raise
