"""Reusable Tkinter view components built with CustomTkinter.

Each frame is a self-contained widget with its own layout (grid/pack).
Frames communicate with the controller through public methods and
callbacks passed at construction time.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING

import customtkinter as ctk
from PIL import Image

from gui.theme import DIAGNOSIS_COLORS, TITLE

if TYPE_CHECKING:
    import numpy as np

logger = logging.getLogger(__name__)

# ── Helpers ──────────────────────────────────────────────────────────


def _ctk_image(pil_img: Image.Image, size: tuple[int, int]) -> ctk.CTkImage:
    return ctk.CTkImage(light_image=pil_img, dark_image=pil_img, size=size)


# ── Header ────────────────────────────────────────────────────────────


class HeaderFrame(ctk.CTkFrame):
    """Top bar with title and theme toggle."""

    def __init__(self, master: ctk.CTk, on_toggle_theme: Callable[[], None], **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self._title = ctk.CTkLabel(
            self,
            text=TITLE,
            font=ctk.CTkFont(size=16, weight="bold"),
            anchor="center",
        )
        self._title.grid(row=0, column=0, padx=20, pady=12, sticky="ew")

        self._theme_btn = ctk.CTkButton(
            self,
            text="🌙",
            width=40,
            command=on_toggle_theme,
        )
        self._theme_btn.grid(row=0, column=1, padx=(0, 16), pady=8)

    def set_theme_icon(self, icon: str) -> None:
        self._theme_btn.configure(text=icon)


# ── Image / Heatmap previews ──────────────────────────────────────────


class _ImageFrame(ctk.CTkFrame):
    """Base for a labelled image preview panel."""

    def __init__(self, master: ctk.CTkFrame, title: str, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self._label = ctk.CTkLabel(self, text=title, font=ctk.CTkFont(weight="bold"))
        self._label.grid(row=0, column=0, pady=(8, 4))

        self._image_label = ctk.CTkLabel(self, text="")
        self._image_label.grid(row=1, column=0, padx=12, pady=(0, 12), sticky="nsew")

    def show(self, pil_img: Image.Image | None, size: tuple[int, int] = (320, 320)) -> None:
        if pil_img is None:
            self._image_label.configure(image=None, text="")
            return
        ctk_img = _ctk_image(pil_img, size)
        self._image_label.configure(image=ctk_img, text="")


class ImagePreviewFrame(_ImageFrame):
    """Preview panel for the original radiograph."""

    def __init__(self, master: ctk.CTkFrame, **kwargs):
        super().__init__(master, title="Imagen Radiográfica", **kwargs)


class HeatmapPreviewFrame(_ImageFrame):
    """Preview panel for the Grad-CAM heatmap."""

    def __init__(self, master: ctk.CTkFrame, **kwargs):
        super().__init__(master, title="Imagen con Heatmap", **kwargs)


# ── Results ────────────────────────────────────────────────────────────


class ResultsFrame(ctk.CTkFrame):
    """Patient ID input, diagnosis badge, and probability display."""

    def __init__(self, master: ctk.CTkFrame, **kwargs):
        super().__init__(master, **kwargs)
        for c in range(4):
            self.grid_columnconfigure(c, weight=0)
        self.grid_columnconfigure(3, weight=1)

        ctk.CTkLabel(self, text="Cédula Paciente:", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0, padx=(16, 4), pady=10, sticky="w"
        )
        self._id_entry = ctk.CTkEntry(self, width=120)
        self._id_entry.grid(row=0, column=1, padx=4, pady=10, sticky="w")
        self._id_entry.focus_set()

        self._badge = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=14, weight="bold"))
        self._badge.grid(row=0, column=2, padx=(24, 4), pady=10, sticky="w")

        self._prob_label = ctk.CTkLabel(self, text="", font=ctk.CTkFont(size=14))
        self._prob_label.grid(row=0, column=3, padx=4, pady=10, sticky="w")

    def show(self, label: str, proba: float) -> None:
        colour = DIAGNOSIS_COLORS.get(label, "#888888")
        self._badge.configure(text=f"● {label}", text_color=colour)
        self._prob_label.configure(text=f"Probabilidad: {proba:.2f}%")

    def clear(self) -> None:
        self._badge.configure(text="")
        self._prob_label.configure(text="")

    def get_patient_id(self) -> str:
        return self._id_entry.get()

    def reset_id(self) -> None:
        self._id_entry.delete(0, "end")


# ── Controls ──────────────────────────────────────────────────────────


class ControlFrame(ctk.CTkFrame):
    """Action buttons row."""

    def __init__(self, master: ctk.CTkFrame, callbacks: dict[str, Callable], **kwargs):
        super().__init__(master, **kwargs)
        for i in range(5):
            self.grid_columnconfigure(i, weight=1)

        self._buttons: dict[str, ctk.CTkButton] = {}
        configs = [
            ("Cargar", "load", callbacks.get("load")),
            ("Predecir", "predict", callbacks.get("predict")),
            ("Guardar", "save", callbacks.get("save")),
            ("PDF", "pdf", callbacks.get("pdf")),
            ("Borrar", "delete", callbacks.get("delete")),
        ]
        for idx, (text, key, cmd) in enumerate(configs):
            btn = ctk.CTkButton(self, text=text, command=cmd)
            btn.grid(row=0, column=idx, padx=6, pady=10, sticky="ew")
            if key == "predict":
                btn.configure(state="disabled")
            self._buttons[key] = btn

    def enable_predict(self) -> None:
        self._buttons["predict"].configure(state="normal")

    def disable_all(self) -> None:
        for btn in self._buttons.values():
            btn.configure(state="disabled")

    def enable_all(self) -> None:
        for btn in self._buttons.values():
            btn.configure(state="normal")
        self._buttons["predict"].configure(state="normal")


# ── Status Bar ────────────────────────────────────────────────────────


class StatusBar(ctk.CTkFrame):
    """Bottom bar with contextual message and indeterminate progress."""

    def __init__(self, master: ctk.CTkFrame, **kwargs):
        super().__init__(master, **kwargs)
        self.grid_columnconfigure(0, weight=1)

        self._message = ctk.CTkLabel(self, text="Listo", anchor="w", font=ctk.CTkFont(size=12))
        self._message.grid(row=0, column=0, padx=12, pady=4, sticky="ew")

        self._progress = ctk.CTkProgressBar(self, mode="indeterminate", width=160)
        self._progress.grid(row=0, column=1, padx=(0, 12), pady=4)
        self._progress.stop()
        self._progress.grid_remove()

    def set_message(self, text: str) -> None:
        self._message.configure(text=text)

    def start_progress(self, message: str = "Procesando...") -> None:
        self._message.configure(text=message)
        self._progress.grid()
        self._progress.start()

    def stop_progress(self, message: str = "Listo") -> None:
        self._progress.stop()
        self._progress.grid_remove()
        self._message.configure(text=message)


# ── PDF Dialog (modal, no tkcap) ─────────────────────────────────────


class PdfDialog(ctk.CTkToplevel):
    """Modal dialog that shows a report preview and saves a PDF using Pillow.

    Args:
        parent: The main application window.
        label: Diagnosis label.
        proba: Confidence percentage.
        heatmap: Grad-CAM heatmap array (H, W, 3).
        original: Original image array (H, W, 3).
        patient_id: Patient identification string.
    """

    def __init__(
        self,
        parent: ctk.CTk,
        label: str,
        proba: float,
        heatmap: np.ndarray,
        original: np.ndarray,
        patient_id: str,
    ) -> None:
        super().__init__(parent)
        self._label = label
        self._proba = proba
        self._heatmap = heatmap
        self._original = original
        self._patient_id = patient_id

        self.title("Vista previa del reporte")
        self.geometry("500x480")
        self.resizable(False, False)
        self.grab_set()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(2, weight=1)

        ctk.CTkLabel(
            self,
            text="Reporte de Diagnóstico",
            font=ctk.CTkFont(size=16, weight="bold"),
        ).grid(row=0, column=0, pady=(16, 8))

        preview_frame = ctk.CTkFrame(self)
        preview_frame.grid(row=1, column=0, padx=24, pady=4, sticky="ew")
        preview_frame.grid_columnconfigure((0, 1), weight=1)

        ctk.CTkLabel(preview_frame, text="Radiografía", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=0
        )
        ctk.CTkLabel(preview_frame, text="Mapa de Calor", font=ctk.CTkFont(weight="bold")).grid(
            row=0, column=1
        )

        orig_pil = Image.fromarray(original).resize((180, 180), Image.LANCZOS)
        heat_pil = Image.fromarray(heatmap).resize((180, 180), Image.LANCZOS)
        ctk.CTkLabel(preview_frame, image=_ctk_image(orig_pil, (180, 180)), text="").grid(
            row=1, column=0, padx=8, pady=4
        )
        ctk.CTkLabel(preview_frame, image=_ctk_image(heat_pil, (180, 180)), text="").grid(
            row=1, column=1, padx=8, pady=4
        )

        colour = DIAGNOSIS_COLORS.get(label, "#888888")
        ctk.CTkLabel(
            self,
            text=f"Diagnóstico:  ● {label}   ({proba:.2f}%)",
            font=ctk.CTkFont(size=14, weight="bold"),
            text_color=colour,
        ).grid(row=2, column=0, pady=12)

        btn_frame = ctk.CTkFrame(self, fg_color="transparent")
        btn_frame.grid(row=3, column=0, pady=(4, 16))
        ctk.CTkButton(btn_frame, text="💾 Guardar PDF", command=self._save_pdf).grid(
            row=0, column=0, padx=6
        )
        ctk.CTkButton(btn_frame, text="Cancelar", command=self.destroy).grid(
            row=0, column=1, padx=6
        )

    def _save_pdf(self) -> None:
        from tkinter.messagebox import showerror, showinfo

        try:
            orig_pil = Image.fromarray(self._original).resize((512, 512), Image.LANCZOS)
            heat_pil = Image.fromarray(self._heatmap)
            combined = Image.new("RGB", (1024, 512))
            combined.paste(orig_pil, (0, 0))
            combined.paste(heat_pil, (512, 0))

            pdf_path = f"Reporte_{self._patient_id or 'pdf'}.pdf"
            combined.save(pdf_path, "PDF")
            logger.info("PDF saved to %s", pdf_path)
            showinfo(title="PDF", message=f"PDF generado: {pdf_path}")
            self.destroy()
        except Exception:
            logger.exception("Failed to create PDF")
            showerror(title="Error", message="No se pudo generar el PDF.")
