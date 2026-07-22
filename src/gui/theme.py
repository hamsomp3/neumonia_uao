"""Application theme configuration.

Defines the medical colour palette, diagnosis badge colours, and
CustomTkinter appearance setup.
"""

DIAGNOSIS_COLORS = {
    "normal": "#2ECC71",
    "viral": "#E67E22",
    "bacteriana": "#E74C3C",
}

COLORS = {
    "primary": "#2E86AB",
    "secondary": "#A23B72",
    "bg_light": "#F0F4F8",
    "bg_dark": "#1A1A2E",
    "card_light": "#FFFFFF",
    "card_dark": "#2D2D44",
    "text_light": "#1A1A2E",
    "text_dark": "#E8E8E8",
}

TITLE = "SOFTWARE PARA EL APOYO AL DIAGNÓSTICO MÉDICO DE NEUMONÍA"
WINDOW_SIZE = "1000x680"
MIN_WINDOW_SIZE = (800, 560)


def configure_ctk() -> None:
    """Set CustomTkinter defaults before the root window is created."""
    import customtkinter as ctk

    ctk.set_appearance_mode("light")
    ctk.set_default_color_theme("blue")
