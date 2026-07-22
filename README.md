<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-blue?logo=python" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/TensorFlow-2.21%2B-orange?logo=tensorflow" alt="TensorFlow">
  <img src="https://img.shields.io/badge/OpenCV-5.0%2B-brightgreen?logo=opencv" alt="OpenCV">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="MIT">
  <img src="https://img.shields.io/badge/uv-package%20manager-purple" alt="uv">
  <img src="https://img.shields.io/badge/tests-115%20passed-brightgreen" alt="Tests">
</p>

<h1 align="center">🫁 Neumonía Detector</h1>
<p align="center">
  <em>Deep Learning aplicado al diagnóstico rápido de neumonía en imágenes radiográficas de tórax</em>
</p>

---

## Clasificación

El modelo clasifica radiografías de tórax en **3 categorías**:

| Clase | Descripción |
|-------|-------------|
| **Neumonía Bacteriana** | Infección pulmonar causada por bacterias |
| **Neumonía Viral** | Infección pulmonar causada por virus |
| **Sin Neumonía** | Pulmones sin hallazgos patológicos |

Incluye **Grad-CAM** (Gradient-weighted Class Activation Mapping) para generar mapas de calor que resaltan las regiones relevantes que la red neuronal utilizó para su decisión.

---

## Features

- **Carga de imágenes** — Compatible con DICOM (`.dcm`), JPEG, PNG
- **Predicción** — Clasificación en neumonía bacteriana, viral o sano con porcentaje de confianza
- **Grad-CAM** — Visualización del mapa de calor sobre la imagen original
- **Exportación** — Guardado de resultados en CSV y generación de PDF con el reporte
- **Interfaz gráfica** — App de escritorio construida con Tkinter

## Arquitectura

El código está organizado en **6 módulos** dentro de `src/` con responsabilidades únicas:

| Módulo | Responsabilidad |
|--------|----------------|
| `detector_neumonia.py` | App Tkinter (entrada principal) |
| `read_img.py` | Lectura de imágenes DICOM/JPG/PNG |
| `preprocess_img.py` | Preprocesamiento: resize, CLAHE, normalización |
| `load_model.py` | Carga y validación del modelo CNN |
| `grad_cam.py` | Generación de mapa de calor Grad-CAM |
| `integrator.py` | Orquestador del pipeline predictivo |

---

## Arquitectura del modelo

La red neuronal convolucional está basada en el artículo *"Efficient Deep Network Architectures for Fast Chest X-Ray Tuberculosis Screening and Visualization"* (Pasa, Golkov, Pfeifer, Cremers & Pfeifer).

| Componente | Detalle |
|------------|---------|
| Bloques convolucionales | 5 bloques, cada uno con 3 convoluciones (2 secuenciales + 1 skip connection) |
| Filtros por bloque | 16, 32, 48, 64, 80 — todos 3×3 |
| Pooling | MaxPooling tras cada bloque convolucional + AveragePooling al final |
| Capas fully-connected | 3 capas Dense: 1024, 1024, 3 neuronas |
| Dropout | 20% en bloques 4 y 5, y después de la primera capa Dense |
| Regularización | Skip connections para evitar desvanecimiento del gradiente |

---

## Tech stack

| Categoría | Tecnologías |
|-----------|-------------|
| **Lenguaje** | [Python 3.10+](https://python.org) |
| **Gestor de paquetes** | [uv](https://docs.astral.sh/uv/) |
| **Deep Learning** | [TensorFlow 2.21+](https://tensorflow.org), [Matplotlib](https://matplotlib.org) |
| **Web / API** | [Streamlit 1.60+](https://streamlit.io), [FastAPI](https://fastapi.tiangolo.com), [Uvicorn](https://www.uvicorn.org) |
| **Visión artificial** | [OpenCV 5.0+](https://opencv.org), [Pillow](https://python-pillow.org) |
| **Imágenes médicas** | [PyDICOM](https://pydicom.github.io) |
| **GUI escritorio** | Tkinter, [PyAutoGUI](https://pyautogui.readthedocs.io), [tkcap](https://pypi.org/project/tkcap/), [img2pdf](https://pypi.org/project/img2pdf/) |
| **Datos** | [Pandas 2.3+](https://pandas.pydata.org) |
| **Testing** | [pytest 8+](https://docs.pytest.org), 120 tests |
| **Dev** | [watchdog](https://python-watchdog.readthedocs.io) |

---

## Makefile

| Target | Descripción |
|--------|-------------|
| `limpiar` | Limpia archivos temporales (`.o`, `.out`, `.exe`, `.log`) |
| `detector` | Limpia y lanza el detector de escritorio (Tkinter) |
| `test` | Limpia y ejecuta los tests |

```bash
make limpiar   # Limpiar temporales
make detector  # App escritorio Tkinter
make test      # Ejecutar tests
```

---

## Setup

```bash
# 1. Instalar uv (gestor de paquetes)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clonar el repositorio
git clone git@github.com:hamsomp3/neumonia_uao.git
cd neumonia_uao

# 3. Instalar dependencias
uv sync

# 4. (macOS) Si Tkinter falla, usar Python de Homebrew con Tk
uv sync --python /opt/homebrew/opt/python@3.10/bin/python3.10
```

> **Nota para macOS**: El Python incluido por `uv` no incluye soporte para Tkinter.
> `uv sync --python ...` recrea el entorno virtual usando el Python de Homebrew
> (requiere `brew install python-tk@3.10`).

---

## Uso

```bash
# App de escritorio (Tkinter)
make detector
```

---

## Tests

El proyecto incluye **115 pruebas unitarias** con `pytest`.

### Cobertura

| Categoría | Tests |
|-----------|-------|
| `preprocess()` | 20 |
| `read_dicom_file()` | 15 |
| `read_jpg_file()` | 15 |
| `predict()` | 15 |
| `grad_cam()` | 15 |
| `model_fun()` | 10 |
| Comparación DICOM vs JPG | 10 |
| Casos límite | 10 |
| Exportación CSV | 5 |

### Ejecutar tests

```bash
# Todos los tests
uv run pytest -v

# Tests de una categoría específica
uv run pytest test/test_detector.py::TestPreprocess -v

# Con coverage (opcional)
uv run pytest --cov=.
```

La interfaz Tkinter permite:
1. Ingresar la cédula del paciente
2. Cargar una imagen radiográfica (DICOM o JPG)
3. Presionar **Predecir** para obtener el diagnóstico
4. Visualizar el mapa de calor Grad-CAM
5. **Guardar** resultados en CSV
6. **PDF** para descargar el reporte

---

## Estructura del proyecto

```
neumonia_uao/
├── src/                          ← Código fuente
│   ├── detector_neumonia.py      ← App Tkinter (entrada principal)
│   ├── read_img.py               ← Lectura DICOM / JPG / PNG
│   ├── preprocess_img.py         ← Preprocesamiento (CLAHE, resize)
│   ├── load_model.py             ← Carga del modelo CNN
│   ├── grad_cam.py               ← Mapa de calor Grad-CAM
│   └── integrator.py             ← Orquestador del pipeline
├── test/                         ← Tests unitarios
│   ├── __init__.py
│   ├── conftest.py               ← Fixtures compartidos
│   └── test_detector.py          ← 115 tests
├── images/                       ← Imágenes de prueba (DICOM / JPG)
├── models/
│   └── conv_MLP_84.h5           ← Modelo CNN pre-entrenado
├── pyproject.toml                ← Configuración y dependencias
├── Makefile                      ← Automatización
├── Dockerfile                    ← Contenedor
├── LICENSE                       ← Licencia MIT
├── README.md                     ← Este archivo
└── CONSTITUTION.md               ← Contexto del proyecto
```

---

## Licencia

Este proyecto está bajo la licencia **MIT**. Ver el archivo [LICENSE](LICENSE) para más detalles.

---

## Créditos

Proyecto inspirado en [UAO-Neumonia](https://github.com/dalquinones/UAO-Neumonia) por:

- **Isabella Torres Revelo** — [github.com/isa-tr](https://github.com/isa-tr)
- **Nicolas Diaz Salazar** — [github.com/nicolasdiazsalazar](https://github.com/nicolasdiazsalazar)
