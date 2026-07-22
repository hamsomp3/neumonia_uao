# Neumonía UAO — Constitución del Proyecto

> Contexto del proyecto para sesiones con asistentes LLM.
> Trackeado al repositorio.

---

## Flujo Diario (commit + push)

```bash
# 1. Ver cambios
git status

# 2. Verificar que la app Tkinter arranca sin errores
make detector
# (cerrar la ventana manualmente)

# 3. Ejecutar tests
uv run pytest -v

# 4. Agregar y commitear
git add -A
git commit -m "tipo: mensaje descriptivo"

# 5. Subir a GitHub
git push
```

**Convención de commits:** Usar [Conventional Commits](https://www.conventionalcommits.org/):
- `feat:` — nueva funcionalidad
- `fix:` — corrección de bug
- `docs:` — documentación
- `refactor:` — refactorización
- `style:` — formato, linting
- `chore:` — tareas de mantenimiento

---

## Stack Técnico del Proyecto

| Categoría | Tecnologías |
|-----------|-------------|
| **Lenguaje** | Python 3.13+ |
| **Gestor de paquetes** | [uv](https://docs.astral.sh/uv/) |
| **Deep Learning** | TensorFlow 2.21+, Matplotlib |
| **Visión artificial** | OpenCV 5.0+, Pillow |
| **Imágenes médicas** | PyDICOM |
| **GUI escritorio** | CustomTkinter, PyAutoGUI |
| **Datos** | Pandas 2.3+ |
| **Testing** | pytest 8+ (117 tests) |
| **Linting** | Ruff |
| **Pre-commit** | pre-commit |
| **Dev** | watchdog |

---

## Historial de Decisiones Técnicas

### Repositorio
- **Nombre:** `neumonia_uao`
- **Cuenta GitHub:** `hamsomp3`
- **URL:** `git@github.com:hamsomp3/neumonia_uao.git`
- **Rama principal:** `main`

### Modelo CNN
- Arquitectura: 5 bloques convolucionales con skip connections (WilhemNet86)
- Pesos pre-entrenados en `models/conv_MLP_84.h5`
- Clasificación: bacteriana, viral, sin neumonía
- Explicabilidad: Grad-CAM

### Migración a uv
- Se migró de pip/conda a uv como gestor de paquetes
- Las dependencias se declaran en `pyproject.toml`
- Entorno virtual en `.venv/`

### Tkinter en macOS
- El Python de uv no incluye Tkinter
- Solución: `uv sync --python /opt/homebrew/opt/python@3.13/bin/python3.13`
- Requiere `brew install python-tk@3.13`

### Refactorización a src/ layout
- Todo el código fuente se movió a `src/`
- Los módulos se importan entre sí con imports absolutos (no relativos)
- `pytest` resuelve las rutas via `pythonpath = ["src"]` en `pyproject.toml`
- Archivos `main.py` y `prueba.py` eliminados (sobrantes del template)
- Empaquetado: `packages = ["src"]` en hatch

### Modularización del detector
- `detector_neumonia.py` quedó como fachada backward-compatible que re-exporta funciones
- 5 módulos con responsabilidad única: `read_img`, `preprocess_img`, `load_model`, `grad_cam`, `integrator`
- Cada módulo tiene type hints, docstrings PEP 257, y manejo de excepciones explícito
- Las 115 pruebas unitarias se actualizaron con las rutas de patch correctas

### Refactorización MVC (CustomTkinter)
- `detector_neumonia.py` se simplificó a facade puro (re-exports + `main()` delegando en `gui.controller.main()`)
- Se creó `src/gui/` package con arquitectura MVC:
  - `theme.py` — paleta de colores médicos, badges diagnóstico (rojo/naranja/verde), `configure_ctk()`
  - `worker.py` — `AsyncWorker` con `ThreadPoolExecutor` + `root.after()` para tareas en background
  - `views.py` — componentes CTk reutilizables (`HeaderFrame`, `ImagePreviewFrame`, `HeatmapPreviewFrame`, `ResultsFrame`, `ControlFrame`, `StatusBar`, `PdfDialog`)
  - `controller.py` — `AppController` que orquesta vistas, worker y callbacks de eventos
- `customtkinter` reemplaza a Tkinter estándar y `tkcap` (PDF ahora usa Pillow directamente)
- Theme toggle (light/dark) con botón en el header
- 117 tests unitarios (se agregaron 2 por el manejo de modos oscuro/claro)

### compile=False para compatibilidad con Keras ≥3
- El modelo `.h5` original usaba `reduction="auto"` en la loss function, incompatible con Keras 3
- Solución: `tf.keras.models.load_model(MODEL_PATH, compile=False)`
- El modelo funciona igual para inferencia y Grad-CAM (no requiere `compile`)
- `test_predict_end_to_end` verifica el pipeline real localmente (se salta si no hay `.h5`)

---

## Configuración Local (no trackear)

> Configuración específica de esta máquina.
> **NO incluir en commits.**
> Copiar a nuevos proyectos personales cuando sea necesario.

### Identidad Git Automática

`~/.gitconfig` (global):
```
[includeIf "gitdir:~/Desktop/Repositorios/personal/"]
    path = ~/.gitconfig-personal
```

`~/.gitconfig-personal`:
```
[user]
    email = hamsomp3@gmail.com
    name = Jan Polanco V.

[core]
    sshCommand = ssh -i ~/.ssh/id_ed25519
```

**Efecto:** Todo proyecto dentro de `~/Desktop/Repositorios/personal/` usa automáticamente:
- **Email:** `hamsomp3@gmail.com`
- **Nombre:** Jan Polanco V.
- **Llave SSH:** `~/.ssh/id_ed25519`

Sin necesidad de `gh`, variables de entorno ni flags manuales.

### Remote actual

```
origin  git@github.com:hamsomp3/neumonia_uao.git (fetch)
origin  git@github.com:hamsomp3/neumonia_uao.git (push)
```
