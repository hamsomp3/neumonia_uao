# Neumonía UAO — Constitución del Proyecto

> Contexto del proyecto para sesiones con asistentes LLM.
> Trackeado al repositorio.

---

## Flujo Diario (commit + push)

```bash
# 1. Ver cambios
git status

# 2. Agregar y commitear
git add -A
git commit -m "tipo: mensaje descriptivo"

# 3. Subir a GitHub
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
| **Lenguaje** | Python 3.10+ |
| **Gestor de paquetes** | [uv](https://docs.astral.sh/uv/) |
| **Deep Learning** | TensorFlow 2.21+, Matplotlib |
| **Visión artificial** | OpenCV 5.0+, Pillow |
| **Imágenes médicas** | PyDICOM |
| **GUI escritorio** | Tkinter, PyAutoGUI, tkcap, img2pdf |
| **Datos** | Pandas 2.3+ |
| **Testing** | pytest 8+ (115 tests) |
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
- Solución: `uv sync --python /opt/homebrew/opt/python@3.10/bin/python3.10`
- Requiere `brew install python-tk@3.10`

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
