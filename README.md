# Mamba-Exoplanet

> Selective State Space Models para detección de exoplanetas en curvas de luz de TESS — un estudio comparativo frente a un baseline CNN 1D.

**Proyecto académico** — Inteligencia Artificial, Instituto Tecnológico de Costa Rica, Semestre I 2026.
**Autores:** José Fabián Zumbado Ruiz, Jeremmy Aguilar Villanueva.
**Profesor:** Kenneth Obando Rodríguez.

## Objetivo

Evaluar si una arquitectura basada en **Mamba** (Gu & Dao, 2023) puede igualar o superar el rendimiento de clasificadores CNN 1D del estado del arte (familia AstroNet / ExoMiner) en la tarea binaria de distinguir **Confirmed Planets (CP)** de **False Positives (FP)** en curvas de luz de TESS a cadencia de 2 minutos, operando directamente sobre la señal cruda `PDCSAP_FLUX`.

Métricas objetivo: AUC-ROC ≥ 0.93, F1 (clase planeta) ≥ 0.85, mejora Mamba sobre CNN ≥ +3 p.p. de AUC, latencia de inferencia < 100 ms por estrella.

## Estructura del repositorio

```
mamba-exoplanet/
├── configs/                # YAMLs por experimento (un archivo = un run reproducible)
├── data/
│   ├── raw/                # .fits descargados de MAST       (gitignored)
│   ├── processed/          # tensores listos para entrenar    (gitignored)
│   └── splits/             # TIC IDs de train/val/test        (versionado)
├── src/exoplanet/          # código fuente como paquete instalable
│   ├── data/               # descarga, preprocesamiento, Dataset, augment
│   ├── models/             # cnn_baseline.py, mamba.py
│   ├── training/           # loop, losses, schedulers
│   ├── evaluation/         # métricas, gráficos
│   └── utils/              # seeds, logging, paths
├── notebooks/              # exploración numerada (01_..., 02_..., 03_...)
├── scripts/                # CLIs reproducibles (download_data, train, evaluate)
├── experiments/            # outputs de cada run               (gitignored)
├── tests/                  # pytest
└── paper/                  # LaTeX, figuras y tablas finales
```

## Instalación

**Requisitos previos:**

- Python **3.10 u 3.11** (probado con 3.11.9). Se recomienda la build oficial de [python.org](https://www.python.org/downloads/) sobre la versión de Microsoft Store, que a veces tiene problemas de permisos en `pip install -e`.
- Git Bash o PowerShell en Windows; bash en Linux/macOS.
- Aproximadamente **2.5 GB libres** en disco para el entorno completo (incluyendo PyTorch con CUDA).

> **Nota sobre OneDrive:** si el repositorio queda dentro de una carpeta sincronizada por OneDrive, mové el repo a una ruta local (p. ej. `C:\dev\mamba-exoplanet\`) **antes** de crear el `.venv`. OneDrive intenta sincronizar miles de archivos del entorno virtual y puede corromper binarios de PyTorch.

### Paso 1 — Clonar y posicionarse

```bash
git clone <url-del-repo> mamba-exoplanet
cd mamba-exoplanet
```

### Paso 2 — Crear y activar el entorno virtual

```bash
python -m venv .venv

# Activar — Git Bash en Windows:
source .venv/Scripts/activate
# Activar — PowerShell:
# .venv\Scripts\Activate.ps1
# Activar — Linux / macOS:
# source .venv/bin/activate
```

### Paso 3 — Instalar el paquete en modo editable

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

Esto instala el paquete `exoplanet` y todas las dependencias declaradas en `pyproject.toml`, incluyendo `torch` (build **CPU** por defecto), `lightkurve`, `astropy`, `jupyterlab`, `pytest` y `ruff`.

### Paso 4 — Reinstalar PyTorch con CUDA (necesario para fases 5–9)

La build CPU de `torch` no usa la GPU. Para entrenar Mamba en la RTX 3050 hay que reemplazarla por la rueda CUDA. **Verificá primero la versión de CUDA de tu driver:**

```bash
nvidia-smi    # mirá "CUDA Version: XX.Y" en la esquina superior derecha
```

Luego desinstalá la build CPU e instalá la build que corresponde. Para CUDA 12.1 (la más común en drivers recientes de la RTX 3050):

```bash
pip uninstall -y torch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

Para otras versiones de CUDA, consultá <https://pytorch.org/get-started/locally/> y copiá el comando correspondiente.

Verificación de CUDA:

```bash
python -c "import torch; print('CUDA OK' if torch.cuda.is_available() else 'CPU only', '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

### Paso 5 — (Fase 8 solamente) Instalar mamba-ssm

> **Advertencia:** `mamba-ssm` requiere compilar extensiones CUDA y **no tiene wheels pre-construidos para Windows nativo**. Si el paso falla, hay tres alternativas (decidir en Fase 8):
> - (a) Usar WSL2 con Ubuntu y repetir el setup allí.
> - (b) Implementar Mamba en PyTorch puro (más lento, portable).
> - (c) Entrenar en Google Colab con GPU A100/T4.

```bash
# Solo intentar si tenés nvcc disponible (CUDA Toolkit completo instalado):
pip install causal-conv1d mamba-ssm
```

### Paso 6 — Verificación final

```bash
pytest -q                                                      # smoke tests deben pasar
python -c "import exoplanet; print(exoplanet.__version__)"     # → 0.1.0
python -c "import torch; print('CUDA OK' if torch.cuda.is_available() else 'CPU only', '|', torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
python -c "import seaborn, tensorboard, einops, imbalanced_learn; print('deps extra OK')"
```

## Reproducir resultados (cuando estén disponibles)

```bash
python scripts/download_data.py --catalog data/splits/tics.csv
python scripts/train.py --config configs/cnn_baseline.yaml
python scripts/train.py --config configs/mamba_small.yaml
python scripts/evaluate.py --run experiments/<run_id>
```

## Hardware de referencia

| Componente | Especificación |
|---|---|
| GPU | NVIDIA RTX 3050 (4 GB VRAM) — cuello de botella |
| CPU | Intel Core i5-12450H (8 cores, 12 threads) |
| RAM | 40 GB |

Las restricciones de VRAM motivan el uso de mixed precision (FP16), `batch_size = 16` y gradient checkpointing en Mamba.

## Roadmap

- [x] **Fase 0** — Setup del repositorio
- [ ] **Fase 1** — Exploración del TOI Catalog
- [ ] **Fase 2** — Pipeline de descarga (MAST + lightkurve)
- [ ] **Fase 3** — Preprocesamiento (normalización, NaN, longitud fija)
- [ ] **Fase 4** — Splits por TIC ID + `Dataset` PyTorch
- [ ] **Fase 5** — CNN 1D baseline
- [ ] **Fase 6** — Training loop (logs, seeds, checkpoints)
- [ ] **Fase 7** — Evaluación (métricas + curvas ROC/PR)
- [ ] **Fase 8** — Modelo Mamba
- [ ] **Fase 9** — Comparación rigurosa CNN vs Mamba
- [ ] **Fase 10** — Paper (figuras y tablas finales)

## Cita

```bibtex
@misc{zumbado_aguilar_2026,
    title       = {Mamba State Space Models for Exoplanet Detection in TESS Light Curves},
    author      = {Zumbado Ruiz, Jos\'e Fabi\'an and Aguilar Villanueva, Jeremmy},
    year        = {2026},
    institution = {Instituto Tecnol\'ogico de Costa Rica}
}
```

## Licencia

MIT. Ver `LICENSE` (pendiente).
