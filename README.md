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

> El proyecto vive dentro de OneDrive. Para evitar que OneDrive sincronice el entorno virtual y se corrompa, **creá `.venv` con un nombre que arranca con `.`** (ya está ignorado por OneDrive y por Git) o, mejor aún, montá el `.venv` fuera de OneDrive y apuntá tu IDE allí.

```bash
# 1. Crear entorno virtual (Python ≥ 3.10)
python -m venv .venv

# 2. Activarlo
source .venv/Scripts/activate    # Git Bash en Windows
# .venv\Scripts\activate          # PowerShell

# 3. Instalar el paquete en modo editable (con extras de desarrollo)
pip install --upgrade pip
pip install -e ".[dev]"

# 4. Verificación rápida
pytest -q
python -c "import exoplanet; print(exoplanet.__version__)"
```

> **PyTorch + CUDA:** la dependencia `torch` se instala en su build CPU por defecto. Para usar la RTX 3050, reinstalá la rueda con CUDA: visitá <https://pytorch.org/get-started/locally/> y copiá el comando que corresponde a tu versión de CUDA.

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
