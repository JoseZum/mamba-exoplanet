# Bitácora de investigación

Registro cronológico de decisiones, descubrimientos y cambios del proyecto.
Complementa el historial de git: git captura qué cambió en el código, esta bitácora captura por qué.

---

## 2026-04-29 | Fase 0: Setup del repositorio

### Lo que se hizo

- Estructura inicial del repositorio creada: `src/exoplanet/`, `data/`, `notebooks/`, `scripts/`, `configs/`, `experiments/`, `paper/`.
- `pyproject.toml` con todas las dependencias del proyecto (PEP 621).
- `.gitignore` configurado para excluir `data/raw/`, `data/processed/`, `experiments/`, archivos `.fits`, `.pt` y `.venv`.
- `README.md` inicial con objetivo, estructura y roadmap.
- Entorno virtual creado en `.venv/` con Python 3.11.9.
- Paquete instalado en modo editable con `pip install -e ".[dev]"`.

### Decisiones tomadas

**Entorno Mamba (Fases 8-9): WSL2 con Ubuntu 24.04.**
`mamba-ssm` requiere compilar extensiones CUDA con `nvcc` y no tiene wheels pre-construidos para Windows nativo. Intentar resolverlo en Fase 8 implicaría perder tiempo cerca del deadline. La decision se tomó en Fase 0 para evitar ese bloqueo. Fases 0-7 (exploración, preprocesamiento, CNN) corren en Windows normalmente.

**PyTorch con CUDA 12.8.**
El driver instalado (581.83) soporta hasta CUDA 13.0, compatible con la rueda `cu128`. Se reinstala `torch` con `--index-url https://download.pytorch.org/whl/cu128` para habilitar la RTX 3050. La build CPU que instala `pip install torch` por defecto no usa GPU.

**Dependencias adicionales añadidas a `pyproject.toml`:**
- `imbalanced-learn`: SMOTE para balanceo de clases (Fase 5+)
- `astroquery`: acceso a MAST para descarga de curvas de luz (Fase 2)
- `pyvo`: consultas TAP al NASA Exoplanet Archive (Fase 1)
- `einops`: operaciones de tensor para implementación de Mamba (Fase 8)
- `tensorboard`: logs de entrenamiento (Fase 6)
- `seaborn`: visualizaciones de métricas y comparaciones (Fases 7/9/10)

### Estado al cierre de esta sesión

- `torch 2.11.0+cu128` instalado, CUDA activo, RTX 3050 detectada.
- Smoke tests pasando (2/2).

---

## 2026-04-30 | Fase 1: Exploración del TOI Catalog

### Lo que se hizo

- Script `scripts/get_data.py` que descarga el TOI Catalog completo desde el NASA Exoplanet Archive via TAP y guarda `data/raw/toi_catalog.csv` (gitignored) y `data/splits/toi_summary.csv` (versionado).
- Notebook `notebooks/01_toi_eda.ipynb` con análisis de las 6 variables clave del catálogo.
- Archivo `data/splits/tics_labeled.csv` generado: solo CP y FP, con label binario (1/0).

### Nombre del script

El script inicialmente se llamó `download_toi_catalog.py`. Se renombró a `get_data.py` por ser el nombre estándar en repos de ML y porque en Fase 2 el mismo script puede extenderse con flags (`--catalog`, `--fits`) sin cambiar el nombre.

### Descubrimientos del análisis

**Conteos reales (al 2026-04-30):**
Los números de la propuesta académica estaban desactualizados. Los valores reales del catálogo son:

| Disposición | Propuesta | Real |
|---|---|---|
| CP (Confirmed Planet) | ~638 | 726 |
| FP (False Positive) | ~1,400 | 1,242 |
| PC (Planet Candidate) | ~6,600 | 4,788 |
| KP (Known Planet) | no estimado | 591 |
| APC (Ambiguous PC) | no estimado | 481 |

Total etiquetado para entrenamiento (CP + FP): 1,968. Ratio real CP:FP = 1:1.71, no 1:2.2 como se asumía. Esto cambia el peso de la `weighted cross-entropy`.

**Magnitud TESS (st_tmag):**
Solo 2 CP y 16 FP tienen tmag > 15 (umbral donde el ruido vuelve las curvas poco confiables). No vale la pena filtrar por magnitud: el 99%+ del dataset está en rango útil.

**Periodo orbital (pl_orbper):**
El 8.8% de los CP (64 estrellas) tienen periodo orbital > 27 días. Algunos llegan hasta 1,134 días. TESS observa por sectores de ~27 días, por lo que estas estrellas probablemente no tienen tránsito completo visible en una sola curva. Decision pendiente para Fase 2: descartar estas 64 estrellas o mantenerlas y dejar que el preprocesamiento las maneje.

Solo el 2.7% de los FP tienen periodo > 27 días, lo que confirma que los FP son señales de corto período (binarias eclipsantes, artefactos).

**Profundidad y duración del tránsito:**
Las distribuciones de CP y FP se solapan casi completamente. Medianas: CP 1,964 ppm vs FP 2,870 ppm en profundidad; CP 2.56 h (77 puntos) vs FP 2.59 h (78 puntos) en duración. No existe un umbral simple que separe las clases. Esto confirma que el problema justifica deep learning.

Un tránsito promedio ocupa aproximadamente 77-78 puntos en una secuencia de 18,000. La señal es real pero pequeña en contexto.

**Columna `sectors` completamente vacía:**
La columna `sectors` del TOI Catalog tiene 100% de valores NaN. El catálogo no incluye en qué sectores fue observada cada estrella. En Fase 2 hay que consultar MAST directamente por TIC ID para obtener esa información. Se eliminó la columna `n_sectors` de `tics_labeled.csv` porque era siempre 0.

**NaNs en variables clave:**
Solo 17 NaNs en `pl_orbper` (0.9%). El resto de las variables clave están completas. No hay problema de NaNs en el catálogo.

### Cambios respecto a la propuesta original

- Conteos actualizados en `README.md` con los valores reales.
- Columna `sectors` descartada como fuente de información para Fase 2.
- Ninguna variable del TOI Catalog entra al modelo como feature. El catálogo solo sirve para seleccionar qué estrellas descargar y con qué label. Esto se documentó explícitamente en el README para evitar confusiones futuras.

### Decisiones pendientes que surgieron

- ¿Descartar los 64 CP con periodo > 27 días antes de Fase 2 o mantenerlos? Decisión antes de Fase 2.
- ¿Sumar KP (591 Known Planets) a la clase positiva para compensar el dataset pequeño? Decisión antes de Fase 4.

### Estado al cierre de esta sesión

- `data/splits/tics_labeled.csv`: 1,968 filas, columnas `tid / tfopwg_disp / st_tmag / pl_orbper / label`.
- Fase 1 completa.
- Listo para arrancar Fase 2 (descarga de curvas de luz desde MAST).
