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

---

## 2026-05-04 | Fase 2: Pipeline de descarga MAST (script y piloto)

### Lo que se hizo

- Script `scripts/download_lightcurves.py` que lee `data/splits/tics_labeled.csv`, consulta MAST por cada TIC ID via `lightkurve`, descarga los archivos `_lc.fits` de cadencia 2 min (autor SPOC) y mantiene un manifest CSV con el resultado por TIC.
- Manifest `data/splits/manifest.csv` con columnas `tid / label / n_sectors_found / n_sectors_downloaded / sectors / total_size_mb / status / error / duration_s / downloaded_at`. Versionado.
- Piloto de 5 estrellas (orden mezclado con seed=42).

### Decisiones tomadas

**SPOC 2-min como única cadencia.**
La búsqueda usa `author="SPOC"` y `exptime=120`. Es la pipeline oficial de NASA que produce `PDCSAP_FLUX`, la misma señal que usan AstroNet y ExoMiner. TOIs que solo tienen FFI 30-min o QLP se marcan como `no_data` y se descartan del dataset. Mantenerlas implicaría manejar dos pipelines de preprocesamiento distintos, lo cual no aporta a Tier 1.

**El script descarga FITS, NO extrae PDCSAP_FLUX.**
La extracción de la serie PDCSAP_FLUX y el preprocesamiento (normalización, NaN handling, longitud fija) ocurren en Fase 3. Esta separación permite reanudar el pipeline desde cualquier punto sin re-descargar.

**Idempotencia con estados terminales.**
Estados `ok` y `no_data` son terminales: el script los saltea en corridas sucesivas. Estados `error` y `download_failed` se reintentan por defecto (típicamente fallos transitorios de MAST). Flag opcional `--no-retry-failed` para deshabilitar reintentos. Esta distinción evita el bug de "marcar como hecho cualquier cosa que esté en el manifest", que perdería data de TICs con fallos temporales.

**Reintento limpio sin duplicados.**
Cuando un TIC se reintenta, su fila vieja se borra del manifest antes de añadir la nueva. Garantiza una fila por TIC.

**Manifest se escribe cada 10 TICs, no al final.**
Una corrida de ~1,968 TICs toma horas. Si se corta, perdemos a lo sumo 10 descargas, no todo el progreso.

**Cap `--max-sectors 3` recomendado para descarga completa.**
El piloto reveló que un TIC tenía 32 sectores (63 MB). Sin cap, el dataset puede pasar de 30 GB. Con cap=3, ~9 GB. Se documenta en el script que capar sectores sesga el dataset (siempre se toman los primeros) y que para entrenamiento del paper hay que decidir explícitamente la política de selección. Para Tier 1 con longitud fija L=18,000 (un sector), 3 es suficiente y deja margen para elegir el mejor sector después.

**Mantener los 1,968 TOIs durante la descarga (incluye los 64 CP con período > 27 días).**
Descartarlos en Fase 2 perdería data irrecuperable. La decisión de filtrarlos o no se traslada a Fase 3 (preprocesamiento), donde se decide cómo manejar tránsitos parciales o ausentes. La descarga no cuesta nada extra por mantenerlos.

**`downloaded_at` en formato ISO-8601 UTC.**
Para reproducibilidad: deja constancia exacta de cuándo se obtuvo cada FITS desde MAST (los datos pueden actualizarse).

### Descubrimientos del piloto

**1 de 5 TICs sin SPOC 2-min (TIC 354400186).**
Era candidato del catálogo TOI pero MAST no devuelve curvas SPOC de cadencia 120s para esa estrella. Confirmamos que la pérdida del 10-20% del dataset por este motivo es esperable. El conteo final etiquetado se confirmará al terminar la descarga completa.

**Outlier de 32 sectores (TIC 272086159).**
Una sola estrella aportó 63 MB y tomó 60 s. Justifica el cap. Estrellas en la zona de visión continua de TESS pueden tener decenas de sectores acumulados.

**Patrón de archivos confirmado.**
Los FITS se guardan como `mastDownload/TESS/tess<fecha>-s<sector>-<tid:016d>-<scid>-s/tess<fecha>-s<sector>-<tid:016d>-<scid>-s_lc.fits`. El TIC aparece padded a 16 dígitos. El cálculo de `total_size_mb` con el patrón `**/*{tid:016d}*_lc.fits` funciona correctamente.

### Estado al cierre de esta sesión

- `scripts/download_lightcurves.py` listo y validado con piloto de 5 TICs (4 ok, 1 no_data, ~76 MB).
- `data/splits/manifest.csv` con 5 filas.
- `data/raw/lightcurves/mastDownload/TESS/...` con 39 archivos FITS (~76 MB).
- Descarga completa (1,963 TICs restantes) **NO ejecutada todavía**: queda como tarea para correr en background del usuario. ETA estimado 3-4 horas, ~9 GB de disco.
