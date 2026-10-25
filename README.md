# mlops-mibici-equipo18

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**TC5044.10 Operaciones de Aprendizaje Automático (MLOps) · Equipo 18 · Fase 1**

Predicción de la **demanda horaria por estación** del sistema de bicicleta pública MiBici (Guadalajara) a partir de
sus datos abiertos, construida con prácticas de MLOps: estructura Cookiecutter Data Science, código modular,
versionado de datos y experimentos con MLflow, y pruebas automáticas.

## Problema y propuesta de valor

Las estaciones de MiBici se vacían en las horas pico de algunas zonas y se saturan en otras. Si el operador sabe
cuántas bicis van a salir de cada estación en las próximas horas, puede rebalancear **antes** de que el usuario
llegue a una estación vacía. La tarea es una regresión de conteos: salidas por estación y hora.
El análisis completo está en [`references/ml_canvas.md`](references/ml_canvas.md).

## Datos

12 meses de viajes (octubre 2025 a septiembre 2026, 4.5 millones de registros) y el catálogo de 484 estaciones,
tomados del [portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/). Columnas, hallazgos de
calidad y la justificación de cada regla de limpieza: [`references/diccionario_datos.md`](references/diccionario_datos.md).

## Cómo reproducir

```bash
python -m venv .venv && source .venv/bin/activate
make requirements      # instala dependencias y el paquete mibici en modo editable
make pipeline          # test -> download -> data (v0, v1) -> features (v2, v3) -> plots -> baseline
make test              # pruebas con datos sintéticos (no requieren descargar nada)
make mlflow-ui         # http://127.0.0.1:8080
```

Cada paso también puede correrse solo, por ejemplo `python -m mibici.dataset clean` o `python -m mibici.features`.

## Pipeline y versiones de datos en MLflow

Cada versión del dataset es un run del experimento `mibici-fase1-datos` con el dataset como input
(`mlflow.log_input`), parámetros, métricas, el **sha256** del archivo y artefactos (resumen por columna, muestra y,
en v1, el reporte de limpieza).

| Versión | Contenido | Filas | Columnas |
|---|---|---:|---:|
| v0 `viajes-v0` | 12 CSV mensuales concatenados, sin cambios | 4,497,877 | 8 |
| v1 `viajes-v1` | Tipos corregidos y 8 reglas de limpieza (−4.1 %) | 4,311,657 | 9 |
| v2 `demanda-v2` | Panel estación-hora (horas de servicio), calendario, festivos, rezagos | 2,870,312 | 19 |
| v3 `demanda-v3` | One-hot de municipio, 15 componentes PCA, escalado con train, partición temporal | 2,804,652 | 35 |

Partición temporal de v3: entrenamiento oct 2025 a jun 2026 (1,973,154 filas), validación jul y ago 2026
(552,001) y prueba sep 2026 (279,497).

## Línea base (experimento `mibici-fase1-baseline`)

Se evalúan dos horizontes en validación (jul y ago 2026). En 1 h el modelo puede usar la demanda de la hora
anterior; en 24 h solo usa rezagos de 24 h o más, como un pronóstico hecho la noche anterior.

| Horizonte | Modelo | MAE | RMSE |
|---|---|---:|---:|
| 1 h | Persistencia (hora anterior) | 1.127 | 1.981 |
| 1 h | HistGradientBoosting, pérdida Poisson | **0.849** | **1.400** |
| 24 h | Persistencia (mismo momento de ayer) | 1.093 | 2.050 |
| 24 h | Persistencia semanal (misma hora, semana anterior) | 1.014 | 1.832 |
| 24 h | HistGradientBoosting, pérdida Poisson | **0.875** | **1.401** |

El modelo simple reduce el MAE 24.7 % en 1 h y 13.7 % en 24 h frente a la mejor persistencia de cada horizonte.
Es el punto de partida de la Fase 2, no el modelo final.

## Estructura del proyecto

```
├── Makefile             <- test, download, data, features, plots, baseline, pipeline, lint, mlflow-ui
├── references/          <- ML Canvas y diccionario de datos con las reglas de limpieza
├── reports/
│   ├── figures/         <- Figuras del EDA (también en MLflow, run eda-figuras)
│   └── reporte_limpieza.json   <- Viajes eliminados por regla (también en MLflow)
├── tests/test_data.py   <- Pruebas del pipeline
├── data/                <- raw, interim, processed (no se versiona en git; se regenera con make)
├── models/              <- Scaler y PCA ajustados (también se registran en MLflow)
└── mibici/              <- Código fuente
    ├── config.py        <- Rutas, URLs, umbrales de limpieza, partición temporal, MLflow
    ├── dataset.py       <- Descarga, lectura, TripCleaner y validación (v0, v1)
    ├── features.py      <- Panel de demanda, variables, PCA, codificación y escalado (v2, v3)
    ├── tracking.py      <- log_dataset_version(): versionado de datasets en MLflow
    ├── plots.py         <- Figuras del EDA
    └── modeling/train.py <- Líneas base sobre v3 (experimento mibici-fase1-baseline)
```

## Equipo

| Integrante | Rol |
|---|---|
| Alonso Rojas Loreto (A01706755) | Data Engineer |
| Daniel Fernando Romero Villacis | Data Scientist |
| José Luis Israel Servín Gómez | ML Engineer |
| César Guillermo Vázquez Álvarez | Software Engineer / DevOps |

## Uso de inteligencia artificial

Anthropic. (2026). *Claude Opus 5.5* [Modelo de lenguaje grande]. Se utilizó para generar y depurar código, y para
redactar la documentación. El equipo revisó y validó el código, las cifras y las decisiones metodológicas.
