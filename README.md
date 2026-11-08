# mlops-mibici-equipo18

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**TC5044.10 Operaciones de Aprendizaje Automático (MLOps) · Equipo 18 · Fase 2**

Predicción de la **demanda horaria por estación** del sistema de bicicleta pública MiBici (Guadalajara) a partir de
sus datos abiertos, construida con prácticas de MLOps: estructura Cookiecutter Data Science, código modular,
un `Pipeline` de scikit-learn, ajuste de hiperparámetros con Optuna, seguimiento de experimentos y Model Registry en
MLflow, pruebas automáticas y CI.

## Problema y propuesta de valor

Las estaciones de MiBici se vacían en las horas pico de algunas zonas y se saturan en otras. Si el operador sabe
cuántas bicis van a salir de cada estación al día siguiente, puede rebalancear **antes** de que el usuario llegue a
una estación vacía. La tarea es una regresión de conteos: salidas por estación y hora, pronosticadas la noche
anterior (horizonte de 24 h). El análisis completo está en [`references/ml_canvas.md`](references/ml_canvas.md).

## Datos

12 meses de viajes (octubre 2025 a septiembre 2026, 4.5 millones de registros) y el catálogo de 484 estaciones,
tomados del [portal de datos abiertos de MiBici](https://www.mibici.net/es/datos-abiertos/). El portal retiró el
catálogo de septiembre, así que la copia usada se versiona en `data/external/`. Columnas, hallazgos de calidad y la
justificación de cada regla de limpieza: [`references/diccionario_datos.md`](references/diccionario_datos.md).

## Cómo reproducir

```bash
python -m venv .venv && source .venv/bin/activate
make requirements      # instala dependencias y el paquete mibici en modo editable
make pipeline          # test -> download -> data -> features -> plots -> tune -> predict
make mlflow-ui         # http://127.0.0.1:8080
```

| Paso | Comando | Qué hace |
|---|---|---|
| Pruebas | `make test` | 17 pruebas con datos sintéticos (no descargan nada); también corren en el CI |
| Descarga | `make download` | 12 CSV mensuales (urllib y, si el portal lo rechaza, curl) |
| Limpieza | `make data` | v0 crudo y v1 limpio, registrados en MLflow |
| Variables | `make features` | v2 panel estación-hora y v4 entrada del modelo con partición temporal |
| Figuras | `make plots` | Figuras del EDA en `reports/figures` y en MLflow |
| Modelos | `make tune` | Optuna para 3 familias; cada Pipeline final va al Model Registry |
| Pronóstico | `make predict` | Septiembre 2026 con el modelo `@champion` |

Con 2 núcleos, `make tune` tarda alrededor de 45 minutos; cada familia puede correrse por separado
(`python -m mibici.modeling.tune --modelos random_forest`) y la tabla comparativa se arma con lo que haya en MLflow.
Semilla fija (`SEED = 42`), rutas, horizonte e hiperparámetros de la búsqueda viven en `mibici/config.py`.

## Versiones de los datos en MLflow (experimento `mibici-fase1-datos`)

Cada versión es un run con el dataset como input (`mlflow.log_input`), parámetros, métricas, el **sha256** del archivo
y artefactos (resumen por columna, muestra y, en v1, el reporte de limpieza).

| Versión | Contenido | Filas |
|---|---|---:|
| v0 `viajes-v0` | 12 CSV mensuales concatenados, sin cambios | 4,497,877 |
| v1 `viajes-v1` | Tipos corregidos y 8 reglas de limpieza (−4.1 %) | 4,311,657 |
| v2 `demanda-v2` | Panel estación-hora (horas de servicio), calendario, festivos, rezagos | 2,870,312 |
| v3 `demanda-v3` | Fase 1: datos ya escalados, codificados y con PCA (se conserva en MLflow) | 2,804,652 |
| **v4 `demanda-v4`** | **Fase 2:** v2 con rezagos completos y partición temporal, sin transformar | 2,804,652 |

Partición temporal: entrenamiento oct 2025 a jun 2026 (1,973,154 filas), validación jul y ago 2026 (552,001) y
prueba sep 2026 (279,497).

## Pipeline de scikit-learn (`mibici/modeling/pipeline.py`)

En la Fase 1 el escalado y el PCA se aplicaban al dataset (v3). En la Fase 2 viven dentro de un `Pipeline` que se
ajusta solo con entrenamiento y se guarda completo en el Model Registry, así que la inferencia aplica exactamente las
mismas transformaciones que el entrenamiento.

```
Pipeline
├── preprocesamiento: ColumnTransformer
│   ├── ciclicas         seno y coseno de hora_dia y dia_semana
│   ├── numericas        StandardScaler: capacidad, lat, lon, lag_24h, lag_168h
│   ├── municipio        OneHotEncoder (una categoría nueva no rompe el modelo)
│   ├── perfil_estacion  StationProfilePCA: perfil horario de cada estación -> PCA al 80 % de varianza
│   └── binarias         fin_de_semana y festivo, sin cambios
└── modelo: PoissonRegressor | RandomForestRegressor | HistGradientBoostingRegressor(loss="poisson")
```

`StationProfilePCA` es un transformador propio (`fit`/`transform`): calcula la demanda media de cada estación por hora
del día y tipo de día con los datos de entrenamiento, la resume con PCA y asigna el perfil promedio (ceros) a una
estación que no vio al entrenar. El horizonte de 24 h excluye `lag_1h` y `media_24h`, que no se conocen la noche
anterior.

## Modelos, ajuste y evaluación (experimento `mibici-fase2-modelos`)

`make tune` compara tres familias para el horizonte de 24 h. Optuna (TPE, semilla 42) hace 25 pruebas por familia
con una muestra de 400,000 filas de entrenamiento y minimiza la **devianza de Poisson** en validación; cada prueba es
un run anidado en MLflow. El mejor juego se reentrena con todo el entrenamiento y se evalúa en validación y una sola
vez en prueba (sep 2026).

| Modelo | Versión en el registro | MAE validación | MAE prueba | MAE prueba, top 20 % |
|---|---:|---:|---:|---:|
| Persistencia semanal | — | 1.014 | 1.169 | 2.375 |
| Regresión de Poisson | 1 | 1.207 | 1.342 | 2.540 |
| Bosque aleatorio | 3 | 0.902 | 1.042 | 2.005 |
| **HistGradientBoosting Poisson** | 2 `@champion` | **0.865** | **1.005** | **1.896** |

HistGradientBoosting Poisson baja el MAE 14.6 % en validación y 14.0 % en prueba frente a la persistencia semanal (20.2 % en las estaciones de mayor demanda), así que cumple la meta de −10 % del ML Canvas. El sesgo de prueba (−0.228) indica que septiembre, el mes de mayor demanda, se subestima.

Justificación de los algoritmos, rangos de búsqueda y conclusiones: [`reports/modelos_fase2.md`](reports/modelos_fase2.md).
Análisis del campeón (pronóstico de una semana, error por hora y por estación):
[`notebooks/2.0-equipo18-analisis-del-modelo.ipynb`](notebooks/2.0-equipo18-analisis-del-modelo.ipynb).

## Model Registry

Cada familia queda como una versión del modelo registrado **`mibici-demanda-estacion-hora`**, con el Pipeline completo,
su firma de entrada, una descripción y etiquetas con los hiperparámetros (`param_*`) y las métricas de validación y
prueba (`mae_valid`, `devianza_poisson_valid`, `mae_test`, `sesgo_test`, …). La versión con menor devianza de Poisson
en validación recibe el alias **`champion`**, que es lo que usa la inferencia:

```python
import mlflow
mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/mibici-demanda-estacion-hora@champion")
model.predict(df[model.feature_names_in_])   # df con las columnas de v4
```

`python -m mibici.modeling.train --modelo hgb_poisson` reentrena una familia con los mejores hiperparámetros de la
última búsqueda (leídos de MLflow) y registra una versión nueva; el alias se recalcula.

## Estructura del proyecto

```
├── Makefile             <- requirements, test, lint, download, data, features, plots, tune, predict, pipeline
├── .github/workflows/   <- CI: ruff y pytest en cada push y pull request
├── data/external/       <- Catálogo de estaciones versionado (raw, interim y processed se regeneran con make)
├── notebooks/           <- Solo exploración; importan el paquete mibici
├── references/          <- ML Canvas, diccionario de datos y verificación de reproducibilidad
├── reports/             <- Hallazgos del EDA, selección de modelos, tabla comparativa y figuras
├── tests/               <- test_data.py, test_pipeline.py, test_tune.py (datos sintéticos)
└── mibici/              <- Código fuente
    ├── config.py        <- Rutas, fuente, umbrales de limpieza, partición, semilla, horizonte, MLflow
    ├── dataset.py       <- Descarga, lectura, TripCleaner y validación (v0, v1)
    ├── features.py      <- Panel de demanda, calendario, festivos, rezagos y partición (v2, v4)
    ├── tracking.py      <- log_dataset_version(): versionado de datasets en MLflow
    ├── plots.py         <- Figuras del EDA y del pronóstico
    └── modeling/
        ├── pipeline.py  <- Pipeline de scikit-learn y StationProfilePCA
        ├── train.py     <- Familias, métricas, reentrenamiento y registro en el Model Registry
        ├── tune.py      <- Búsqueda de hiperparámetros con Optuna
        └── predict.py   <- Pronósticos con el modelo @champion
```

## Equipo y forma de trabajo

| Integrante | Rol | En la Fase 2 |
|---|---|---|
| Alonso Rojas Loreto (A01706755) | Data Engineer | v4, Pipeline de scikit-learn, Model Registry, inferencia y catálogo versionado |
| Daniel Fernando Romero Villacis | Data Scientist | Selección de algoritmos y ajuste con Optuna (`tune.py`), análisis de resultados |
| José Luis Israel Servín Gómez | ML Engineer | Revisión del Pipeline y del registro; verificación de reproducibilidad |
| César Guillermo Vázquez Álvarez | Software Engineer / DevOps | CI, revisión de código y flujo de ramas y pull requests |

Cada cambio entra por una rama y un pull request; el CI corre ruff y las pruebas, y otro integrante revisa y aprueba
antes de integrarlo a `main`.

## Uso de inteligencia artificial

Anthropic. (2026). *Claude Opus 5.5* [Modelo de lenguaje grande]. Se utilizó para generar y depurar código, correr
experimentos y redactar la documentación. Cada cambio generado con apoyo de IA pasa por la revisión de otro integrante
en un pull request; el equipo revisó y validó el código, las cifras y las decisiones metodológicas.
