# Selección de modelos, ajuste de hiperparámetros y registro — Fase 2

**Roles:** Data Scientist (Daniel Romero: algoritmos, búsqueda con Optuna y análisis) y Data Engineer (Alonso Rojas:
Pipeline, v4 y Model Registry) · **Datos:** v4 `demanda-v4` · **Horizonte:** 24 h (pronóstico hecho la noche anterior:
sin `lag_1h` ni `media_24h`) · **Experimento MLflow:** `mibici-fase2-modelos` · **Modelo registrado:**
`mibici-demanda-estacion-hora` · **Comando:** `make tune` (`python -m mibici.modeling.tune`). Las cifras salen de
[`comparacion_modelos_24h.csv`](comparacion_modelos_24h.csv), que se arma con las corridas registradas en MLflow.

## Por qué estos algoritmos

La demanda es un **conteo con muchos ceros**: el 46 % de las estación-horas no tiene salidas y la cola derecha es larga
([hallazgos del EDA](hallazgos_eda.md), figura 5). La meta del [ML Canvas](../references/ml_canvas.md) es bajar el MAE
al menos 10 % frente a la persistencia semanal (misma hora de la semana anterior).

| Familia | Por qué se probó |
|---|---|
| Regresión de Poisson (GLM, liga log) | Es la referencia para conteos: predice medias positivas y sus coeficientes se interpretan. Si un modelo lineal bastara, no haría falta nada más complejo. |
| Bosque aleatorio | Capta interacciones no lineales (hora × tipo de estación) promediando árboles independientes; es poco sensible a los hiperparámetros. |
| HistGradientBoosting con pérdida Poisson | Une la flexibilidad de los árboles con la pérdida propia de un conteo y escala a millones de filas. Era la línea base de la Fase 1, sin ajustar. |

## Cómo se ajustó

- Cada candidato es un **Pipeline de scikit-learn** (`mibici/modeling/pipeline.py`): codificación cíclica, escalado,
  one-hot del municipio y PCA del perfil horario de cada estación, seguidos del estimador. El preprocesamiento se ajusta
  dentro de cada prueba con los mismos datos que el estimador, así que no hay fuga de validación hacia entrenamiento.
- **Optuna** con búsqueda bayesiana (TPE, semilla 42), 25 pruebas por familia. Cada prueba es un run anidado en MLflow
  con sus hiperparámetros, MAE, RMSE, devianza de Poisson, sesgo, MAE en las estaciones de mayor demanda y segundos de
  ajuste.
- Cada prueba entrena con una muestra de 400,000 filas del entrenamiento (oct 2025 a jun 2026) y se evalúa con toda la
  validación (jul y ago 2026, 552,001 filas). Optuna minimiza la **devianza de Poisson**, la métrica que corresponde a
  un modelo de conteos.
- El mejor juego de cada familia se **reentrena con todo el entrenamiento** (1,973,154 filas), se evalúa en validación
  y **una sola vez en prueba** (sep 2026, 279,497 filas) y se registra en el Model Registry.
- **De dónde salen los rangos.** En una primera búsqueda sobre v3 (PR #1), el mejor boosting quedó en el límite del
  rango (600 iteraciones) y el bosque también quedó en los bordes, así que se ampliaron los rangos (tabla de abajo).
  Con esos rangos, la búsqueda se repitió dentro del Pipeline sobre v4.

| Familia | Hiperparámetro | Rango | Mejor valor |
|---|---|---|---:|
| Regresión de Poisson | `alpha` (regularización) | 1e-6 a 1, log | 1.1e-06 |
| Bosque aleatorio | `n_estimators` | 50 a 250 | 200 |
| | `max_depth` | 8 a 22 | 17 |
| | `min_samples_leaf` | 30 a 1,000, log | 30 |
| | `max_features` | 0.3 a 0.8 | 0.537 |
| HistGradientBoosting Poisson | `learning_rate` | 0.02 a 0.3, log | 0.0323 |
| | `max_iter` | 200 a 1,500 | 600 |
| | `max_leaf_nodes` | 15 a 255, log | 179 |
| | `min_samples_leaf` | 20 a 500, log | 155 |
| | `l2_regularization` | 0.001 a 10, log | 1.187 |

## Resultados (horizonte 24 h)

| Modelo | Versión | MAE valid | MAE prueba | RMSE prueba | Devianza prueba | Sesgo prueba | MAE prueba, 20 % de estaciones con más demanda |
|---|---:|---:|---:|---:|---:|---:|---:|
| Persistencia semanal | — | 1.014 | 1.169 | 2.171 | — | −0.140 | 2.375 |
| Regresión de Poisson | 1 | 1.207 | 1.342 | 2.900 | 1.974 | −0.120 | 2.540 |
| Bosque aleatorio | 3 | 0.902 | 1.042 | 1.784 | 1.319 | −0.150 | 2.005 |
| **HistGradientBoosting Poisson** | 2 (`@champion`) | **0.865** | **1.005** | **1.712** | **1.281** | −0.228 | **1.896** |

MAE y RMSE en bicis por estación y hora. El sesgo es la predicción media menos la real: negativo = el modelo se queda
corto. La devianza no aplica a la persistencia, porque predice ceros exactos y la devianza de Poisson exige
predicciones positivas.

- **El boosting gana en todas las métricas salvo el sesgo.** Baja el MAE 14.6 % en validación y 14.0 % en prueba frente
  a la persistencia semanal (20.2 % en las estaciones de mayor demanda), así que cumple la meta de −10 % del ML Canvas.
- **El modelo lineal no alcanza:** queda peor que la persistencia en MAE. Sin árboles no capta interacciones como
  «estación de oficinas a las 8 h», lo que justifica los modelos basados en árboles.
- **El bosque queda en segundo lugar** (MAE de prueba 1.042) y su modelo pesa 162 MB contra 12 MB del boosting, lo que
  importa para servirlo en un contenedor en la Fase 3.
- **Ajustar hiperparámetros aporta poco frente a elegir bien el algoritmo.** El boosting pasó de 0.875 (configuración
  por defecto de la Fase 1) a 0.865 en validación (−1.1 %); casi toda la ganancia viene de usar árboles con la pérdida
  correcta para conteos.
- **Mover el preprocesamiento al Pipeline no cambió el desempeño:** con la búsqueda sobre v3 (PR #1) el boosting tuvo
  0.861 en validación y 1.000 en prueba; ahora, 0.865 y 1.005, y el bosque 0.901/1.040 contra 0.902/1.042.

![Historial de la optimización del boosting](figures/fase2_optuna_historial_hgb_poisson.png)

![Importancia de los hiperparámetros del boosting](figures/fase2_optuna_importancia_hgb_poisson.png)

La mejor prueba del boosting fue la última (devianza 1.133 contra 1.142 de la segunda): la búsqueda seguía mejorando,
pero a un ritmo bajo (0.7 % en 23 pruebas). Según la figura de Optuna, el número de iteraciones (34 %) y la
regularización L2 (31 %) explican la mayor parte de la variación de la devianza entre pruebas. Con 25 pruebas esa
estimación es ruidosa, así que conviene leerla como orientación y no como un ranking exacto.

## Model Registry

El Pipeline final de cada familia es una versión de `mibici-demanda-estacion-hora`, con:

- **Descripción** con la familia, el horizonte, el dataset y el MAE de validación y de prueba.
- **Etiquetas** con los hiperparámetros (`param_*`) y todas las métricas de validación y prueba (`mae_valid`,
  `devianza_poisson_valid`, `mae_test`, `sesgo_test`, `mae_top_estaciones_test`, …).
- **Firma de entrada** (las 11 columnas de v4 que usa el Pipeline) y un ejemplo de entrada.
- En su run: figuras de Optuna, el pronóstico de una semana, el MAE por hora y la lista de variables.

`set_champion()` asigna el alias **`champion`** a la versión con menor devianza de Poisson en validación, la misma
métrica que optimiza Optuna; `predict.py` y el notebook `2.0-equipo18-analisis-del-modelo` cargan ese alias. Hoy es
la versión 2 (HistGradientBoosting Poisson). Si un reentrenamiento produce una versión mejor, el alias se
mueve y el código de inferencia no cambia.

## Qué vigilar

- **Septiembre se subestima** (sesgo −0.23 bicis por hora en prueba). El EDA lo anticipaba: septiembre es el mes más
  alto del año, ≈35 % arriba del promedio de los meses de entrenamiento. Antes de pronosticar un mes nuevo conviene
  reentrenar con los meses más recientes (julio y agosto incluidos) y, en la Fase 3, vigilar este desvío como deriva.
- **El error se concentra donde más importa:** el MAE de las estaciones de mayor demanda (1.9) duplica el global (1.0),
  y por hora del día el error es máximo en los picos de las 8 y las 18 h. El monitoreo debería reportarlos por separado.

## Interacción con los otros roles

- **Data Engineer → Data Scientist:** v4 y el Pipeline definen qué recibe cada modelo; la búsqueda solo cambia el
  estimador final. Daniel encontró que el catálogo `nomenclatura_2026_09.csv` ya no existe en mibici.net (HTTP 404) y lo
  recuperó del Internet Archive (misma huella SHA-1); ahora esa copia se versiona en `data/external/`.
- **ML Engineer:** revisa el Pipeline, el registro y la carga con `predict.py`, y verifica que el flujo se reproduzca en
  otra máquina (en la Fase 1 encontró el HTTP 406 de la descarga, que ahora tiene respaldo con curl).
- **Software Engineer / DevOps:** `tests/test_tune.py` y `tests/test_pipeline.py` corren en el CI con datos sintéticos
  (no necesitan MLflow ni descargas). Con 2 núcleos, `make tune` tarda alrededor de 45 minutos; cada familia puede
  correrse por separado.

## Uso de inteligencia artificial

Anthropic. (2026). *Claude Opus 5.5* [Modelo de lenguaje grande]. Se utilizó para escribir `mibici/modeling/tune.py`,
`train.py`, `pipeline.py` y sus pruebas, correr los experimentos y redactar este documento. Los criterios de evaluación
(sesgo y error en las estaciones de mayor demanda) salen de los hallazgos del EDA de la Fase 1. El código pasa por la
revisión de otro integrante del equipo en el pull request antes de integrarse a `main`.
