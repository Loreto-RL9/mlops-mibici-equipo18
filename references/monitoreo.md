# Monitoreo: data drift y pérdida de desempeño (Entrega Final)

`make drift` compara la **línea base**, que es el set de validación (julio–agosto 2026, MAE 0.865), contra:
- **septiembre real**, el cambio estacional del mes de prueba;
- **tres escenarios sintéticos** que alteran la validación.

Para cada caso mide el drift de las entradas, de la predicción y de la demanda real. También calcula el desempeño contra
la línea base, enciende alertas con umbrales fijos y propone una acción. Todo queda en `reports/monitoreo/` y en el
experimento `mibici-monitoreo` de MLflow (un run por escenario, con tablas y figuras). `make drift-evidently` genera los
mismos reportes con el `DataDriftPreset` de Evidently.

## Pruebas de drift (regla por omisión de Evidently con más de 1,000 filas)

| Tipo de columna | Prueba | Hay drift si |
|---|---|---|
| Numérica (`capacidad`, `lat`, `lon`, `lag_24h`, `lag_168h`, `hora_dia`, predicción, demanda real) | Distancia de Wasserstein / desviación estándar de la referencia | > 0.1 |
| Categórica (`municipio`, `dia_semana`, `fin_de_semana`, `festivo`) | Distancia de Jensen-Shannon | > 0.1 |
| Dataset completo | Proporción de entradas con drift | ≥ 50 % |

Se reporta además el **PSI**: menos de 0.1 es estable; más de 0.2, cambio significativo. Las muestras son de 60,000 filas
con semilla 42.

**¿Por qué una implementación propia además de Evidently?** Evidently 0.7.23, la versión más reciente, falla al
importarse con pandas 3:

```
ImportError: cannot import name 'infer_dtype_from_object' from 'pandas.core.dtypes.common'
```

`uv.lock` fija pandas 3.1.1, así que el pipeline usa la misma regla implementada con SciPy y Evidently corre en un
entorno aislado con pandas 2 (`make drift-evidently`). Las dos implementaciones coinciden en todos los escenarios:

| Escenario | Columnas con drift: nuestra prueba | Columnas con drift: Evidently |
|---|---|---|
| Validación | 0 | 0 |
| Septiembre real | 3 | 3 |
| Aumento de demanda | 3 | 3 |
| Rezagos rotos | 2 | 2 |
| Cierre de estaciones | 0 | 0 |

## Umbrales de alerta (consecuencias acordadas de antemano)

| Alerta | Se enciende si | Por qué ese umbral |
|---|---|---|
| Desempeño | MAE > 1.15 × MAE de la línea base | En entrenamiento, el MAE mensual varía entre −7 % y +6 % alrededor de su media |
| Sesgo | \|sesgo medio\| > 0.15 salidas | ≈10 % de la demanda media (1.5 salidas por estación-hora) |
| Persistencia gana | MAE del modelo ≥ MAE de la persistencia semanal | El modelo dejó de aportar valor; la API debería servir el fallback |
| Edad del modelo | Más de 8 días desde su registro (`/health`) | Reentrenamiento semanal + 1 día de margen |

## Resultados

| Escenario | Qué simula | Entradas con drift | Demanda real con drift | MAE (vs base) | Sesgo | Alertas | Acción propuesta |
|---|---|---|---|---|---|---|---|
| Validación (base) | — | 0 | No | 0.865 (1.00×) | +0.06 | — | Sin acción |
| **Septiembre real** | Cambio estacional, 58 estaciones nuevas | 2 (`lag_24h`, `festivo`) | **Sí** | 1.005 (**1.16×**) | **−0.23** | Desempeño y sesgo | Reentrenar con los meses recientes |
| Aumento de demanda | +60 % en 30 % de las estaciones (desplazamiento de medias) | 2 (`lag_24h`, `lag_168h`) | **Sí** | 1.088 (**1.26×**) | −0.13 | Desempeño | Reentrenar con los meses recientes |
| Rezagos rotos | 25 % de filas con los rezagos en 0 (variables faltantes) | 2 (`lag_24h`, `lag_168h`) | No | 0.909 (1.05×) | −0.08 | — | Revisar el pipeline de features y la fuente |
| Cierre de estaciones | La demanda cae al 10 % en 15 % de las estaciones y los rezagos no cambian (drift de concepto) | **0** | No | 0.903 (1.04×) | **+0.19** | Sesgo | Confirmar con el operador y reentrenar |

**Lectura:**
1. **El drift de entradas no basta.** El cierre de estaciones no mueve ninguna entrada (drift de concepto) y sólo lo
   delata el sesgo, al comparar el pronóstico con la demanda real (*actuals*). Al revés, los rezagos rotos sí mueven las
   entradas, pero el modelo apenas empeora. Por eso se monitorean entradas, predicción, demanda real y error.
2. **Cada patrón de alertas lleva a una acción distinta.** Cuando cambió la demanda real se reentrena; cuando cambiaron
   las entradas y no el error, se revisa la fuente de datos; cuando sube el sesgo sin drift de entradas, se investiga
   con el operador.
3. **En todos los escenarios el modelo sigue ganándole a la persistencia semanal**: nunca se encendió esa alerta.

## Hallazgo: septiembre trae 58 estaciones que el modelo no vio

| Estaciones | Cuántas | Filas | Demanda media | MAE modelo | Sesgo | MAE persistencia | MAE con arranque en frío |
|---|---|---|---|---|---|---|---|
| Vistas al entrenar | 410 | 245,292 | 1.83 | 1.042 | −0.31 | 1.254 | 1.042 |
| **Nuevas** | 58 | 34,205 | 0.54 | **0.743** | **+0.39** | **0.562** | 0.562 |
| Total | 468 | 279,497 | 1.67 | 1.005 | −0.23 | 1.169 | **0.983** |

Una estación nueva recibe el perfil horario promedio (`StationProfilePCA`) y el modelo la **sobreestima**: ahí la
persistencia semanal es mejor. Por eso la API aplica una **política de arranque en frío**
(`config.COLD_START_FALLBACK`): a las estaciones que el modelo no vio les responde con la persistencia y lo marca con
`motivo: "estacion_nueva"`. En septiembre eso baja el MAE total de 1.005 a **0.983 (−2.2 %)** sin reentrenar.

En las estaciones que sí vio, el sesgo de −0.31 confirma el cambio estacional: septiembre es el mes de más demanda y
el modelo se entrenó hasta junio. La acción es **reentrenar incluyendo julio y agosto**; queda como trabajo futuro,
junto con el entrenamiento continuo.

## Monitoreo en operación

La API escribe una línea JSON por petición en `logs/predicciones.jsonl`, con entradas, salidas, versión y latencia.
`make drift-log` compara esas peticiones contra la referencia guardada, con las mismas pruebas.
