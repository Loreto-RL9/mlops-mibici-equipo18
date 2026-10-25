# Diccionario de datos y reglas de limpieza

## Fuente

Datos abiertos de MiBici, el sistema de bicicleta pública del Área Metropolitana de Guadalajara
(<https://www.mibici.net/es/datos-abiertos/>). Se usaron **12 meses, de octubre de 2025 a septiembre de 2026**
(un CSV por mes) y el catálogo de estaciones `nomenclatura_2026_09.csv`. `make download` los descarga.

## Viajes (`datos_abiertos_AAAA_MM.csv`)

| Columna original | Columna en el proyecto | Tipo | Descripción |
|---|---|---|---|
| Viaje_Id | `viaje_id` | entero | Identificador del viaje |
| Usuario_Id | `usuario_id` | entero | Identificador seudónimo del usuario |
| Genero | `genero` | texto | M, F o `Desconocido` (en crudo viene como `NULL`) |
| Año_de_nacimiento | `anio_nacimiento` | entero | Nulo si no es número o está fuera de [1930, 2012] |
| Inicio_del_viaje | `inicio` | fecha-hora | Hora local de salida (Guadalajara no aplica horario de verano) |
| Fin_del_viaje | `fin` | fecha-hora | Hora local de llegada |
| Origen_Id | `origen_id` | entero | Estación de salida |
| Destino_Id | `destino_id` | entero | Estación de llegada |
| (derivada) | `duracion_min` | real | `fin - inicio` en minutos |

## Estaciones (`nomenclatura_2026_09.csv`)

| Columna | En el proyecto | Descripción |
|---|---|---|
| id | `estacion_id` | Identificador (484 estaciones) |
| name | `nombre` | Incluye el municipio: `(GDL-001)`, `(ZPN-...)`, `(TLQ-...)` |
| latitude, longitude | `lat`, `lon` | Coordenadas |
| dpcapacity | `capacidad` | Anclajes de la estación (0 a 39) |
| (derivada) | `municipio` | GDL (357), ZPN (110), TLQ (17) |
| (derivada) | `retirada` | El nombre empieza con `ex-` (14 estaciones dadas de baja) |

## Hallazgos de calidad en los datos crudos (4,497,877 viajes)

* El sistema no opera de 1:00 a 4:59: solo hay 120 salidas registradas en ese horario en 12 meses.

* El archivo de enero de 2026 viene en **latin-1**; los otros 11 meses, en UTF-8. El lector prueba ambos.
* 758 filas duplicadas exactas (mismo `viaje_id`).
* 3,791 viajes con duración cero o negativa.
* 295,263 viajes que regresan a la misma estación; 164,384 duran menos de 2 minutos (arranques fallidos).
* Duración mediana de 9.35 min; máximo de 5,920 min (4 días).
* 534 años de nacimiento no numéricos y 319 anteriores a 1930 (incluso el año 199).
* 32,982 registros con género `NULL`.
* Viajes con estaciones que no están en el catálogo actual: 3,954 (1,960 por el origen y 2,123 por el
  destino; algunos fallan en ambos).
* La estación 194 (Av. Vallarta / Av. de Los Arcos) registra picos de 90 a 143 salidas por hora el
  primer sábado del mes, entre 20 y 22 h, en 7 de los 12 meses. El patrón es recurrente y compatible con un
  evento ciclista nocturno, así que **se conserva** como demanda real.

## Reglas de limpieza (en este orden; `mibici/dataset.py`, clase `TripCleaner`)

Las reglas se aplican en orden y cada una cuenta solo los viajes que quedaban después de las anteriores.
Por eso algunos conteos son menores que los del perfilado crudo (por ejemplo, 162,225 arranques fallidos
frente a 164,384, porque los duplicados y las duraciones cero o negativas ya se habían quitado).

| # | Regla | Viajes eliminados | Justificación |
|---|---|---:|---|
| 1 | Duplicados por `viaje_id` | 758 | El mismo viaje contado dos veces infla la demanda |
| 2 | Fecha de inicio o fin no interpretable | 0 | Sin fecha no se puede asignar a una hora |
| 3 | Duración ≤ 0 | 3,791 | Un viaje no puede terminar antes de empezar (error de registro) |
| 4 | Misma estación y < 2 min | 162,225 | El usuario sacó y devolvió la bici: no es demanda real de viaje |
| 5 | Duración < 1 min | 14,983 | Registros técnicos; no da tiempo de llegar a otra estación |
| 6 | Duración > 180 min | 510 | Bici mal anclada o incidencia del sistema (0.01 %) |
| 7 | Salida de 1:00 a 4:59 | 100 | El sistema no opera en ese horario (0.002 % de los viajes) |
| 8 | Estación fuera del catálogo | 3,853 | No hay ubicación ni capacidad para esas estaciones |
| | **Total** | **186,220 (4.1 %)** | Quedan **4,311,657** viajes |

Correcciones que no eliminan filas: `NULL` en género → `Desconocido`; año de nacimiento no numérico o
fuera de [1930, 2012] → nulo (300 viajes limpios con año fuera de rango).

Después de limpiar, `check_clean_trips()` verifica con `assert` que no haya duplicados, que todo viaje
termine después de empezar, que la duración esté en [1, 180] min, que ninguna salida caiga en horas sin
servicio y que ambas estaciones existan. Si alguna
condición falla, el pipeline se detiene.

## Panel de demanda (v2) y dataset para modelar (v3)

* **Unidad de análisis:** estación-hora. **Objetivo:** `salidas`, el número de viajes que inician en la estación
  durante esa hora.
* Las horas sin viajes cuentan como 0, pero **solo dentro del periodo activo** de cada estación (de su primera a su
  última salida), para no inventar ceros antes de su apertura o después de su retiro.
* Las horas sin servicio (1:00 a 4:59) se quitan del panel **después** de calcular los rezagos, para que el rezago de
  24 h siga apuntando a la misma hora del día anterior. El panel queda en 2,870,312 estación-horas de 469 estaciones.
* Variables: `hora_dia`, `dia_semana`, `fin_de_semana`, `festivo` (calendario oficial de México, librería
  `holidays`), codificación cíclica (`hora_sin`, `hora_cos`, `dia_sin`, `dia_cos`), `capacidad`, `lat`, `lon`,
  `municipio`, rezagos de 1, 24 y 168 h y la media móvil de las 24 h previas.
* v3 agrega: one-hot de `municipio`, componentes PCA del perfil horario de cada estación (40 valores: 20 horas de servicio entre semana y
  20 en fin de semana; 15 componentes explican el 80.7 % de la varianza) y `StandardScaler`
  ajustado **solo con la partición de entrenamiento** (oct 2025 a jun 2026). Validación: jul y ago 2026.
  Prueba: sep 2026.
* El mes no se usa como variable: con 12 meses de datos, los meses de validación y prueba nunca aparecen en
  entrenamiento.
