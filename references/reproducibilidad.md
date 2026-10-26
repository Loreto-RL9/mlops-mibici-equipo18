# Verificación de reproducibilidad — Fase 1

Responsable: José Luis Israel Servín Gómez (ML Engineer).

## Entorno

- Fecha de la verificación: 25 de octubre de 2026.
- Sistema operativo: Windows 10, compilación 19045 (`Windows-10-10.0.19045-SP0`).
- Terminal: PowerShell.
- Python utilizado: 3.13.2, dentro del entorno virtual `.venv`.
- Python predeterminado del sistema: 3.14.0; no se utilizó para esta ejecución.
- Repositorio: https://github.com/Loreto-RL9/mlops-mibici-equipo18
- Commit de partida: `249b09ba6cb646e65e1f4f5b07a46ce8a253bffa`.

El repositorio ya estaba clonado al iniciar la verificación. No se realizó una segunda clonación.
GNU Make no estaba disponible en esta computadora, por lo que se utilizaron los comandos
equivalentes de las recetas del `Makefile` desde PowerShell.

## Instalación

Comandos equivalentes a `make requirements`, precedidos por la creación del entorno:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

La primera creación del entorno falló por permisos de escritura en los temporales de
`ensurepip` dentro del entorno restringido del asistente. Se reintentó con los permisos
de ejecución correspondientes. Esta incidencia pertenece al entorno de ejecución.

La actualización de pip terminó correctamente (versión 26.2.2) y la instalación
de `requirements.txt` se completó con código 0.

Versiones instaladas: pandas 3.1.0, NumPy 2.5.4, scikit-learn 1.9.1,
MLflow 3.17.2, PyArrow 25.0.1, holidays 0.107 y pytest 9.1.1.
El archivo de requisitos fija mínimos, por lo que estas versiones pueden variar
en instalaciones futuras.

Para generar las gráficas se utilizó Matplotlib 3.11.2 con una caché de fuentes
dentro del entorno virtual:

```powershell
$env:MPLCONFIGDIR = Join-Path (Get-Location).Path '.venv\matplotlib-cache'
.\.venv\Scripts\python.exe -m mibici.plots
```

Con esta configuración se generaron correctamente las ocho figuras y se registraron
en MLflow bajo el run `eda-figuras`.

## Ejecución del pipeline

Estado: **pipeline completado**, mediante comandos equivalentes en PowerShell y
la alternativa de descarga descrita abajo.

Las pruebas con pytest 9.1.1 finalizaron correctamente:
**5 passed**, con cinco avisos de obsolescencia de `Series.dt.dayofweek`.

Los comandos equivalentes a `make pipeline` se ejecutan en este orden,
continuando únicamente si el anterior termina correctamente:

```powershell
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe -m mibici.dataset download
.\.venv\Scripts\python.exe -m mibici.dataset clean
.\.venv\Scripts\python.exe -m mibici.features
.\.venv\Scripts\python.exe -m mibici.plots
.\.venv\Scripts\python.exe -m mibici.modeling.train
```

La descarga con `urllib` falló con `HTTP Error 406: Not Acceptable` en el primer CSV,
también al probar encabezados de navegador. El mismo enlace respondió HTTP 200 con
`curl.exe`. Se descargaron las mismas URLs de `mibici/config.py` mediante este comando:

```powershell
@'
import subprocess
from mibici import config

urls = [
    f'{config.BASE_URL}/{folder}/datos_abiertos_{month}.csv'
    for month, folder in config.TRIP_FILES.items()
] + [config.STATIONS_URL]
for url in urls:
    target = config.RAW_DATA_DIR / url.rsplit('/', 1)[-1]
    subprocess.run([
        'curl.exe', '--fail', '--location', '--silent', '--show-error',
        '--retry', '2', '--max-time', '180', '--output', str(target), url,
    ], check=True)
'@ | .\.venv\Scripts\python.exe -
```

Se obtuvieron 13 CSV (12 meses y catálogo), con **390,347,882 bytes (390.35 MB)**.
Después, el comando original de descarga finalizó con código 0 al reconocer los
archivos existentes. La limpieza finalizó correctamente; enero de 2026 se leyó
como Latin-1 mediante la alternativa ya prevista en el código.
Las cuatro versiones Parquet, las ocho gráficas y la evaluación de los modelos
finalizaron correctamente. Todos los runs quedaron en estado `FINISHED`.
No se modificó el código del pipeline.

Las métricas de validación también coinciden con el README al redondear a tres decimales:

| Modelo | MAE | RMSE |
|---|---:|---:|
| Persistencia 1 h | 1.127 | 1.981 |
| Persistencia 24 h | 1.093 | 2.050 |
| Persistencia semanal | 1.014 | 1.832 |
| HGB Poisson 1 h | 0.849 | 1.400 |
| HGB Poisson 24 h | 0.875 | 1.401 |

## Comparación con el README

| Versión | Dataset | Filas esperadas según README | Filas obtenidas | ¿Coinciden? |
|---|---|---:|---:|---|
| v0 | Viajes crudos | 4,497,877 | 4,497,877 | Sí |
| v1 | Viajes limpios | 4,311,657 | 4,311,657 | Sí |
| v2 | Panel de demanda | 2,870,312 | 2,870,312 | Sí |
| v3 | Datos listos para modelar | 2,804,652 | 2,804,652 | Sí |

Los conteos se consultaron directamente en los metadatos de los Parquet locales
usando `pyarrow.parquet.ParquetFile(path).metadata.num_rows`:
`data/interim/trips_raw.parquet`, `data/interim/trips_clean.parquet`,
`data/processed/demand_panel.parquet` y `data/processed/model_ready.parquet`.
Las particiones de v3 también coinciden: train **1,973,154**, valid **552,001** y
test **279,497** filas.

## MLflow

Se inició el equivalente a `make mlflow-ui`, limitado a la interfaz local:

```powershell
.\.venv\Scripts\python.exe -m mlflow ui --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 8080 --workers 1
```

La interfaz respondió **HTTP 200** y se abrió en el navegador en
`http://127.0.0.1:8080`. El backend es `mlflow.db` en la raíz del proyecto.
El arranque inicial con varios workers presentó errores de socket en Windows
(`WinError 10022`) y una conexión interrumpida al consultar la API. Se reinició con
`--workers 1` para evitar el problema de procesos concurrentes.
Tras el reinicio se confirmó HTTP 200 en la interfaz y se consultó la API
`/api/2.0/mlflow/runs/search`: los cuatro runs de datos aparecen como `FINISHED`
y sus métricas `filas` coinciden con los Parquet y con la tabla del README.

Runs de esta ejecución en `mibici-fase1-datos`:

| Versión | Run ID |
|---|---|
| v0 | `71decd8a6b924cdcb16ee4255e4744de` |
| v1 | `0196b0fb31cf4263ad32d1c8029bf13a` |
| v2 | `6c7948b309394413aba27b73201f8b39` |
| v3 | `3206645cb62e404580690787f7036e65` |

MLflow avisó que su backend de ejecución de jobs no soporta Windows; la interfaz
de seguimiento sí inició y respondió. Ese aviso no impidió registrar los datasets
desde los scripts locales.

## Conclusión

Los conteos de **v0, v1, v2 y v3 coinciden exactamente con el README** en Windows 10
y Python 3.13.2. La reproducción requirió descargar con `curl.exe`
los archivos que el servidor rechazó al usar `urllib`.
También se completaron las ocho gráficas y los modelos iniciales, cuyas métricas
coinciden con las publicadas a tres decimales. Las figuras de `reports/figures/`
se regeneraron localmente y Git las muestra como modificadas; sus archivos binarios
pueden diferir por las versiones de Matplotlib y las fuentes del sistema.

## Uso de inteligencia artificial

Se utilizó OpenAI Codex (GPT-6) para asistir en la ejecución de comandos, revisar los
resultados y redactar este registro. El responsable debe revisar el documento y las
evidencias antes de incluirlos en la entrega.
