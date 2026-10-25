"""Configuración central: rutas, fuente de datos, reglas de limpieza y MLflow."""

import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# Rutas -----------------------------------------------------------------------
PROJ_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

for _d in (RAW_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Fuente: datos abiertos de MiBici (https://www.mibici.net/es/datos-abiertos/)
# 12 meses, octubre 2025 a septiembre 2026. Cada mes vive en su propia carpeta.
BASE_URL = "https://www.mibici.net/site/assets/files"
TRIP_FILES = {
    "2025_10": 115104, "2025_11": 116773, "2025_12": 118153,
    "2026_01": 120163, "2026_02": 121761, "2026_03": 123236,
    "2026_04": 124927, "2026_05": 125943, "2026_06": 127734,
    "2026_07": 129006, "2026_08": 130362, "2026_09": 131825,
}  # fmt: skip
STATIONS_URL = f"{BASE_URL}/1118/nomenclatura_2026_09.csv"
STATIONS_FILE = RAW_DATA_DIR / "nomenclatura_2026_09.csv"

# Reglas de limpieza (justificación en references/diccionario_datos.md) --------
MIN_DURATION_MIN = 1  # menos de 1 minuto: registro técnico, no un viaje
SAME_STATION_MAX_MIN = 2  # misma estación y < 2 min: arranque fallido (bici devuelta)
MAX_DURATION_MIN = 180  # más de 3 h: bici mal anclada o incidencia del sistema
BIRTH_YEAR_RANGE = (1930, 2012)  # fuera de rango -> se marca como desconocido
CLOSED_HOURS = (1, 2, 3, 4)  # el sistema no opera: solo 100 salidas en 12 meses

# Partición temporal: entrenamiento oct-jun, validación jul-ago, prueba sep
TRAIN_END = "2026-06-30 23:00"
VALID_END = "2026-08-31 23:00"

# PCA de los perfiles horarios de cada estación: componentes hasta explicar el 80 %
PCA_VARIANCE = 0.80

# MLflow ----------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "mibici-fase1-datos"

# loguru compatible con barras de progreso de tqdm
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
