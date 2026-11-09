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
# El portal retiró este catálogo (404 desde nov 2026, hallazgo de Daniel); la copia se versiona
# en el repositorio, en data/external, para que el pipeline no dependa de que siga publicado.
STATIONS_FILE = EXTERNAL_DATA_DIR / "nomenclatura_2026_09.csv"

# Reglas de limpieza (justificación en references/diccionario_datos.md) --------
MIN_DURATION_MIN = 1  # menos de 1 minuto: registro técnico, no un viaje
SAME_STATION_MAX_MIN = 2  # misma estación y < 2 min: arranque fallido (bici devuelta)
MAX_DURATION_MIN = 180  # más de 3 h: bici mal anclada o incidencia del sistema
BIRTH_YEAR_RANGE = (1930, 2012)  # fuera de rango -> se marca como desconocido
CLOSED_HOURS = (1, 2, 3, 4)  # 1:00-4:59, el sistema no opera (120 salidas en 12 meses)

# Partición temporal: entrenamiento oct-jun, validación jul-ago, prueba sep
TRAIN_END = "2026-06-30 23:00"
VALID_END = "2026-08-31 23:00"

# PCA de los perfiles horarios de cada estación: componentes hasta explicar el 80 %
PCA_VARIANCE = 0.80

# Selección de modelos y ajuste de hiperparámetros (Fase 2) -------------------
SEED = 42
# Horizonte del pronóstico: "24h" = la noche anterior (solo rezagos de 24 h o más); "1h" = la hora
# siguiente (puede usar la demanda de la hora anterior).
HORIZON = "24h"
OPTUNA_TRIALS = 25  # pruebas por familia de modelos
OPTUNA_SAMPLE_ROWS = 400_000  # muestra de entrenamiento para que cada prueba tarde segundos
TOP_STATIONS_SHARE = 0.20  # el 20 % de las estaciones genera el 54 % de las salidas
MODEL_DEFAULTS = {
    "poisson_glm": {"max_iter": 1000},
    "random_forest": {"max_samples": 0.2, "n_jobs": -1, "random_state": SEED},
    "hgb_poisson": {"loss": "poisson", "early_stopping": False, "random_state": SEED},
}
# Cada especificación se pasa al método suggest_{type} de Optuna. Los rangos quedan aquí para
# que el módulo de búsqueda no contenga hiperparámetros fijos.
OPTUNA_SEARCH_SPACES = {
    "poisson_glm": {
        "alpha": {"type": "float", "low": 1e-6, "high": 1.0, "log": True},
    },
    "random_forest": {
        "n_estimators": {"type": "int", "low": 50, "high": 250, "step": 25},
        "max_depth": {"type": "int", "low": 8, "high": 22},
        "min_samples_leaf": {"type": "int", "low": 30, "high": 1000, "log": True},
        "max_features": {"type": "float", "low": 0.3, "high": 0.8},
    },
    "hgb_poisson": {
        "learning_rate": {"type": "float", "low": 0.02, "high": 0.3, "log": True},
        "max_iter": {"type": "int", "low": 200, "high": 1500, "step": 100},
        "max_leaf_nodes": {"type": "int", "low": 15, "high": 255, "log": True},
        "min_samples_leaf": {"type": "int", "low": 20, "high": 500, "log": True},
        "l2_regularization": {"type": "float", "low": 1e-3, "high": 10.0, "log": True},
    },
}

# MLflow ----------------------------------------------------------------------
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{PROJ_ROOT / 'mlflow.db'}")
MLFLOW_EXPERIMENT = "mibici-fase1-datos"
MLFLOW_EXPERIMENT_MODELOS = "mibici-fase2-modelos"
REGISTERED_MODEL = "mibici-demanda-estacion-hora"  # nombre en el Model Registry
MODEL_ALIAS = "champion"

# loguru compatible con barras de progreso de tqdm
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
