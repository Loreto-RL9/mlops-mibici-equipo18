"""Capa de servicio: arma las variables de cada petición, pronostica y registra el evento.

No depende de FastAPI: la API (mibici/api.py) sólo valida y traduce HTTP, y esta capa se prueba
por separado. Tres decisiones de diseño:

1. **Las mismas transformaciones que en entrenamiento.** El calendario sale de
   `features.calendar_features` y los datos de la estación del mismo catálogo versionado
   (data/external); el resto del preprocesamiento viaja dentro del Pipeline.
2. **Modo degradado en vez de error.** Si el modelo no cargó, la estación no está en el catálogo
   o faltan los rezagos, se responde con la persistencia semanal (la demanda de la misma hora hace
   una semana), que es la línea base del proyecto, y la respuesta lo dice (`fuente`).
3. **Un evento por petición.** Cada pronóstico deja una línea JSON con la versión del modelo,
   entradas, salidas y latencia: es la materia prima del monitoreo de drift.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid

from loguru import logger
import mlflow
import numpy as np
import pandas as pd

from mibici import config
from mibici.features import STATION_COLUMNS, calendar_features

LAG_COLUMNS = ["lag_24h", "lag_168h"]


@dataclass
class ModelBundle:
    """El Pipeline exportado y sus metadatos. `model` es None si no se pudo cargar."""

    model: object | None
    meta: dict = field(default_factory=dict)
    error: str | None = None

    @property
    def version(self) -> str | None:
        return self.meta.get("version")

    def age_days(self, now: datetime | None = None) -> float | None:
        """Días desde que se registró la versión (la alerta usa config.MODEL_MAX_AGE_DAYS)."""
        if "registrado" not in self.meta:
            return None
        now = now or datetime.now(timezone.utc)
        return (now - datetime.fromisoformat(self.meta["registrado"])).total_seconds() / 86400


def load_bundle(model_dir: Path = config.SERVING_MODEL_DIR) -> ModelBundle:
    """Carga el modelo exportado con `make export-model`. Si falla, el servicio arranca en modo
    degradado (persistencia semanal) en lugar de caerse."""
    model_dir = Path(model_dir)
    meta_path = model_dir / "model_meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    try:
        model = mlflow.sklearn.load_model(str(model_dir))
    except Exception as error:  # noqa: BLE001 - cualquier falla deja al servicio en modo degradado
        logger.error(f"No se pudo cargar el modelo de {model_dir}: {error}")
        return ModelBundle(None, meta, error=str(error))
    logger.info(f"Modelo cargado: versión {meta.get('version', '?')} de {model_dir}")
    return ModelBundle(model, meta)


def build_features(requests: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Une cada fila (estación, hora y rezagos) con el catálogo de estaciones y el calendario.
    Una estación que no está en el catálogo queda con capacidad, ubicación y municipio vacíos."""
    df = requests.merge(stations[STATION_COLUMNS], on="estacion_id", how="left")
    return df.join(calendar_features(df["hora"]))


def fallback_values(df: pd.DataFrame) -> pd.Series:
    """Persistencia semanal; si no hay, la de ayer; si tampoco, 0 salidas."""
    return df["lag_168h"].fillna(df["lag_24h"]).fillna(0.0).astype(float)


def predict(bundle: ModelBundle, df: pd.DataFrame) -> pd.DataFrame:
    """Pronóstico por fila con su fuente: "modelo" o "fallback" y el motivo del fallback."""
    out = pd.DataFrame(index=df.index)
    out["salidas_pred"] = fallback_values(df)
    out["fuente"] = "fallback"
    out["motivo"] = None
    known_station = df["capacidad"].notna()
    has_lags = df[LAG_COLUMNS].notna().all(axis=1)
    out.loc[~known_station, "motivo"] = "estacion_desconocida"
    out.loc[known_station & ~has_lags, "motivo"] = "sin_rezagos"
    usable = known_station & has_lags

    if bundle.model is None:
        out.loc[usable, "motivo"] = "modelo_no_disponible"
        return out
    if usable.any():
        try:
            columns = list(bundle.model.feature_names_in_)
            y = bundle.model.predict(df.loc[usable, columns])
            out.loc[usable, "salidas_pred"] = np.clip(y, 0, None)
            out.loc[usable, "fuente"] = "modelo"
        except Exception as error:  # noqa: BLE001 - el servicio responde con el fallback
            logger.error(f"Falló la predicción; se usa el fallback: {error}")
            out.loc[usable, "motivo"] = "error_del_modelo"
    return out


class EventLogger:
    """Agrega una línea JSON por petición (JSON Lines). Es barato, legible y se analiza con
    pandas: `pd.read_json(path, lines=True)`."""

    def __init__(self, path: Path = config.PREDICTION_LOG):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict) -> None:
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def serve(
    bundle: ModelBundle,
    requests: pd.DataFrame,
    stations: pd.DataFrame,
    event_logger: EventLogger | None = None,
) -> tuple[str, pd.DataFrame]:
    """De la petición al pronóstico: variables, predicción y evento. Devuelve (request_id, filas)."""
    start = time.perf_counter()
    request_id = uuid.uuid4().hex
    features = build_features(requests, stations)
    result = predict(bundle, features)
    rows = pd.concat([requests[["estacion_id", "hora"]], result], axis=1)
    if event_logger is not None:
        event_logger.log(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "request_id": request_id,
                "modelo": bundle.meta.get("nombre", config.REGISTERED_MODEL),
                "version": bundle.version,
                "alias": bundle.meta.get("alias"),
                "filas": len(rows),
                "filas_fallback": int((rows["fuente"] == "fallback").sum()),
                "latencia_ms": round(1000 * (time.perf_counter() - start), 2),
                "entradas": requests.assign(hora=requests["hora"].astype(str)).to_dict("records"),
                "salidas": rows[["salidas_pred", "fuente", "motivo"]].to_dict("records"),
            }
        )
    return request_id, rows
