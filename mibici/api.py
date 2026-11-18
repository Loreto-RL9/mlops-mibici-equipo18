"""API de pronóstico de demanda de MiBici (FastAPI).

La noche anterior, el operador pide la demanda de cada estación para cada hora del día siguiente y
con eso planea el rebalanceo. La API valida la petición, delega en mibici/serving.py y responde
cada pronóstico con su llave (estación y hora) y su fuente (modelo o fallback).

Correr en local:
    make serve        # uvicorn en http://127.0.0.1:8000 (documentación en /docs)

Endpoints:
    GET  /health    estado del servicio, versión del modelo y edad (para el SRE y el contenedor)
    GET  /modelo    metadatos de la versión servida (hiperparámetros y métricas)
    POST /predict   lote de hasta API_MAX_BATCH filas estación-hora
"""

from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from fastapi import FastAPI
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator

from mibici import config
from mibici.dataset import load_stations
from mibici.serving import EventLogger, ModelBundle, load_bundle, serve

LOCAL_TZ = ZoneInfo("America/Mexico_City")  # Guadalajara; los datos de MiBici vienen en hora local


class Fila(BaseModel):
    """Una estación en una hora. Los rezagos son las salidas observadas de la misma hora ayer y
    hace una semana: los conoce el operador la noche anterior."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "estacion_id": 2,
                "hora": "2026-09-15T08:00:00",
                "lag_24h": 3,
                "lag_168h": 4,
            }
        }
    )

    estacion_id: int = Field(gt=0, description="id del catálogo de estaciones de MiBici")
    hora: datetime = Field(description="inicio de la hora a pronosticar (hora local)")
    lag_24h: float | None = Field(None, ge=0, description="salidas a la misma hora, un día antes")
    lag_168h: float | None = Field(None, ge=0, description="salidas a la misma hora, 7 días antes")

    @field_validator("hora")
    @classmethod
    def hora_de_servicio(cls, value: datetime) -> datetime:
        if value.tzinfo is not None:  # se convierte a hora local y se trabaja sin zona, como en v4
            value = value.astimezone(LOCAL_TZ).replace(tzinfo=None)
        if value.minute or value.second or value.microsecond:
            raise ValueError("la hora debe ser exacta (minuto 0)")
        if value.hour in config.CLOSED_HOURS:
            raise ValueError("el sistema no opera de 1:00 a 4:59")
        return value


class Peticion(BaseModel):
    filas: list[Fila] = Field(min_length=1, max_length=config.API_MAX_BATCH)


class Pronostico(BaseModel):
    estacion_id: int
    hora: datetime
    salidas_pred: float = Field(description="salidas esperadas en esa hora")
    fuente: str = Field(description='"modelo" o "fallback" (persistencia semanal)')
    motivo: str | None = Field(None, description="por qué se usó el fallback")


class Respuesta(BaseModel):
    request_id: str
    modelo: str
    version: str | None
    pronosticos: list[Pronostico]


def create_app(
    model_dir: Path | None = None,
    stations: pd.DataFrame | None = None,
    log_path: Path | None = None,
) -> FastAPI:
    """Fábrica de la aplicación: las pruebas le pasan un modelo, un catálogo y un log temporales."""
    state: dict = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        state["bundle"] = load_bundle(model_dir or config.SERVING_MODEL_DIR)
        state["stations"] = stations if stations is not None else load_stations()
        state["events"] = EventLogger(log_path or config.PREDICTION_LOG)
        yield

    app = FastAPI(
        title="MiBici · demanda por estación y hora",
        description="Pronóstico de salidas por estación para el día siguiente (Equipo 18).",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict:
        bundle: ModelBundle = state["bundle"]
        age = bundle.age_days()
        alerts = []
        if bundle.model is None:
            alerts.append("modelo_no_disponible: se responde con la persistencia semanal")
        if age is not None and age > config.MODEL_MAX_AGE_DAYS:
            alerts.append(f"modelo_viejo: {age:.1f} días > {config.MODEL_MAX_AGE_DAYS}")
        return {
            "estado": "ok" if bundle.model is not None else "degradado",
            "modelo_cargado": bundle.model is not None,
            "version": bundle.version,
            "edad_modelo_dias": None if age is None else round(age, 2),
            "datos_entrenamiento_hasta": bundle.meta.get("datos_entrenamiento_hasta"),
            "alertas": alerts,
        }

    @app.get("/modelo")
    def modelo() -> dict:
        bundle: ModelBundle = state["bundle"]
        return {**bundle.meta, "cargado": bundle.model is not None, "error": bundle.error}

    @app.post("/predict", response_model=Respuesta)
    def predict(peticion: Peticion) -> Respuesta:
        bundle: ModelBundle = state["bundle"]
        requests = pd.DataFrame([f.model_dump() for f in peticion.filas])
        requests["hora"] = pd.to_datetime(requests["hora"])
        requests[["lag_24h", "lag_168h"]] = requests[["lag_24h", "lag_168h"]].astype(float)
        request_id, rows = serve(bundle, requests, state["stations"], state["events"])
        rows = rows.astype(object).where(rows.notna(), None)  # NaN -> null en el JSON
        return Respuesta(
            request_id=request_id,
            modelo=bundle.meta.get("nombre", config.REGISTERED_MODEL),
            version=bundle.version,
            pronosticos=rows.to_dict("records"),
        )

    return app


app = create_app()
