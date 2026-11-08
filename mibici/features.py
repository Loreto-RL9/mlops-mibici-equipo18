"""Del viaje a la demanda: panel estación-hora con sus variables.

El escalado, la codificación y el PCA ya no se aplican aquí: viven en el Pipeline de
scikit-learn (mibici/modeling/pipeline.py), que los ajusta solo con entrenamiento.

Comando:
    python -m mibici.features   # v2 (panel con variables) y v4 (v2 con partición temporal)
"""

from pathlib import Path

import holidays
from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici import config
from mibici.dataset import load_stations
from mibici.tracking import log_dataset_version

app = typer.Typer()

LAGS = [1, 24, 168]  # hora anterior, mismo momento ayer, mismo momento la semana pasada


def build_demand_panel(trips: pd.DataFrame) -> pd.DataFrame:
    """Salidas por estación y hora. Una hora sin viajes cuenta como 0, pero solo dentro
    del periodo en que la estación estuvo activa (de su primera a su última salida)."""
    hora = trips["inicio"].dt.floor("h").rename("hora")
    counts = trips.groupby([trips["origen_id"].rename("estacion_id"), hora]).size()
    stations = counts.index.get_level_values("estacion_id").unique()
    hours = pd.date_range(hora.min(), hora.max(), freq="h")
    grid = pd.MultiIndex.from_product([stations, hours], names=["estacion_id", "hora"])
    panel = counts.reindex(grid, fill_value=0).rename("salidas").reset_index()

    window = counts.reset_index().groupby("estacion_id")["hora"].agg(["min", "max"])
    panel = panel.join(window, on="estacion_id")
    active = panel["hora"].between(panel["min"], panel["max"])
    return panel.loc[active, ["estacion_id", "hora", "salidas"]].reset_index(drop=True)


def add_features(panel: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Calendario, festivos, datos de la estación y rezagos de la demanda."""
    cols = ["estacion_id", "capacidad", "lat", "lon", "municipio"]
    df = panel.merge(stations[cols], on="estacion_id", how="left")
    df = df.sort_values(["estacion_id", "hora"]).reset_index(drop=True)

    h = df["hora"]
    df["hora_dia"] = h.dt.hour
    df["dia_semana"] = h.dt.day_of_week
    df["fin_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    mx_holidays = pd.to_datetime(list(holidays.Mexico(years=h.dt.year.unique().tolist())))
    df["festivo"] = h.dt.normalize().isin(mx_holidays).astype(int)

    # Rezagos por estación (el panel es continuo hora a hora dentro de cada estación)
    by_station = df.groupby("estacion_id")["salidas"]
    for k in LAGS:
        df[f"lag_{k}h"] = by_station.shift(k)
    df["media_24h"] = by_station.transform(lambda s: s.shift(1).rolling(24).mean())
    # Las horas sin servicio son ceros seguros: se quitan después de calcular los rezagos
    return df[~df["hora_dia"].isin(config.CLOSED_HOURS)].reset_index(drop=True)


def assign_split(hora: pd.Series) -> pd.Series:
    """Partición temporal: el pasado entrena y el futuro evalúa (sin fuga de información)."""
    labels = np.select(
        [hora <= config.TRAIN_END, hora <= config.VALID_END], ["train", "valid"], "test"
    )
    return pd.Series(labels, index=hora.index)


def make_model_input(features: pd.DataFrame) -> pd.DataFrame:
    """v4: filas con todos sus rezagos (la primera semana de cada estación no los tiene)
    y la partición temporal. Los valores no se transforman."""
    df = features.dropna(subset=[f"lag_{k}h" for k in LAGS] + ["media_24h"]).copy()
    df["particion"] = assign_split(df["hora"])
    return df.reset_index(drop=True)


@app.command()
def main(
    trips_path: Path = config.INTERIM_DATA_DIR / "trips_clean.parquet",
    out_dir: Path = config.PROCESSED_DATA_DIR,
):
    trips = pd.read_parquet(trips_path)
    stations = load_stations()

    features = add_features(build_demand_panel(trips), stations)
    assert features["salidas"].sum() == len(trips), "el panel no conserva todos los viajes"
    assert not features.duplicated(["estacion_id", "hora"]).any(), "estación-hora duplicada"
    panel_path = out_dir / "demand_panel.parquet"
    features.to_parquet(panel_path, index=False)
    log_dataset_version(
        features, panel_path, name="demanda", version="v2", parent="viajes-v1",
        description="Panel estación-hora con ceros en horas activas sin viajes, "
                    "calendario, festivos y rezagos",
        params={"rezagos_h": str(LAGS), "ventana_media_h": 24},
        metrics={"estaciones": features["estacion_id"].nunique(),
                 "pct_horas_sin_salidas": round(100 * (features["salidas"] == 0).mean(), 2),
                 "salidas_media_hora": round(features["salidas"].mean(), 4)},
    )  # fmt: skip

    model_input = make_model_input(features)
    input_path = out_dir / "model_input.parquet"
    model_input.to_parquet(input_path, index=False)
    split_sizes = model_input["particion"].value_counts()
    log_dataset_version(
        model_input, input_path, name="demanda", version="v4", parent="demanda-v2",
        description="Filas con rezagos completos y partición temporal; el preprocesamiento "
                    "se aplica dentro del Pipeline de scikit-learn",
        params={"train_hasta": config.TRAIN_END, "valid_hasta": config.VALID_END},
        metrics={f"filas_{k}": int(v) for k, v in split_sizes.items()},
    )  # fmt: skip
    logger.success(f"v2: {len(features):,} filas | v4: {len(model_input):,} filas -> {out_dir}")


if __name__ == "__main__":
    app()
