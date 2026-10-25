"""Del viaje a la demanda: panel estación x hora, variables y preprocesamiento.

Comando:
    python -m mibici.features   # v2 (panel con variables) y v3 (listo para modelar)
"""

from pathlib import Path

import holidays
import joblib
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import typer

from mibici import config
from mibici.dataset import load_stations
from mibici.tracking import log_dataset_version

app = typer.Typer()

LAGS = [1, 24, 168]  # hora anterior, mismo momento ayer, mismo momento la semana pasada
BASE_SCALED = ["capacidad", "lat", "lon", "hora_sin", "hora_cos", "dia_sin", "dia_cos",
               *[f"lag_{k}h" for k in LAGS], "media_24h"]  # fmt: skip


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


def cyclic(values: pd.Series, period: int):
    angle = 2 * np.pi * values / period
    return np.sin(angle), np.cos(angle)


def add_features(panel: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """Calendario, codificación cíclica, datos de la estación y rezagos de la demanda."""
    cols = ["estacion_id", "capacidad", "lat", "lon", "municipio"]
    df = panel.merge(stations[cols], on="estacion_id", how="left")
    df = df.sort_values(["estacion_id", "hora"]).reset_index(drop=True)

    h = df["hora"]
    df["hora_dia"] = h.dt.hour
    df["dia_semana"] = h.dt.dayofweek
    df["fin_de_semana"] = (df["dia_semana"] >= 5).astype(int)
    mx_holidays = pd.to_datetime(list(holidays.Mexico(years=h.dt.year.unique().tolist())))
    df["festivo"] = h.dt.normalize().isin(mx_holidays).astype(int)
    # Codificación cíclica: las 23 h quedan junto a las 0 h y el domingo junto al lunes
    df["hora_sin"], df["hora_cos"] = cyclic(h.dt.hour, 24)
    df["dia_sin"], df["dia_cos"] = cyclic(h.dt.dayofweek, 7)

    # Rezagos por estación (el panel es continuo hora a hora dentro de cada estación)
    by_station = df.groupby("estacion_id")["salidas"]
    for k in LAGS:
        df[f"lag_{k}h"] = by_station.shift(k)
    df["media_24h"] = by_station.transform(lambda s: s.shift(1).rolling(24).mean())
    return df


def station_profiles(panel: pd.DataFrame, end: str = config.TRAIN_END) -> pd.DataFrame:
    """Perfil horario de cada estación con datos de entrenamiento: 48 valores (24 h entre
    semana + 24 h en fin de semana). Se divide entre la media de la estación para comparar
    la forma de la demanda y no su volumen."""
    train = panel[panel["hora"] <= end]
    keys = [train["estacion_id"], train["hora"].dt.dayofweek >= 5, train["hora"].dt.hour]
    prof = train.groupby(keys)["salidas"].mean().unstack([1, 2])
    return prof.div(prof.mean(axis=1), axis=0).fillna(0)


def fit_station_pca(profiles: pd.DataFrame, variance: float = config.PCA_VARIANCE):
    """Reduce los 48 valores del perfil a los componentes que explican `variance`."""
    pca = PCA(n_components=variance, svd_solver="full", random_state=0).fit(profiles)
    cols = [f"pc{i + 1}" for i in range(pca.n_components_)]
    comps = pd.DataFrame(pca.transform(profiles), index=profiles.index, columns=cols)
    return pca, comps


def assign_split(hora: pd.Series) -> pd.Series:
    """Partición temporal: el pasado entrena y el futuro evalúa (sin fuga de información)."""
    labels = np.select(
        [hora <= config.TRAIN_END, hora <= config.VALID_END], ["train", "valid"], "test"
    )
    return pd.Series(labels, index=hora.index)


def make_model_ready(features: pd.DataFrame, comps: pd.DataFrame):
    """Une los componentes PCA, codifica el municipio y escala con estadísticas de train."""
    pc_cols = list(comps.columns)
    scaled_cols = BASE_SCALED + pc_cols
    df = features.join(comps, on="estacion_id")
    df[pc_cols] = df[pc_cols].fillna(0)  # estación sin historia en train = perfil promedio
    df = df.dropna(subset=[f"lag_{k}h" for k in LAGS] + ["media_24h"])
    df["particion"] = assign_split(df["hora"])
    df = pd.get_dummies(df, columns=["municipio"], prefix="mun", dtype=int)

    scaler = StandardScaler().fit(df.loc[df["particion"] == "train", scaled_cols])
    df[scaled_cols] = scaler.transform(df[scaled_cols])
    keep = ["estacion_id", "hora", "particion", "salidas", "fin_de_semana", "festivo",
            *scaled_cols, *[c for c in df.columns if c.startswith("mun_")]]  # fmt: skip
    return df[keep].reset_index(drop=True), scaler


@app.command()
def main(
    trips_path: Path = config.INTERIM_DATA_DIR / "trips_clean.parquet",
    out_dir: Path = config.PROCESSED_DATA_DIR,
):
    trips = pd.read_parquet(trips_path)
    stations = load_stations()

    features = add_features(build_demand_panel(trips), stations)
    assert features["salidas"].sum() == len(trips), "el panel no conserva todos los viajes"
    panel_path = out_dir / "demand_panel.parquet"
    features.to_parquet(panel_path, index=False)
    log_dataset_version(
        features, panel_path, name="demanda", version="v2", parent="viajes-v1",
        description="Panel estación x hora con ceros en horas activas sin viajes, "
                    "calendario, festivos, codificación cíclica y rezagos",
        params={"rezagos_h": str(LAGS), "ventana_media_h": 24},
        metrics={"estaciones": features["estacion_id"].nunique(),
                 "pct_horas_sin_salidas": round(100 * (features["salidas"] == 0).mean(), 2),
                 "salidas_media_hora": round(features["salidas"].mean(), 4)},
    )  # fmt: skip

    profiles = station_profiles(features)
    pca, comps = fit_station_pca(profiles)
    model_ready, scaler = make_model_ready(features, comps)
    joblib.dump(scaler, config.MODELS_DIR / "scaler.joblib")
    joblib.dump(pca, config.MODELS_DIR / "pca_perfiles.joblib")
    pca_table = config.MODELS_DIR / "pca_componentes_por_estacion.csv"
    comps.round(4).to_csv(pca_table)
    ready_path = out_dir / "model_ready.parquet"
    model_ready.to_parquet(ready_path, index=False)
    split_sizes = model_ready["particion"].value_counts()
    log_dataset_version(
        model_ready, ready_path, name="demanda", version="v3", parent="demanda-v2",
        description="One-hot de municipio, PCA de perfiles semanales, StandardScaler "
                    "ajustado solo con train y partición temporal",
        params={"pca_varianza_objetivo": config.PCA_VARIANCE, "train_hasta": config.TRAIN_END,
                "valid_hasta": config.VALID_END, "perfil_dimensiones": profiles.shape[1]},
        metrics={"pca_componentes": pca.n_components_,
                 "pca_var_total": round(pca.explained_variance_ratio_.sum(), 4),
                 "pca_var_pc1": round(pca.explained_variance_ratio_[0], 4),
                 **{f"filas_{k}": int(v) for k, v in split_sizes.items()}},
        artifacts=[config.MODELS_DIR / "scaler.joblib", config.MODELS_DIR / "pca_perfiles.joblib",
                   pca_table],
    )  # fmt: skip
    logger.success(f"v2: {len(features):,} filas | v3: {len(model_ready):,} filas -> {out_dir}")


if __name__ == "__main__":
    app()
