"""Pruebas del pipeline de datos con datos sintéticos pequeños (no requieren descargar nada)."""

import numpy as np
import pandas as pd
import pytest

from mibici.dataset import TripCleaner, check_clean_trips
from mibici.features import (
    add_features,
    build_demand_panel,
    fit_station_pca,
    make_model_ready,
    station_profiles,
)

STATIONS = [1, 2, 3]


def raw_trip(viaje, inicio, fin, origen=1, destino=2, genero="M", anio="1990"):
    return {"viaje_id": str(viaje), "usuario_id": "10", "genero": genero,
            "anio_nacimiento": anio, "inicio": inicio, "fin": fin,
            "origen_id": str(origen), "destino_id": str(destino)}  # fmt: skip


@pytest.fixture
def raw():
    return pd.DataFrame([
        raw_trip(1, "2026-01-05 08:00:00", "2026-01-05 08:15:00"),  # válido
        raw_trip(1, "2026-01-05 08:00:00", "2026-01-05 08:15:00"),  # duplicado
        raw_trip(2, "2026-01-05 09:00:00", "2026-01-05 08:59:00"),  # termina antes de empezar
        raw_trip(3, "2026-01-05 09:00:00", "2026-01-05 09:01:00", 3, 3),  # arranque fallido
        raw_trip(4, "2026-01-05 10:00:00", "2026-01-05 10:00:30"),  # menos de 1 min
        raw_trip(5, "2026-01-05 11:00:00", "2026-01-05 15:00:00"),  # más de 180 min
        raw_trip(6, "2026-01-05 12:00:00", "2026-01-05 12:20:00", destino=99),  # no existe
        raw_trip(7, "2026-01-05 13:00:00", "2026-01-05 13:10:00", genero="NULL", anio="199"),
        raw_trip(8, "2026-01-05 03:10:00", "2026-01-05 03:25:00"),  # sistema cerrado
    ])  # fmt: skip


def stations_df(ids):
    return pd.DataFrame({"estacion_id": ids, "capacidad": 15, "lat": 20.67, "lon": -103.35,
                         "municipio": ["GDL", "ZPN"][: len(ids)]})  # fmt: skip


def test_cleaner_applies_each_rule(raw):
    cleaner = TripCleaner(STATIONS)
    clean = cleaner.clean(raw)
    assert clean["viaje_id"].tolist() == [1, 7]
    for rule in ["duplicados", "duracion_no_positiva", "arranque_fallido",
                 "duracion_menor_1min", "duracion_mayor_180min", "fuera_de_horario",
                 "estacion_fuera_de_catalogo"]:  # fmt: skip
        assert cleaner.report[rule] == 1, rule
    fixed = clean.set_index("viaje_id").loc[7]
    assert fixed["genero"] == "Desconocido"
    assert pd.isna(fixed["anio_nacimiento"])


def test_temporal_rule_is_enforced(raw):
    clean = TripCleaner(STATIONS).clean(raw)
    assert (clean["fin"] > clean["inicio"]).all()
    broken = clean.copy()
    broken.loc[0, "fin"] = broken.loc[0, "inicio"] - pd.Timedelta(minutes=5)
    with pytest.raises(AssertionError):
        check_clean_trips(broken, STATIONS)


def test_panel_fills_zeros_only_inside_active_window():
    trips = pd.DataFrame({
        "inicio": pd.to_datetime(["2026-01-05 08:10", "2026-01-05 08:40",
                                  "2026-01-05 11:05", "2026-01-05 09:30"]),
        "origen_id": [1, 1, 1, 2],
    })  # fmt: skip
    panel = build_demand_panel(trips)
    assert panel.loc[panel["estacion_id"] == 1, "salidas"].tolist() == [2, 0, 0, 1]  # 8h a 11h
    assert (panel["estacion_id"] == 2).sum() == 1  # la estación 2 solo estuvo activa a las 9h
    assert panel["salidas"].sum() == len(trips)
    assert not panel.duplicated(["estacion_id", "hora"]).any()


def test_lags_use_only_past_values():
    hours = pd.date_range("2026-01-05", periods=24 * 9, freq="h")
    panel = pd.DataFrame({"estacion_id": 1, "hora": hours, "salidas": np.arange(len(hours))})
    f = add_features(panel, stations_df([1]))
    last = f.iloc[-1]
    assert last["lag_1h"] == last["salidas"] - 1
    assert last["lag_24h"] == last["salidas"] - 24
    assert last["lag_168h"] == last["salidas"] - 168
    assert last["media_24h"] == last["salidas"] - 12.5  # media de las 24 horas previas
    assert not f["hora"].dt.hour.isin([1, 2, 3, 4]).any()  # horas sin servicio fuera del panel


def test_scaler_uses_train_statistics_only():
    rng = np.random.default_rng(0)
    hours = pd.date_range("2026-06-15", "2026-07-15 23:00", freq="h")
    panel = pd.concat([
        pd.DataFrame({"estacion_id": sid, "hora": hours, "salidas": rng.poisson(lam, len(hours))})
        for sid, lam in [(1, 2), (2, 5)]
    ])  # fmt: skip
    features = add_features(panel, stations_df([1, 2]))
    _, comps = fit_station_pca(station_profiles(features))
    ready, _ = make_model_ready(features, comps)

    train, valid = ready[ready["particion"] == "train"], ready[ready["particion"] == "valid"]
    assert train["hora"].max() < valid["hora"].min()
    assert np.allclose(train[["lag_24h", "media_24h", "hora_sin"]].mean(), 0, atol=1e-9)
    assert ready.notna().all().all()
