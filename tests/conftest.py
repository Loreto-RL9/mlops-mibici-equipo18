"""Datos sintéticos compartidos por las pruebas del Pipeline y del Model Registry."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def data():
    """Tres estaciones con patrones distintos durante dos semanas de horas de servicio."""
    rng = np.random.default_rng(0)
    hours = pd.date_range("2026-03-02", periods=24 * 14, freq="h")
    hours = hours[~hours.hour.isin([1, 2, 3, 4])]
    rows = []
    for sid, peak, municipio in [(1, 8, "GDL"), (2, 18, "ZPN"), (3, 13, "TLQ")]:
        lam = 1 + 4 * np.exp(-((hours.hour - peak) ** 2) / 8)
        rows.append(pd.DataFrame({
            "estacion_id": sid, "hora": hours, "salidas": rng.poisson(lam),
            "hora_dia": hours.hour, "dia_semana": hours.day_of_week,
            "fin_de_semana": (hours.day_of_week >= 5).astype(int), "festivo": 0,
            "capacidad": 10 + sid, "lat": 20.6 + sid / 100, "lon": -103.3 - sid / 100,
            "municipio": municipio, "lag_1h": rng.poisson(lam), "lag_24h": rng.poisson(lam),
            "lag_168h": rng.poisson(lam), "media_24h": lam.to_numpy(),
        }))  # fmt: skip
    return pd.concat(rows, ignore_index=True)


@pytest.fixture
def parts(data):
    """Partición temporal como la real: entrenamiento, validación y prueba, en ese orden."""
    day = data["hora"].dt.normalize()
    return {
        "train": data[day < "2026-03-12"].reset_index(drop=True),
        "valid": data[day.between("2026-03-12", "2026-03-13")].reset_index(drop=True),
        "test": data[day >= "2026-03-14"].reset_index(drop=True),
    }
