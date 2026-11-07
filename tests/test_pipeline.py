"""Pruebas del Pipeline de scikit-learn con datos sintéticos (no requieren descargar nada)."""

import numpy as np
import pandas as pd
import pytest

from mibici.modeling.pipeline import (
    StationProfilePCA,
    build_pipeline,
    build_preprocessor,
    cyclic_encode,
    evaluate,
    input_columns,
)


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


def test_cyclic_encoding_puts_23h_next_to_0h():
    encoded = cyclic_encode(pd.DataFrame({"hora_dia": [0, 23, 12], "dia_semana": [0, 6, 3]}))
    hour_0, hour_23, hour_12 = encoded[:, :2]
    assert np.linalg.norm(hour_0 - hour_23) < np.linalg.norm(hour_0 - hour_12)


def test_24h_horizon_does_not_use_recent_information():
    assert {"lag_1h", "media_24h"}.isdisjoint(input_columns("24h"))
    assert {"lag_1h", "media_24h"} <= set(input_columns("1h"))


def test_station_profile_pca_uses_train_only_and_handles_new_stations(data):
    encoder = StationProfilePCA(variance=0.9).fit(data, data["salidas"])
    new_station = data.head(2).assign(estacion_id=99)
    assert np.allclose(encoder.transform(new_station), 0)  # estación nueva = perfil promedio
    assert encoder.transform(data).shape == (len(data), encoder.n_components_)


def test_preprocessor_scales_with_train_statistics(data):
    X = data[input_columns("24h")]
    preprocessor = build_preprocessor("24h").fit(X, data["salidas"])
    out = pd.DataFrame(preprocessor.transform(X), columns=preprocessor.get_feature_names_out())
    assert np.allclose(out[["capacidad", "lag_24h", "lag_168h"]].mean(), 0, atol=1e-9)
    assert {"municipio_GDL", "hora_dia_sin", "pc1"} <= set(out.columns)


@pytest.mark.parametrize("model_name, params", [
    ("poisson_glm", {"alpha": 1e-3}),
    ("hgb_poisson", {"max_iter": 30, "learning_rate": 0.1}),
])  # fmt: skip
def test_pipelines_learn_better_than_the_mean(data, model_name, params):
    X, y = data[input_columns("24h")], data["salidas"]
    pred = build_pipeline(model_name, params, "24h").fit(X, y).predict(X)
    assert (pred >= 0).all()  # pérdida Poisson: nunca pronostica salidas negativas
    assert evaluate(y, pred)["mae"] < evaluate(y, np.full(len(y), y.mean()))["mae"]


def test_evaluate_reports_zero_error_for_a_perfect_forecast():
    metrics = evaluate([0, 2, 5], [0, 2, 5], top_mask=np.array([False, True, True]))
    assert metrics["mae"] == 0 and metrics["sesgo"] == 0 and metrics["mae_top20"] == 0
