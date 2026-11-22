"""Prueba de integración de punta a punta, con datos sintéticos en el formato del portal:

CSV crudos -> limpieza (v1) -> panel estación-hora y variables (v2) -> partición temporal (v4)
-> Pipeline entrenado -> pronóstico -> métricas -> modelo guardado -> API -> mismos pronósticos.

Usa las mismas funciones que `make pipeline` y `make serve`; sólo cambia el origen de los datos.
"""

from fastapi.testclient import TestClient
import mlflow
import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici import config
from mibici.api import create_app
from mibici.dataset import TripCleaner, load_raw, load_stations
from mibici.features import add_features, build_demand_panel, make_model_input
from mibici.modeling.pipeline import build_pipeline, input_columns
from mibici.modeling.train import evaluate

STATIONS = {2: "GDL-001", 3: "GDL-002", 4: "ZPN-001", 5: "TLQ-001"}
PEAKS = {2: 8, 3: 18, 4: 13, 5: 9}


@pytest.fixture(scope="module")
def portal_files(tmp_path_factory):
    """Viajes de jun a sep 2026 en CSV mensuales (como datos_abiertos_AAAA_MM.csv) y catálogo."""
    root = tmp_path_factory.mktemp("portal")
    raw_dir = root / "raw"
    raw_dir.mkdir()
    rng = np.random.default_rng(config.SEED)
    hours = pd.date_range("2026-06-01 05:00", "2026-09-30 23:00", freq="h")
    hours = hours[~hours.hour.isin(config.CLOSED_HOURS) & (hours.hour > 0)]
    trips, trip_id = [], 0
    for sid, peak in PEAKS.items():
        lam = 0.5 + 4 * np.exp(-((hours.hour - peak) ** 2) / 6) * (1 + 0.3 * (hours.month == 9))
        for start, n in zip(hours, rng.poisson(lam)):
            for _ in range(n):
                trip_id += 1
                begin = start + pd.Timedelta(minutes=int(rng.integers(0, 50)))
                end = begin + pd.Timedelta(minutes=int(rng.integers(5, 40)))
                trips.append([trip_id, 10, "M", 1990, begin.strftime("%Y-%m-%d %H:%M:%S"),
                              end.strftime("%Y-%m-%d %H:%M:%S"), sid, 2 if sid != 2 else 3])  # fmt: skip
    df = pd.DataFrame(trips, columns=["Viaje_Id", "Usuario_Id", "Genero", "Anio_de_nacimiento",
                                      "Inicio_del_viaje", "Fin_del_viaje", "Origen_Id",
                                      "Destino_Id"])  # fmt: skip
    month = pd.to_datetime(df["Inicio_del_viaje"]).dt.strftime("%Y_%m")
    for m, part in df.groupby(month):
        part.to_csv(raw_dir / f"datos_abiertos_{m}.csv", index=False)
    stations = pd.DataFrame(
        {
            "id": list(STATIONS),
            "name": [f"({code}) Estación de prueba" for code in STATIONS.values()],
            "latitude": [20.66, 20.67, 20.71, 20.64],
            "longitude": [-103.35, -103.36, -103.40, -103.31],
            "dpcapacity": [15, 20, 12, 18],
        }
    )
    stations_path = root / "nomenclatura.csv"
    stations.to_csv(stations_path, index=False)
    return raw_dir, stations_path


@pytest.fixture(scope="module")
def pipeline_run(portal_files):
    raw_dir, stations_path = portal_files
    raw = load_raw(raw_dir)
    stations = load_stations(stations_path)
    trips = TripCleaner(stations["estacion_id"]).clean(raw)
    features = add_features(build_demand_panel(trips), stations)
    model_input = make_model_input(features)
    parts = {p: model_input[model_input["particion"] == p] for p in ("train", "valid", "test")}
    model = build_pipeline(HistGradientBoostingRegressor(loss="poisson", max_iter=60))
    model.fit(parts["train"][input_columns()], parts["train"]["salidas"])
    return {"raw": raw, "trips": trips, "features": features, "parts": parts,
            "model": model, "stations": stations}  # fmt: skip


def test_datos_conservan_todos_los_viajes(pipeline_run):
    """Cada viaje limpio es una salida en el panel, y las tres particiones existen en orden."""
    run = pipeline_run
    assert len(run["trips"]) == len(run["raw"])  # los datos sintéticos son todos válidos
    assert run["features"]["salidas"].sum() == len(run["trips"])
    parts = run["parts"]
    assert parts["train"]["hora"].max() < parts["valid"]["hora"].min()
    assert parts["valid"]["hora"].max() < parts["test"]["hora"].min()


def test_modelo_pronostica_y_le_gana_a_la_persistencia(pipeline_run):
    test = pipeline_run["parts"]["test"]
    y_pred = pipeline_run["model"].predict(test[input_columns()])
    metrics = evaluate(test["salidas"], y_pred)
    persistence = evaluate(test["salidas"], test["lag_168h"])
    assert all(np.isfinite(v) for v in metrics.values())
    assert (y_pred >= 0).all()
    assert metrics["mae"] < persistence["mae"]


def test_api_sirve_el_mismo_modelo_de_punta_a_punta(tmp_path, pipeline_run):
    """El modelo guardado y servido por la API da los mismos números que el entrenado."""
    model, test = pipeline_run["model"], pipeline_run["parts"]["test"].head(50)
    model_dir = tmp_path / "champion"
    mlflow.sklearn.save_model(model, str(model_dir), serialization_format="cloudpickle")
    rows = test[["estacion_id", "hora", "lag_24h", "lag_168h"]]
    body = {"filas": rows.assign(hora=rows["hora"].astype(str)).to_dict("records")}
    app = create_app(
        model_dir=model_dir, stations=pipeline_run["stations"], log_path=tmp_path / "log.jsonl"
    )
    with TestClient(app) as client:
        served = client.post("/predict", json=body).json()["pronosticos"]
    expected = model.predict(test[input_columns()])
    np.testing.assert_allclose([p["salidas_pred"] for p in served], expected, rtol=1e-9)
    assert {p["fuente"] for p in served} == {"modelo"}
