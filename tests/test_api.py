"""Pruebas de la API por capa (con datos sintéticos; no necesitan MLflow ni descargas).

1. Variables de servicio: el calendario y el catálogo se arman igual que en el entrenamiento.
2. Capa de servicio: pronóstico con el modelo, fallback y evento en el log.
3. HTTP: /health, /modelo y /predict con TestClient, incluidas las entradas inválidas (422).
"""

from datetime import datetime, timedelta, timezone
import json

from fastapi.testclient import TestClient
import mlflow
import pandas as pd
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor

from mibici.api import create_app
from mibici.features import add_features
from mibici.modeling.pipeline import build_pipeline, input_columns
from mibici.serving import EventLogger, ModelBundle, build_features, load_bundle, predict, serve


@pytest.fixture
def stations(data):
    return (
        data[["estacion_id", "capacidad", "lat", "lon", "municipio"]]
        .drop_duplicates("estacion_id")
        .reset_index(drop=True)
    )


@pytest.fixture
def model_dir(tmp_path, parts):
    """Un Pipeline pequeño guardado como lo deja `make export-model`."""
    model = build_pipeline(HistGradientBoostingRegressor(loss="poisson", max_iter=30))
    model.fit(parts["train"][input_columns()], parts["train"]["salidas"])
    path = tmp_path / "champion"
    mlflow.sklearn.save_model(model, str(path), serialization_format="cloudpickle")
    registered = datetime.now(timezone.utc) - timedelta(days=2)
    meta = {"nombre": "mibici-prueba", "alias": "champion", "version": "7",
            "registrado": registered.isoformat(timespec="seconds"),
            "datos_entrenamiento_hasta": "2026-03-11 23:00"}  # fmt: skip
    (path / "model_meta.json").write_text(json.dumps(meta))
    return path


@pytest.fixture
def requests_df(parts):
    rows = parts["test"].head(6)
    return rows[["estacion_id", "hora", "lag_24h", "lag_168h"]].astype(
        {"lag_24h": float, "lag_168h": float}
    )


def client_for(model_dir, stations, log_path) -> TestClient:
    return TestClient(create_app(model_dir=model_dir, stations=stations, log_path=log_path))


# 1. Variables de servicio ------------------------------------------------------------------------
def test_variables_de_servicio_iguales_a_las_de_entrenamiento(data, stations, requests_df):
    """Sin training-serving skew: lo que arma la API coincide con lo que arma features.py."""
    panel = data[["estacion_id", "hora", "salidas"]]
    trained = add_features(panel, stations).set_index(["estacion_id", "hora"])
    served = build_features(requests_df, stations).set_index(["estacion_id", "hora"])
    cols = ["capacidad", "lat", "lon", "municipio", "hora_dia", "dia_semana", "fin_de_semana",
            "festivo"]  # fmt: skip
    pd.testing.assert_frame_equal(served[cols], trained.loc[served.index, cols])


# 2. Capa de servicio -----------------------------------------------------------------------------
def test_pronostico_con_modelo(model_dir, stations, requests_df):
    bundle = load_bundle(model_dir)
    out = predict(bundle, build_features(requests_df, stations))
    assert (out["fuente"] == "modelo").all()
    assert (out["salidas_pred"] >= 0).all()


def test_fallback_sin_modelo(tmp_path, stations, requests_df):
    """Si el modelo no carga, responde con la persistencia semanal y lo dice."""
    bundle = load_bundle(tmp_path / "no-existe")
    assert bundle.model is None
    out = predict(bundle, build_features(requests_df, stations))
    assert (out["fuente"] == "fallback").all()
    assert (out["motivo"] == "modelo_no_disponible").all()
    assert out["salidas_pred"].tolist() == requests_df["lag_168h"].tolist()


def test_fallback_estacion_desconocida_y_sin_rezagos(model_dir, stations, requests_df):
    df = requests_df.copy().reset_index(drop=True)
    df.loc[0, "estacion_id"] = 9999
    df.loc[1, ["lag_24h", "lag_168h"]] = [2.0, None]
    out = predict(load_bundle(model_dir), build_features(df, stations))
    assert out.loc[0, "motivo"] == "estacion_desconocida"
    assert out.loc[1, "motivo"] == "sin_rezagos" and out.loc[1, "salidas_pred"] == 2.0
    assert (out.loc[2:, "fuente"] == "modelo").all()


def test_evento_en_el_log(tmp_path, model_dir, stations, requests_df):
    log = EventLogger(tmp_path / "eventos.jsonl")
    request_id, rows = serve(load_bundle(model_dir), requests_df, stations, log)
    events = pd.read_json(log.path, lines=True)
    assert len(events) == 1 and events.loc[0, "request_id"] == request_id
    assert events.loc[0, "version"] == 7 and events.loc[0, "filas"] == len(rows)
    assert events.loc[0, "latencia_ms"] >= 0


# 3. HTTP -----------------------------------------------------------------------------------------
def test_health_y_modelo(tmp_path, model_dir, stations):
    with client_for(model_dir, stations, tmp_path / "log.jsonl") as client:
        health = client.get("/health").json()
        assert health["estado"] == "ok" and health["version"] == "7"
        assert 1.9 < health["edad_modelo_dias"] < 2.1 and health["alertas"] == []
        assert client.get("/modelo").json()["cargado"] is True


def test_health_degradado(tmp_path, stations):
    with client_for(tmp_path / "no-existe", stations, tmp_path / "log.jsonl") as client:
        health = client.get("/health").json()
        assert health["estado"] == "degradado" and health["alertas"]


def test_predict_devuelve_cada_pronostico_con_su_llave(tmp_path, model_dir, stations, requests_df):
    body = {"filas": requests_df.assign(hora=requests_df["hora"].astype(str)).to_dict("records")}
    with client_for(model_dir, stations, tmp_path / "log.jsonl") as client:
        response = client.post("/predict", json=body)
    assert response.status_code == 200
    payload = response.json()
    keys = [(p["estacion_id"], p["hora"]) for p in payload["pronosticos"]]
    expected = [(r["estacion_id"], pd.Timestamp(r["hora"]).isoformat()) for r in body["filas"]]
    assert keys == expected
    assert payload["version"] == "7"
    assert {p["fuente"] for p in payload["pronosticos"]} == {"modelo"}


@pytest.mark.parametrize(
    "fila",
    [
        {"estacion_id": 1, "hora": "no-es-fecha", "lag_24h": 1, "lag_168h": 1},
        {"estacion_id": 1, "hora": "2026-03-16T08:30:00", "lag_24h": 1, "lag_168h": 1},
        {"estacion_id": 1, "hora": "2026-03-16T03:00:00", "lag_24h": 1, "lag_168h": 1},
        {"estacion_id": 0, "hora": "2026-03-16T08:00:00", "lag_24h": 1, "lag_168h": 1},
        {"estacion_id": 1, "hora": "2026-03-16T08:00:00", "lag_24h": -1, "lag_168h": 1},
    ],
    ids=["fecha-invalida", "no-exacta", "hora-sin-servicio", "estacion-0", "rezago-negativo"],
)
def test_entradas_invalidas_dan_422(tmp_path, model_dir, stations, fila):
    with client_for(model_dir, stations, tmp_path / "log.jsonl") as client:
        assert client.post("/predict", json={"filas": [fila]}).status_code == 422


def test_lote_vacio_da_422(tmp_path, model_dir, stations):
    with client_for(model_dir, stations, tmp_path / "log.jsonl") as client:
        assert client.post("/predict", json={"filas": []}).status_code == 422


def test_hora_con_zona_se_convierte_a_hora_local(tmp_path, model_dir, stations):
    fila = {"estacion_id": 1, "hora": "2026-03-16T14:00:00Z", "lag_24h": 1, "lag_168h": 1}
    with client_for(model_dir, stations, tmp_path / "log.jsonl") as client:
        payload = client.post("/predict", json={"filas": [fila]}).json()
    assert payload["pronosticos"][0]["hora"] == "2026-03-16T08:00:00"  # UTC-6


def test_bundle_sin_modelo_no_rompe_la_edad():
    assert ModelBundle(None).age_days() is None


def test_arranque_en_frio_estacion_que_el_modelo_no_vio(model_dir, stations, requests_df):
    """Una estación del catálogo que no estuvo en el entrenamiento recibe la persistencia."""
    catalog = pd.concat(
        [stations, stations.head(1).assign(estacion_id=4, municipio="TLQ")], ignore_index=True
    )
    df = requests_df.head(2).reset_index(drop=True).assign(estacion_id=[4, 1])
    out = predict(load_bundle(model_dir), build_features(df, catalog))
    assert out.loc[0, "motivo"] == "estacion_nueva" and out.loc[0, "fuente"] == "fallback"
    assert out.loc[0, "salidas_pred"] == df.loc[0, "lag_168h"]
    assert out.loc[1, "fuente"] == "modelo"
