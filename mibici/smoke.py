"""Prueba de humo y de portabilidad del servicio (contenedor o `make serve`).

El *golden set* es un lote fijo de filas estación-hora de septiembre 2026 con los pronósticos que
dio el campeón al exportarse. Si la API, en cualquier máquina o sistema operativo, devuelve esos
mismos números, el servicio es portable; si cambian sin que cambie el modelo, algo se rompió
(entorno, versiones o variables de servicio).

Comandos:
    python -m mibici.smoke golden                     # genera reports/golden_set.json (requiere
                                                      # data/processed y models/champion)
    python -m mibici.smoke check --url http://127.0.0.1:8000   # compara la API contra el golden
    python -m mibici.smoke basico --url http://127.0.0.1:8000  # sólo /health y un /predict
    python -m mibici.smoke modelo-sintetico           # modelo pequeño con datos sintéticos (CI)
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from loguru import logger
import numpy as np
import pandas as pd
import typer

from mibici import config

app = typer.Typer()

GOLDEN_PATH = config.REPORTS_DIR / "golden_set.json"
TOLERANCE = 1e-6


def _post(url: str, payload: dict, timeout: float = 60) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read())


def _get(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.loads(response.read())


def wait_until_healthy(url: str, seconds: int = 90) -> dict:
    """Espera a que /health responda (el contenedor tarda unos segundos en cargar el modelo)."""
    deadline = time.time() + seconds
    while True:
        try:
            return _get(f"{url}/health")
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            if time.time() > deadline:
                raise
            time.sleep(2)


@app.command()
def golden(n_stations: int = 6, seed: int = config.SEED, out: Path = GOLDEN_PATH):
    """Lote fijo: n estaciones (de alta a baja demanda) x las 20 horas de servicio de un martes
    de septiembre, más 2 filas que deben caer en fallback (estación desconocida y sin rezagos)."""
    from mibici.serving import build_features, load_bundle, predict

    data = pd.read_parquet(config.PROCESSED_DATA_DIR / "model_input.parquet")
    test = data[data["particion"] == "test"]
    day = test[test["hora"].dt.normalize() == "2026-09-15"]
    ranking = day.groupby("estacion_id")["salidas"].sum().sort_values(ascending=False).index
    picks = ranking[np.linspace(0, len(ranking) - 1, n_stations).round().astype(int)]
    rows = day[day["estacion_id"].isin(picks)][["estacion_id", "hora", "lag_24h", "lag_168h"]]
    rows = rows.sort_values(["estacion_id", "hora"]).reset_index(drop=True)
    extra = pd.DataFrame(
        {
            "estacion_id": [99999, int(picks[0])],
            "hora": pd.to_datetime(["2026-09-15 08:00", "2026-09-15 09:00"]),
            "lag_24h": [3.0, 2.0],
            "lag_168h": [4.0, np.nan],
        }
    )
    rows = pd.concat([rows, extra], ignore_index=True).astype(
        {"lag_24h": float, "lag_168h": float}
    )

    from mibici.dataset import load_stations

    bundle = load_bundle()
    assert bundle.model is not None, "falta models/champion: corre make export-model"
    result = predict(bundle, build_features(rows, load_stations()))
    golden_rows = rows.assign(hora=rows["hora"].dt.strftime("%Y-%m-%dT%H:%M:%S")).join(result)
    payload = {
        "modelo": bundle.meta.get("nombre"),
        "version": bundle.version,
        "run_id": bundle.meta.get("run_id"),
        "generado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "filas": json.loads(golden_rows.to_json(orient="records")),
    }
    out.write_text(json.dumps(payload, indent=1, ensure_ascii=False))
    logger.success(
        f"Golden set: {len(golden_rows)} filas del modelo versión {bundle.version} -> {out}"
    )


@app.command()
def check(url: str = "http://127.0.0.1:8000", golden_path: Path = GOLDEN_PATH):
    """Compara la API contra el golden set. Termina con error si algún pronóstico cambió."""
    health = wait_until_healthy(url)
    logger.info(f"/health: {health}")
    golden_set = json.loads(golden_path.read_text())
    expected = pd.DataFrame(golden_set["filas"])
    filas = [
        {k: (None if pd.isna(v) else v) for k, v in row.items()}
        for row in expected[["estacion_id", "hora", "lag_24h", "lag_168h"]].to_dict("records")
    ]
    response = _post(f"{url}/predict", {"filas": filas})
    got = pd.DataFrame(response["pronosticos"])
    diff = (got["salidas_pred"] - expected["salidas_pred"]).abs()
    same_source = (got["fuente"] == expected["fuente"]).all()
    logger.info(
        f"Versión servida {response['version']} vs golden {golden_set['version']}; "
        f"{len(got)} filas; diferencia máxima {diff.max():.2e}; fuentes iguales: {same_source}"
    )
    if diff.max() > TOLERANCE or not same_source:
        raise typer.Exit(code=1)
    logger.success("La API reproduce el golden set: el servicio es portable")


@app.command()
def basico(url: str = "http://127.0.0.1:8000"):
    """Prueba de humo sin golden set (CI): /health responde y /predict usa el modelo cargado."""
    health = wait_until_healthy(url)
    fila = {"estacion_id": 2, "hora": "2026-09-15T08:00:00", "lag_24h": 3, "lag_168h": 4}
    response = _post(f"{url}/predict", {"filas": [fila]})
    pronostico = response["pronosticos"][0]
    logger.info(f"/health: {health} | /predict: {pronostico}")
    if health["estado"] != "ok" or pronostico["fuente"] != "modelo":
        raise typer.Exit(code=1)
    logger.success("Servicio arriba: /health ok y /predict con el modelo")


@app.command("modelo-sintetico")
def synthetic_model(out_dir: Path = config.SERVING_MODEL_DIR):
    """Modelo pequeño entrenado con datos sintéticos, sólo para probar la imagen en el CI (el
    runner no tiene los datos ni el Model Registry). No sirve para pronosticar."""
    import mlflow
    from sklearn.ensemble import HistGradientBoostingRegressor

    from mibici.modeling.pipeline import build_pipeline, input_columns

    rng = np.random.default_rng(config.SEED)
    hours = pd.date_range("2026-03-02", periods=24 * 14, freq="h")
    hours = hours[~hours.hour.isin(config.CLOSED_HOURS)]
    stations = [(2, "GDL"), (3, "GDL"), (4, "ZPN")]
    frames = []
    for sid, municipio in stations:
        lam = 1 + 3 * np.exp(-((hours.hour - 8 - sid) ** 2) / 8)
        frames.append(
            pd.DataFrame(
                {
                    "estacion_id": sid,
                    "salidas": rng.poisson(lam),
                    "hora_dia": hours.hour,
                    "dia_semana": hours.day_of_week,
                    "fin_de_semana": (hours.day_of_week >= 5).astype(int),
                    "festivo": 0,
                    "capacidad": 15,
                    "lat": 20.66,
                    "lon": -103.35,
                    "municipio": municipio,
                    "lag_24h": rng.poisson(lam),
                    "lag_168h": rng.poisson(lam),
                }
            )
        )
    df = pd.concat(frames, ignore_index=True)
    model = build_pipeline(HistGradientBoostingRegressor(loss="poisson", max_iter=20))
    model.fit(df[input_columns()], df["salidas"])
    if out_dir.exists():
        import shutil

        shutil.rmtree(out_dir)
    mlflow.sklearn.save_model(model, str(out_dir), serialization_format="cloudpickle")
    meta = {
        "nombre": "modelo-sintetico-ci",
        "alias": "ci",
        "version": "0",
        "registrado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datos_entrenamiento_hasta": "sintético",
    }
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2))
    logger.success(f"Modelo sintético para el CI -> {out_dir}")


if __name__ == "__main__":
    app()
