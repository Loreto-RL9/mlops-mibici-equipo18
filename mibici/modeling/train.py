"""Línea base: comprueba que el dataset v3 ya sirve para modelar (punto de partida de la Fase 2).

Compara en la partición de validación (jul-ago 2026):
  * persistencia semanal: misma hora de la semana pasada (lag de 168 h)
  * media de las últimas 24 h
  * HistGradientBoostingRegressor con pérdida Poisson (la demanda es un conteo)
Cada modelo queda como un run del experimento `mibici-fase1-baseline`.

Comando:
    python -m mibici.modeling.train
"""

from loguru import logger
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
import typer

from mibici import config

app = typer.Typer()
ID_COLS = ["estacion_id", "hora", "particion", "salidas"]


def evaluate(y_true, y_pred) -> dict:
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred)}  # fmt: skip


@app.command()
def main(max_iter: int = 200, learning_rate: float = 0.1):
    ready = pd.read_parquet(config.PROCESSED_DATA_DIR / "model_ready.parquet")
    panel = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "demand_panel.parquet",
        columns=["estacion_id", "hora", "lag_168h", "media_24h"],
    )
    valid = ready[ready["particion"] == "valid"].merge(panel, on=["estacion_id", "hora"],
                                                       suffixes=("", "_orig"))  # fmt: skip
    train = ready[ready["particion"] == "train"]
    features = [c for c in ready.columns if c not in ID_COLS]

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("mibici-fase1-baseline")
    results = {}
    for name, pred in [("persistencia_semanal", valid["lag_168h_orig"]),
                       ("media_24h", valid["media_24h_orig"])]:  # fmt: skip
        with mlflow.start_run(run_name=name):
            results[name] = evaluate(valid["salidas"], pred)
            mlflow.log_params({"modelo": name, "dataset": "demanda-v3"})
            mlflow.log_metrics(results[name])

    with mlflow.start_run(run_name="hgb_poisson"):
        model = HistGradientBoostingRegressor(
            loss="poisson", max_iter=max_iter, learning_rate=learning_rate, random_state=42
        )
        model.fit(train[features], train["salidas"])
        results["hgb_poisson"] = evaluate(valid["salidas"], model.predict(valid[features]))
        mlflow.log_params({"modelo": "HistGradientBoostingRegressor", "loss": "poisson",
                           "max_iter": max_iter, "learning_rate": learning_rate,
                           "dataset": "demanda-v3", "n_features": len(features)})  # fmt: skip
        mlflow.log_metrics(results["hgb_poisson"])
        mlflow.log_text("\n".join(features), "features.txt")

    best = results["persistencia_semanal"]["mae"]
    for name, m in results.items():
        gain = 100 * (1 - m["mae"] / best)
        logger.info(
            f"{name:>22}: MAE={m['mae']:.3f} RMSE={m['rmse']:.3f} ({gain:+.1f} % vs lag 168h)"
        )
    assert np.isfinite(results["hgb_poisson"]["mae"]), "el modelo no produjo predicciones válidas"


if __name__ == "__main__":
    app()
