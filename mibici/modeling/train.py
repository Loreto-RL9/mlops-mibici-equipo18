"""Línea base: comprueba que el dataset v3 ya sirve para modelar (punto de partida de la Fase 2).

Se evalúan dos horizontes en la partición de validación (jul-ago 2026):
  * 1 h: pronóstico para la hora siguiente; puede usar la demanda de la hora anterior.
  * 24 h: pronóstico hecho un día antes; solo usa rezagos de 24 h o más.
En cada horizonte se compara una persistencia ingenua con HistGradientBoostingRegressor
con pérdida Poisson (la demanda es un conteo). Cada modelo es un run del experimento
`mibici-fase1-baseline`.

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
# Variables que no se conocen al pronosticar con 24 h de anticipación
NOT_AVAILABLE_24H = ["lag_1h", "media_24h"]


def evaluate(y_true, y_pred) -> dict:
    return {"mae": mean_absolute_error(y_true, y_pred),
            "rmse": root_mean_squared_error(y_true, y_pred)}  # fmt: skip


@app.command()
def main(max_iter: int = 200, learning_rate: float = 0.1):
    ready = pd.read_parquet(config.PROCESSED_DATA_DIR / "model_ready.parquet")
    raw_lags = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "demand_panel.parquet",
        columns=["estacion_id", "hora", "lag_1h", "lag_24h", "lag_168h"],
    )
    valid = ready[ready["particion"] == "valid"].merge(raw_lags, on=["estacion_id", "hora"],
                                                       suffixes=("", "_orig"))  # fmt: skip
    train = ready[ready["particion"] == "train"]
    all_features = [c for c in ready.columns if c not in ID_COLS]

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment("mibici-fase1-baseline")
    results = {}
    naive = {"persistencia_1h": ("1h", "lag_1h_orig"), "persistencia_24h": ("24h", "lag_24h_orig"),
             "persistencia_semanal": ("24h", "lag_168h_orig")}  # fmt: skip
    for name, (horizon, col) in naive.items():
        with mlflow.start_run(run_name=name):
            results[name] = evaluate(valid["salidas"], valid[col])
            mlflow.log_params({"modelo": name, "horizonte": horizon, "dataset": "demanda-v3"})
            mlflow.log_metrics(results[name])

    for horizon, excluded in [("1h", []), ("24h", NOT_AVAILABLE_24H)]:
        features = [c for c in all_features if c not in excluded]
        name = f"hgb_poisson_{horizon}"
        with mlflow.start_run(run_name=name):
            model = HistGradientBoostingRegressor(
                loss="poisson", max_iter=max_iter, learning_rate=learning_rate, random_state=42
            )
            model.fit(train[features], train["salidas"])
            results[name] = evaluate(valid["salidas"], model.predict(valid[features]))
            mlflow.log_params({"modelo": "HistGradientBoostingRegressor", "loss": "poisson",
                               "horizonte": horizon, "max_iter": max_iter,
                               "learning_rate": learning_rate, "dataset": "demanda-v3",
                               "n_features": len(features)})  # fmt: skip
            mlflow.log_metrics(results[name])
            mlflow.log_text("\n".join(features), "features.txt")

    for name, m in results.items():
        logger.info(f"{name:>22}: MAE={m['mae']:.3f} RMSE={m['rmse']:.3f}")
    for horizon in ("1h", "24h"):  # se compara contra la mejor persistencia del mismo horizonte
        reference = min((n for n, (h, _) in naive.items() if h == horizon),
                        key=lambda n: results[n]["mae"])  # fmt: skip
        gain = 1 - results[f"hgb_poisson_{horizon}"]["mae"] / results[reference]["mae"]
        logger.info(
            f"Horizonte {horizon}: HGB mejora el MAE {100 * gain:.1f} % frente a {reference}"
        )
    assert all(np.isfinite(m["mae"]) for m in results.values()), "hay métricas no válidas"


if __name__ == "__main__":
    app()
