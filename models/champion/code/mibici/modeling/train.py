"""Entrenamiento final, evaluación y registro de un modelo (Fase 2).

Un modelo es un Pipeline de scikit-learn (mibici/modeling/pipeline.py): el preprocesamiento y el
estimador se ajustan juntos, solo con entrenamiento (oct 2025 a jun 2026), y se guardan juntos en
el MLflow Model Registry. La inferencia (predict.py) aplica así exactamente las mismas
transformaciones que el entrenamiento.

Este módulo reúne lo que comparten la búsqueda de hiperparámetros (tune.py) y la inferencia: las
familias de algoritmos, las métricas, la carga de particiones, el reentrenamiento con todo el
entrenamiento y el registro de la versión con sus hiperparámetros, métricas y gráficas.

Comando:
    python -m mibici.modeling.train --modelo hgb_poisson   # mejores hiperparámetros de make tune
"""

import ast

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error
import typer

from mibici import config
from mibici.modeling.pipeline import build_pipeline, input_columns
from mibici.plots import plot_error_by_hour, plot_forecast_week

app = typer.Typer()

# Familias de algoritmos (justificación en tune.py y reports/modelos_fase2.md). Cada una recibe
# sus hiperparámetros y devuelve el estimador que va al final del Pipeline.
MODELS = {
    "poisson_glm": lambda p: PoissonRegressor(max_iter=1000, **p),
    "random_forest": lambda p: RandomForestRegressor(
        max_samples=0.2, n_jobs=-1, random_state=config.SEED, **p
    ),
    "hgb_poisson": lambda p: HistGradientBoostingRegressor(
        loss="poisson", early_stopping=False, random_state=config.SEED, **p
    ),
}
PARTITIONS = ("valid", "test")


def evaluate(y_true, y_pred, top_mask=None) -> dict:
    """MAE y RMSE en bicis por hora, devianza de Poisson, sesgo (si es negativo, el modelo
    subestima) y MAE en las estaciones de mayor demanda, que son las que importan al rebalancear."""
    y_true, y_pred = np.asarray(y_true), np.clip(y_pred, 1e-6, None)  # la devianza exige > 0
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "devianza_poisson": mean_poisson_deviance(y_true, y_pred),
        "sesgo": float(np.mean(y_pred - y_true)),
    }
    if top_mask is not None:
        top_mask = np.asarray(top_mask)
        metrics["mae_top_estaciones"] = mean_absolute_error(y_true[top_mask], y_pred[top_mask])
    return metrics


def top_stations(train: pd.DataFrame, share: float = config.TOP_STATIONS_SHARE) -> set:
    """Estaciones con más salidas en entrenamiento (el 20 % concentra el 54 % de la demanda)."""
    totals = train.groupby("estacion_id")["salidas"].sum().sort_values(ascending=False)
    return set(totals.index[: max(1, round(share * len(totals)))])


def load_partitions() -> dict:
    """Particiones de v4, sin transformar: el preprocesamiento vive en el Pipeline."""
    data = pd.read_parquet(config.PROCESSED_DATA_DIR / "model_input.parquet")
    return {p: data[data["particion"] == p].reset_index(drop=True) for p in ("train", *PARTITIONS)}


def fit_and_evaluate(name: str, params: dict, parts: dict, horizon: str = config.HORIZON):
    """Ajusta el Pipeline con todo el entrenamiento y lo evalúa en validación y en prueba (la
    prueba se mira una sola vez, con el modelo ya elegido)."""
    features = input_columns(horizon)
    model = build_pipeline(MODELS[name](params), horizon)
    model.fit(parts["train"][features], parts["train"]["salidas"])
    top = top_stations(parts["train"])
    metrics, predictions = {}, {}
    for p in PARTITIONS:
        predictions[p] = model.predict(parts[p][features])
        m = evaluate(parts[p]["salidas"], predictions[p], parts[p]["estacion_id"].isin(top))
        metrics.update({f"{k}_{p}": v for k, v in m.items()})
    return model, metrics, predictions


def log_and_register(model, name, params, metrics, predictions, parts, horizon) -> str:
    """En el run activo: métricas, gráficas del pronóstico y el Pipeline en el Model Registry,
    con los hiperparámetros y las métricas como etiquetas de la versión."""
    mlflow.log_metrics(metrics)
    test = parts["test"]
    figures = {
        "pronostico_semana.png": plot_forecast_week(test, predictions["test"]),
        "mae_por_hora.png": plot_error_by_hour(
            test, {name: predictions["test"], "persistencia semanal": test["lag_168h"]}
        ),
    }
    for file, fig in figures.items():
        mlflow.log_figure(fig, file)
        plt.close(fig)

    sample = parts["valid"][list(model.feature_names_in_)].head(1000)
    info = mlflow.sklearn.log_model(
        model, name="model", registered_model_name=config.REGISTERED_MODEL,
        signature=infer_signature(sample, model.predict(sample)), input_example=sample.head(5),
        # skops no admite transformadores propios (StationProfilePCA): se usa cloudpickle y el
        # código del paquete viaja con el modelo
        serialization_format="cloudpickle", code_paths=[str(config.PROJ_ROOT / "mibici")],
    )  # fmt: skip

    client, version = MlflowClient(), info.registered_model_version
    client.update_model_version(
        config.REGISTERED_MODEL, version,
        description=f"{name}, horizonte {horizon}: Pipeline (preprocesamiento + modelo) entrenado "
                    f"con demanda-v4, oct 2025 a jun 2026. MAE validación "
                    f"{metrics['mae_valid']:.3f}, prueba {metrics['mae_test']:.3f} salidas por "
                    f"estación-hora.",
    )  # fmt: skip
    tags = {"modelo": name, "horizonte": horizon, "dataset": "demanda-v4",
            **{f"param_{k}": v for k, v in params.items()},
            **{k: round(v, 4) for k, v in metrics.items()}}  # fmt: skip
    for key, value in tags.items():
        client.set_model_version_tag(config.REGISTERED_MODEL, version, key, str(value))
    return version


def set_champion(horizon: str = config.HORIZON) -> str:
    """Alias "champion" para la versión con menor devianza de Poisson en validación (la métrica
    que optimiza Optuna) entre todas las registradas para el horizonte."""
    client = MlflowClient()
    versions = [v for v in client.search_model_versions(f"name='{config.REGISTERED_MODEL}'")
                if v.tags.get("horizonte") == horizon]  # fmt: skip
    best = min(versions, key=lambda v: float(v.tags["devianza_poisson_valid"]))
    client.set_registered_model_alias(config.REGISTERED_MODEL, "champion", best.version)
    return f"{best.tags['modelo']} (versión {best.version})"


def best_params(name: str, horizon: str = config.HORIZON) -> dict:
    """Mejores hiperparámetros de la última búsqueda de Optuna de la familia (run "{name}-{h}")."""
    runs = mlflow.search_runs(
        experiment_names=[config.MLFLOW_EXPERIMENT_MODELOS],
        filter_string=f"attributes.run_name = '{name}-{horizon}'",
        order_by=["attributes.start_time DESC"], max_results=1,
    )  # fmt: skip
    assert len(runs), f"no hay búsqueda de Optuna para {name}-{horizon}: corre make tune"
    row = runs.iloc[0]
    return {col.removeprefix("params.mejor_"): ast.literal_eval(value)
            for col, value in row.items()
            if col.startswith("params.mejor_") and isinstance(value, str)}  # fmt: skip


@app.command()
def main(modelo: str = "hgb_poisson", horizonte: str = config.HORIZON):
    """Reentrena una familia con sus mejores hiperparámetros y registra una versión nueva."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_MODELOS)
    params = best_params(modelo, horizonte)
    parts = load_partitions()
    with mlflow.start_run(run_name=f"{modelo}-{horizonte}-reentrenamiento"):
        mlflow.log_params({"modelo": modelo, "horizonte": horizonte, "dataset": "demanda-v4",
                           "semilla": config.SEED, **params})  # fmt: skip
        model, metrics, predictions = fit_and_evaluate(modelo, params, parts, horizonte)
        version = log_and_register(model, modelo, params, metrics, predictions, parts, horizonte)
    logger.info(f"MAE validación {metrics['mae_valid']:.3f}, prueba {metrics['mae_test']:.3f}")
    logger.success(f"{modelo}: versión {version}. Campeón: {set_champion(horizonte)}")


if __name__ == "__main__":
    app()
