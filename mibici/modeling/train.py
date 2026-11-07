"""Entrenamiento, ajuste de hiperparámetros y registro de modelos (Fase 2).

1. Líneas base: persistencias del mismo horizonte (no se entrenan).
2. Para cada modelo candidato, Optuna busca hiperparámetros minimizando el MAE en validación
   (jul-ago 2026). Cada configuración probada es un run anidado en MLflow.
3. La mejor configuración de cada modelo se reentrena con entrenamiento (oct 2025-jun 2026),
   se evalúa en validación y en prueba (sep 2026) y se registra en el Model Registry con sus
   hiperparámetros, métricas y gráficas.
4. El modelo con menor MAE en validación recibe el alias "champion"; prueba solo se usa
   para reportar, nunca para elegir.

Comando:
    python -m mibici.modeling.train
"""

from pathlib import Path

from loguru import logger
import mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
import optuna
import pandas as pd
import typer

from mibici import config
from mibici.modeling.pipeline import build_pipeline, evaluate, input_columns
from mibici.plots import plot_error_by_hour, plot_forecast_week

app = typer.Typer()
TARGET = "salidas"
# Persistencias ingenuas para cada horizonte: repetir la demanda de un momento anterior
NAIVE = {
    "1h": {"persistencia_1h": "lag_1h"},
    "24h": {"persistencia_24h": "lag_24h", "persistencia_semanal": "lag_168h"},
}


def load_splits(path: Path) -> dict[str, pd.DataFrame]:
    df = pd.read_parquet(path)
    return {name: part.reset_index(drop=True) for name, part in df.groupby("particion")}


def top_station_mask(train: pd.DataFrame, df: pd.DataFrame, share: float = 0.2):
    """Filas de las estaciones que concentran la demanda (20 % con más salidas en train)."""
    totals = train.groupby("estacion_id")[TARGET].sum().sort_values(ascending=False)
    return df["estacion_id"].isin(totals.index[: int(share * len(totals))]).to_numpy()


def suggest(trial: optuna.Trial, space: dict) -> dict:
    return {
        name: trial.suggest_float(name, low, high, log=True)
        if kind == "log"
        else trial.suggest_int(name, low, high)
        for name, (kind, low, high) in space.items()
    }


def tune(model_name: str, horizon: str, train: pd.DataFrame, valid: pd.DataFrame):
    """Busca hiperparámetros con Optuna (muestreador TPE con semilla fija)."""
    X_train, X_valid = train[input_columns(horizon)], valid[input_columns(horizon)]

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, config.SEARCH_SPACE[model_name])
        with mlflow.start_run(run_name=f"{model_name}-trial-{trial.number:02d}", nested=True):
            model = build_pipeline(model_name, params, horizon).fit(X_train, train[TARGET])
            metrics = evaluate(valid[TARGET], model.predict(X_valid))
            mlflow.set_tags({"modelo": model_name, "tipo": "trial_optuna"})
            mlflow.log_params(params)
            mlflow.log_metrics({f"valid_{k}": v for k, v in metrics.items()})
        return metrics[config.TUNING_METRIC]

    sampler = optuna.samplers.TPESampler(seed=config.RANDOM_STATE)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=config.N_TRIALS[model_name])
    return study


def describe_version(client: MlflowClient, version: str, model_name: str, params: dict,
                     metrics: dict, horizon: str) -> None:  # fmt: skip
    """Deja en la versión del Model Registry los hiperparámetros y las métricas."""
    name = config.REGISTERED_MODEL
    client.update_model_version(
        name, version,
        description=f"{model_name}, horizonte {horizon}, entrenado con demanda-v4 (oct 2025-jun "
                    f"2026). MAE validación {metrics['valid_mae']:.3f}, prueba "
                    f"{metrics['test_mae']:.3f} salidas por estación-hora.",
    )  # fmt: skip
    tags = {"modelo": model_name, "horizonte": horizon, "dataset": "demanda-v4",
            **{f"param_{k}": v for k, v in params.items()},
            **{k: round(v, 4) for k, v in metrics.items()}}  # fmt: skip
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, str(value))


@app.command()
def main(
    input_path: Path = config.PROCESSED_DATA_DIR / "model_input.parquet",
    horizon: str = config.HORIZON,
):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    splits = load_splits(input_path)
    train, valid, test = splits["train"], splits["valid"], splits["test"]
    top = {"valid": top_station_mask(train, valid), "test": top_station_mask(train, test)}
    evaluation = {"valid": valid, "test": test}

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MODELS_EXPERIMENT)
    client = MlflowClient()
    valid_mae = {}

    # 1. Líneas base
    for name, column in NAIVE[horizon].items():
        with mlflow.start_run(run_name=name):
            mlflow.set_tags({"modelo": name, "tipo": "linea_base", "horizonte": horizon})
            for split, df in evaluation.items():
                metrics = evaluate(df[TARGET], df[column], top[split])
                mlflow.log_metrics({f"{split}_{k}": v for k, v in metrics.items()})
            valid_mae[name] = evaluate(valid[TARGET], valid[column])["mae"]

    # 2-3. Modelos candidatos: ajuste, evaluación y registro
    versions = {}
    for model_name in config.SEARCH_SPACE:
        with mlflow.start_run(run_name=f"{model_name}-optuna"):
            study = tune(model_name, horizon, train, valid)
            model = build_pipeline(model_name, study.best_params, horizon)
            model.fit(train[input_columns(horizon)], train[TARGET])

            metrics, predictions = {}, {}
            for split, df in evaluation.items():
                predictions[split] = model.predict(df[input_columns(horizon)])
                split_metrics = evaluate(df[TARGET], predictions[split], top[split])
                metrics.update({f"{split}_{k}": v for k, v in split_metrics.items()})
            mlflow.set_tags({"modelo": model_name, "tipo": "mejor_modelo", "horizonte": horizon})
            mlflow.log_params({**study.best_params, "modelo": model_name, "horizonte": horizon,
                               "trials": len(study.trials), "semilla": config.RANDOM_STATE})  # fmt: skip
            mlflow.log_metrics(metrics)
            mlflow.log_figure(
                plot_forecast_week(test, predictions["test"]), "pronostico_semana.png"
            )
            mlflow.log_figure(
                plot_error_by_hour(test, {model_name: predictions["test"],
                                          "persistencia semanal": test["lag_168h"]}),
                "mae_por_hora.png",
            )  # fmt: skip

            sample = valid[input_columns(horizon)].head(1000)
            info = mlflow.sklearn.log_model(
                model, name="modelo", registered_model_name=config.REGISTERED_MODEL,
                signature=infer_signature(sample, model.predict(sample)),
                input_example=sample.head(5), code_paths=[str(config.PROJ_ROOT / "mibici")],
                # skops no admite transformadores propios (StationProfilePCA); el código del
                # paquete viaja con el modelo en code_paths
                serialization_format="cloudpickle",
            )  # fmt: skip
            versions[model_name] = info.registered_model_version
            describe_version(client, versions[model_name], model_name, study.best_params,
                             metrics, horizon)  # fmt: skip
            valid_mae[model_name] = metrics["valid_mae"]
            logger.info(f"{model_name}: versión {versions[model_name]}, MAE valid "
                        f"{metrics['valid_mae']:.3f}, test {metrics['test_mae']:.3f}")  # fmt: skip

    # 4. Campeón: menor MAE en validación entre los modelos registrados
    champion = min(versions, key=valid_mae.get)
    client.set_registered_model_alias(config.REGISTERED_MODEL, "champion", versions[champion])
    for name, mae in sorted(valid_mae.items(), key=lambda kv: kv[1]):
        logger.info(f"{name:>22}: MAE validación {mae:.3f}")
    logger.success(f"Campeón: {champion} (versión {versions[champion]})")
    assert valid_mae[champion] < min(valid_mae[n] for n in NAIVE[horizon]), (
        "el campeón no supera a las líneas base"
    )


if __name__ == "__main__":
    app()
