"""Selección de algoritmos y ajuste de hiperparámetros con Optuna (Fase 2).

El problema es de conteos con muchos ceros: el 46 % de las estación-horas no tiene salidas y la
cola derecha es larga (reports/hallazgos_eda.md). Se comparan tres familias de algoritmos:
  * poisson_glm: regresión de Poisson (modelo lineal generalizado con liga log). Es la
    referencia natural para conteos: predice medias positivas y sus coeficientes se interpretan.
  * random_forest: bosque aleatorio. Capta interacciones no lineales (hora x tipo de estación)
    promediando árboles independientes; menos sensible a hiperparámetros que el boosting.
  * hgb_poisson: gradient boosting por histogramas con pérdida Poisson. Une la flexibilidad de
    los árboles con la pérdida adecuada para conteos y escala a millones de filas.

Cada familia es un run padre del experimento `mibici-fase2-modelos` y cada prueba de Optuna es un
run anidado con sus hiperparámetros y métricas en validación (jul-ago 2026). Optuna minimiza la
devianza de Poisson en validación, la métrica propia de un modelo de conteos. Después, el mejor
juego de cada familia se reentrena con todo el entrenamiento, se evalúa en validación y una sola
vez en prueba (sep 2026), y el modelo queda guardado en el run padre.

Comando:
    python -m mibici.modeling.tune                                   # 3 familias, horizonte 24 h
    python -m mibici.modeling.tune --modelos hgb_poisson --trials 40 --horizonte 1h
"""

import time
from typing import Annotated
import warnings

from loguru import logger
import mlflow
import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error
import typer

from mibici import config
from mibici.modeling.train import ID_COLS, NOT_AVAILABLE_24H

app = typer.Typer()
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# Cada familia: cómo se construye el modelo y en qué rango busca Optuna sus hiperparámetros
MODELS = {
    "poisson_glm": lambda p: PoissonRegressor(max_iter=1000, **p),
    "random_forest": lambda p: RandomForestRegressor(
        max_samples=0.2, n_jobs=-1, random_state=config.SEED, **p
    ),
    "hgb_poisson": lambda p: HistGradientBoostingRegressor(
        loss="poisson", early_stopping=False, random_state=config.SEED, **p
    ),
}
SPACES = {
    "poisson_glm": lambda t: {"alpha": t.suggest_float("alpha", 1e-6, 1.0, log=True)},
    "random_forest": lambda t: {
        "n_estimators": t.suggest_int("n_estimators", 50, 250, step=25),
        "max_depth": t.suggest_int("max_depth", 8, 22),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 30, 1000, log=True),
        "max_features": t.suggest_float("max_features", 0.3, 0.8),
    },
    "hgb_poisson": lambda t: {
        "learning_rate": t.suggest_float("learning_rate", 0.02, 0.3, log=True),
        "max_iter": t.suggest_int("max_iter", 200, 1500, step=100),
        "max_leaf_nodes": t.suggest_int("max_leaf_nodes", 15, 255, log=True),
        "min_samples_leaf": t.suggest_int("min_samples_leaf", 20, 500, log=True),
        "l2_regularization": t.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
    },
}
ALL_MODELS = list(MODELS)
# MLflow guarda los modelos con skops, que por seguridad rechaza los árboles de scikit-learn si no
# se declaran. Son modelos entrenados aquí mismo, así que cargarlos es seguro.
SKOPS_TRUSTED = ["sklearn.tree._tree.Tree",
                 "sklearn.ensemble._hist_gradient_boosting.predictor.TreePredictor"]  # fmt: skip


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
    """Particiones de v3. Validación y prueba llevan el rezago semanal sin escalar para medir la
    persistencia semanal, la línea base que el ML Canvas pide superar en al menos 10 %."""
    ready = pd.read_parquet(config.PROCESSED_DATA_DIR / "model_ready.parquet")
    lag = pd.read_parquet(
        config.PROCESSED_DATA_DIR / "demand_panel.parquet",
        columns=["estacion_id", "hora", "lag_168h"],
    ).rename(columns={"lag_168h": "persistencia_semanal"})
    parts = {p: ready[ready["particion"] == p] for p in ("train", "valid", "test")}
    for p in ("valid", "test"):
        parts[p] = parts[p].merge(lag, on=["estacion_id", "hora"])
    return parts


def tune_family(name, X_fit, y_fit, X_valid, y_valid, top_mask, n_trials) -> optuna.Study:
    """Búsqueda bayesiana (TPE) de hiperparámetros; cada prueba es un run anidado en MLflow."""

    def objective(trial):
        params = SPACES[name](trial)
        with mlflow.start_run(run_name=f"{name}-prueba-{trial.number:02d}", nested=True):
            start = time.perf_counter()
            model = MODELS[name](params).fit(X_fit, y_fit)
            metrics = evaluate(y_valid, model.predict(X_valid), top_mask)
            mlflow.log_params({"modelo": name, "prueba": trial.number, **params})
            mlflow.log_metrics({**metrics, "segundos_ajuste": time.perf_counter() - start})
        logger.info(f"{name} prueba {trial.number:02d}: devianza={metrics['devianza_poisson']:.4f}"
                    f" MAE={metrics['mae']:.3f}")  # fmt: skip
        return metrics["devianza_poisson"]

    sampler = optuna.samplers.TPESampler(seed=config.SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler, study_name=name)
    study.optimize(objective, n_trials=n_trials)
    return study


def log_study_figures(study: optuna.Study):
    """Historial de la optimización e importancia de cada hiperparámetro, como figuras de MLflow."""
    import matplotlib

    matplotlib.use("Agg")
    from optuna.visualization import matplotlib as optuna_plots

    figures = {"optuna_historial.png": optuna_plots.plot_optimization_history}
    if len(study.best_params) > 1:  # la importancia necesita al menos dos hiperparámetros
        figures["optuna_importancia.png"] = optuna_plots.plot_param_importances
    for file, plot in figures.items():
        ax = plot(study)
        ax.figure.tight_layout()
        mlflow.log_figure(ax.figure, file)
        matplotlib.pyplot.close(ax.figure)


@app.command()
def main(
    modelos: Annotated[list[str], typer.Option(help="Familias a comparar")] = ALL_MODELS,
    trials: int = config.OPTUNA_TRIALS,
    horizonte: Annotated[str, typer.Option(help="24h (la noche anterior) o 1h")] = "24h",
):
    unknown = set(modelos) - set(MODELS)
    assert not unknown, f"modelos desconocidos: {unknown}"
    parts = load_partitions()
    excluded = NOT_AVAILABLE_24H if horizonte == "24h" else []
    features = [c for c in parts["train"].columns if c not in ID_COLS + excluded]
    fit = parts["train"].sample(
        min(config.OPTUNA_SAMPLE_ROWS, len(parts["train"])), random_state=config.SEED
    )
    top = top_stations(parts["train"])
    masks = {p: parts[p]["estacion_id"].isin(top) for p in ("valid", "test")}

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_MODELOS)
    rows = []
    with mlflow.start_run(run_name=f"persistencia_semanal-{horizonte}"):  # referencia en la UI
        mlflow.log_params({"modelo": "persistencia_semanal", "horizonte": horizonte,
                           "dataset": "demanda-v3"})  # fmt: skip
        for p in ("valid", "test"):
            m = evaluate(parts[p]["salidas"], parts[p]["persistencia_semanal"], masks[p])
            mlflow.log_metrics({f"{k}_{p}": v for k, v in m.items()})
            rows.append({"modelo": "persistencia_semanal", "particion": p, **m})

    for name in modelos:
        with mlflow.start_run(run_name=f"{name}-{horizonte}"):
            mlflow.log_params({"modelo": name, "horizonte": horizonte, "dataset": "demanda-v3",
                               "pruebas_optuna": trials, "filas_ajuste_optuna": len(fit),
                               "objetivo": "devianza_poisson_valid", "semilla": config.SEED,
                               "n_features": len(features)})  # fmt: skip
            study = tune_family(name, fit[features], fit["salidas"], parts["valid"][features],
                                parts["valid"]["salidas"], masks["valid"], trials)  # fmt: skip
            mlflow.log_params({f"mejor_{k}": v for k, v in study.best_params.items()})

            # El mejor juego se reentrena con todo el entrenamiento; prueba se mira una sola vez
            model = MODELS[name](study.best_params).fit(parts["train"][features],
                                                        parts["train"]["salidas"])  # fmt: skip
            for p in ("valid", "test"):
                m = evaluate(parts[p]["salidas"], model.predict(parts[p][features]), masks[p])
                mlflow.log_metrics({f"{k}_{p}": v for k, v in m.items()})
                rows.append({"modelo": name, "particion": p, **m})
            mlflow.log_text(study.trials_dataframe().to_csv(index=False), "optuna_pruebas.csv")
            mlflow.log_text("\n".join(features), "features.txt")
            log_study_figures(study)
            # Ejemplo en float: así el esquema admite enteros y nulos al predecir
            example = parts["valid"][features].head(5).astype("float64")
            mlflow.sklearn.log_model(model, name="model", input_example=example,
                                     skops_trusted_types=SKOPS_TRUSTED)  # fmt: skip

    table = pd.DataFrame(rows).round(4)
    out = config.REPORTS_DIR / f"comparacion_modelos_{horizonte}.csv"
    table.to_csv(out, index=False)
    logger.success(
        f"Comparación de modelos ({horizonte}):\n{table.to_string(index=False)}\n-> {out}"
    )


if __name__ == "__main__":
    app()
