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

Cada modelo es un Pipeline de scikit-learn (mibici/modeling/pipeline.py): el preprocesamiento se
ajusta dentro de cada prueba con los mismos datos que el estimador. El modelo final de cada familia
se registra en el MLflow Model Registry (versión, hiperparámetros, métricas y gráficas) y la versión
con menor devianza de Poisson en validación recibe el alias "champion".

Comando:
    python -m mibici.modeling.tune                                   # 3 familias, horizonte 24 h
    python -m mibici.modeling.tune --modelos random_forest           # una familia (otra sesión)
    python -m mibici.modeling.tune --modelos hgb_poisson --trials 40 --horizonte 1h

La tabla reports/comparacion_modelos_{h}.csv se arma con la última corrida de cada modelo en MLflow,
así que las familias pueden correrse por separado sin perder la comparación.
"""

import time
from typing import Annotated
import warnings

from loguru import logger
import mlflow
import optuna
import pandas as pd
import typer

from mibici import config
from mibici.modeling.pipeline import build_pipeline, input_columns
from mibici.modeling.train import (
    MODELS,
    PARTITIONS,
    evaluate,
    fit_and_evaluate,
    load_partitions,
    log_and_register,
    set_champion,
    top_stations,
)

app = typer.Typer()
optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore", category=optuna.exceptions.ExperimentalWarning)

# Rango en el que Optuna busca los hiperparámetros de cada familia (MODELS vive en train.py)
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


def tune_family(name, X_fit, y_fit, X_valid, y_valid, top_mask, n_trials, horizon) -> optuna.Study:
    """Búsqueda bayesiana (TPE) de hiperparámetros; cada prueba es un run anidado en MLflow."""

    def objective(trial):
        params = SPACES[name](trial)
        with mlflow.start_run(run_name=f"{name}-prueba-{trial.number:02d}", nested=True):
            start = time.perf_counter()
            model = build_pipeline(MODELS[name](params), horizon).fit(X_fit, y_fit)
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


def latest_runs(horizon: str) -> pd.DataFrame:
    """Última corrida de cada modelo para el horizonte (runs padre, sin las pruebas anidadas)."""
    runs = mlflow.search_runs(
        experiment_names=[config.MLFLOW_EXPERIMENT_MODELOS],
        filter_string=f"attributes.run_name LIKE '%-{horizon}' AND attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
    )  # fmt: skip
    if runs.empty:
        return pd.DataFrame(columns=["modelo"])
    runs["modelo"] = runs["params.modelo"]
    return runs.drop_duplicates("modelo")


def comparison_table(horizon: str) -> pd.DataFrame:
    """Tabla de métricas leída de MLflow: sirve aunque las familias se corran por separado
    (por ejemplo, `--modelos random_forest` en otra sesión)."""
    rows = []
    for _, run in latest_runs(horizon).iterrows():
        for p in PARTITIONS:
            metrics = {col.removeprefix("metrics.").removesuffix(f"_{p}"): run[col]
                       for col in run.index
                       if col.startswith("metrics.") and col.endswith(f"_{p}")}  # fmt: skip
            rows.append({"modelo": run["modelo"], "particion": p, **metrics})
    order = {name: i for i, name in enumerate(["persistencia_semanal", *MODELS, *PARTITIONS])}
    table = pd.DataFrame(rows).sort_values(["modelo", "particion"], key=lambda c: c.map(order))
    cols = [
        "modelo",
        "particion",
        "mae",
        "rmse",
        "devianza_poisson",
        "sesgo",
        "mae_top_estaciones",
    ]
    return table[cols].round(4).reset_index(drop=True)


@app.command()
def main(
    modelos: Annotated[list[str], typer.Option(help="Familias a comparar")] = ALL_MODELS,
    trials: int = config.OPTUNA_TRIALS,
    horizonte: Annotated[str, typer.Option(help="24h (la noche anterior) o 1h")] = config.HORIZON,
):
    unknown = set(modelos) - set(MODELS)
    assert not unknown, f"modelos desconocidos: {unknown}"
    parts = load_partitions()
    features = input_columns(horizonte)  # columnas que recibe el Pipeline
    fit = parts["train"].sample(
        min(config.OPTUNA_SAMPLE_ROWS, len(parts["train"])), random_state=config.SEED
    )
    top = top_stations(parts["train"])
    masks = {p: parts[p]["estacion_id"].isin(top) for p in PARTITIONS}

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT_MODELOS)
    if latest_runs(horizonte).query("modelo == 'persistencia_semanal'").empty:
        with mlflow.start_run(run_name=f"persistencia_semanal-{horizonte}"):  # referencia
            mlflow.log_params({"modelo": "persistencia_semanal", "horizonte": horizonte,
                               "dataset": "demanda-v4"})  # fmt: skip
            for p in PARTITIONS:  # la persistencia semanal es el rezago de 168 h sin transformar
                m = evaluate(parts[p]["salidas"], parts[p]["lag_168h"], masks[p])
                mlflow.log_metrics({f"{k}_{p}": v for k, v in m.items()})

    for name in modelos:
        with mlflow.start_run(run_name=f"{name}-{horizonte}"):
            mlflow.log_params({"modelo": name, "horizonte": horizonte, "dataset": "demanda-v4",
                               "pruebas_optuna": trials, "filas_ajuste_optuna": len(fit),
                               "objetivo": "devianza_poisson_valid", "semilla": config.SEED,
                               "n_features": len(features)})  # fmt: skip
            study = tune_family(name, fit[features], fit["salidas"], parts["valid"][features],
                                parts["valid"]["salidas"], masks["valid"], trials,
                                horizonte)  # fmt: skip
            mlflow.log_params({f"mejor_{k}": v for k, v in study.best_params.items()})
            mlflow.log_text(study.trials_dataframe().to_csv(index=False), "optuna_pruebas.csv")
            mlflow.log_text("\n".join(features), "features.txt")
            log_study_figures(study)

            # El mejor juego se reentrena con todo el entrenamiento y se registra
            model, metrics, predictions = fit_and_evaluate(name, study.best_params, parts,
                                                           horizonte)  # fmt: skip
            version = log_and_register(model, name, study.best_params, metrics, predictions,
                                       parts, horizonte)  # fmt: skip
            logger.info(f"{name}: MAE validación {metrics['mae_valid']:.3f}, versión {version}")

    table = comparison_table(horizonte)
    out = config.REPORTS_DIR / f"comparacion_modelos_{horizonte}.csv"
    table.to_csv(out, index=False)
    logger.success(f"Campeón: {set_champion(horizonte)}")
    logger.success(
        f"Comparación de modelos ({horizonte}):\n{table.to_string(index=False)}\n-> {out}"
    )


if __name__ == "__main__":
    app()
