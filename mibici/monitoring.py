"""Monitoreo del modelo: drift de datos, drift de predicciones y pérdida de desempeño.

Preguntas que responde (Entrega Final):
1. ¿Cambió la distribución de las entradas, de las predicciones o de la demanda real entre la línea
   base (validación, jul–ago 2026) y el mes actual (septiembre 2026)?
2. Si simulamos un cambio en el mundo (escenarios sintéticos), ¿el monitoreo lo detecta y cuánto
   empeora el modelo?
3. ¿Cómo evoluciona el error mes a mes frente a la persistencia semanal?
4. ¿Dónde falla? Septiembre trae 58 estaciones que el modelo no vio al entrenar.

Prueba de drift por columna (la misma regla por omisión que usa Evidently con más de 1,000 filas):
- numéricas: distancia de Wasserstein normalizada por la desviación estándar de la referencia;
  hay drift si es mayor que 0.1;
- categóricas: distancia de Jensen-Shannon entre frecuencias; hay drift si es mayor que 0.1;
- el dataset tiene drift si al menos la mitad de las columnas lo tiene.

Se reporta además el PSI (Population Stability Index), que es fácil de explicar: menos de 0.1 es
estable, entre 0.1 y 0.2 es un cambio moderado y más de 0.2 es significativo.

El reporte HTML de Evidently se genera aparte (`make drift-evidently`), en un entorno aislado,
porque Evidently 0.7.23 todavía no es compatible con pandas 3 (ver references/monitoreo.md).

Comandos:
    python -m mibici.monitoring drift          # referencia vs septiembre y escenarios -> MLflow
    python -m mibici.monitoring log            # drift de las peticiones reales (logs/*.jsonl)
"""

from dataclasses import dataclass
import json
from pathlib import Path

from loguru import logger
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance
import typer

from mibici import config

app = typer.Typer()

NUMERIC = ["capacidad", "lat", "lon", "lag_24h", "lag_168h", "hora_dia"]
CATEGORICAL = ["municipio", "dia_semana", "fin_de_semana", "festivo"]
PREDICTION, TARGET = "salidas_pred", "salidas"
THRESHOLD = 0.1  # regla de Evidently para numéricas (Wasserstein normalizado) y categóricas (JS)
DATASET_DRIFT_SHARE = 0.5
SAMPLE_ROWS = 60_000
MONITORING_DIR = config.REPORTS_DIR / "monitoreo"
MLFLOW_EXPERIMENT_MONITOREO = "mibici-monitoreo"


# --- Medidas de drift ----------------------------------------------------------------------------
def psi(reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
    """Population Stability Index con cortes en los cuantiles de la referencia."""
    ref, cur = reference.dropna().to_numpy(float), current.dropna().to_numpy(float)
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:  # columna casi constante (p. ej. muchos ceros): usa sus valores únicos
        edges = np.unique(np.concatenate([ref, cur]))
        edges = np.append(edges, edges[-1] + 1)
    edges[0], edges[-1] = -np.inf, np.inf
    p = np.histogram(ref, edges)[0] / len(ref)
    q = np.histogram(cur, edges)[0] / len(cur)
    p, q = np.clip(p, 1e-6, None), np.clip(q, 1e-6, None)
    return float(np.sum((q - p) * np.log(q / p)))


def numeric_drift(reference: pd.Series, current: pd.Series) -> float:
    """Wasserstein normalizado: cuántas desviaciones estándar de la referencia se movió la
    distribución. Es la prueba que usa Evidently para columnas numéricas grandes."""
    ref, cur = reference.dropna().to_numpy(float), current.dropna().to_numpy(float)
    scale = ref.std() or 1.0
    return float(wasserstein_distance(ref, cur) / scale)


def categorical_drift(reference: pd.Series, current: pd.Series) -> float:
    """Distancia de Jensen-Shannon entre las frecuencias de cada categoría."""
    levels = sorted(set(reference.dropna()) | set(current.dropna()), key=str)
    p = reference.value_counts(normalize=True).reindex(levels, fill_value=0)
    q = current.value_counts(normalize=True).reindex(levels, fill_value=0)
    return float(jensenshannon(p, q, base=2))


def drift_table(reference: pd.DataFrame, current: pd.DataFrame, columns: dict) -> pd.DataFrame:
    """Una fila por columna: tipo, prueba, distancia, drift sí/no, PSI y medias."""
    rows = []
    for column, kind in columns.items():
        if kind == "num":
            distance, method = numeric_drift(reference[column], current[column]), "wasserstein_n"
            the_psi = psi(reference[column], current[column])
        else:
            distance, method = (
                categorical_drift(reference[column], current[column]),
                "jensen_shannon",
            )
            the_psi = np.nan
        rows.append(
            {
                "columna": column,
                "tipo": kind,
                "prueba": method,
                "distancia": round(distance, 4),
                "drift": distance > THRESHOLD,
                "psi": round(the_psi, 4) if not np.isnan(the_psi) else np.nan,
                "media_referencia": _mean(reference[column]),
                "media_actual": _mean(current[column]),
            }
        )
    return pd.DataFrame(rows)


def _mean(series: pd.Series):
    return round(float(series.mean()), 4) if pd.api.types.is_numeric_dtype(series) else None


def monitored_columns() -> dict:
    """Entradas del modelo, la predicción y la demanda real (drift de etiqueta)."""
    return {**{c: "num" for c in NUMERIC}, **{c: "cat" for c in CATEGORICAL},
            PREDICTION: "num", TARGET: "num"}  # fmt: skip


# --- Datos de referencia, actuales y escenarios ---------------------------------------------------
def load_scored(model) -> pd.DataFrame:
    """v4 completo con la predicción del campeón y el mes, para comparar periodos."""
    data = pd.read_parquet(config.PROCESSED_DATA_DIR / "model_input.parquet")
    data[PREDICTION] = model.predict(data[list(model.feature_names_in_)])
    data["mes"] = data["hora"].dt.to_period("M").astype(str)
    return data


def sample(df: pd.DataFrame, n: int = SAMPLE_ROWS, seed: int = config.SEED) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=seed).reset_index(drop=True)


@dataclass
class Scenario:
    nombre: str
    descripcion: str
    base: str = "valid"  # periodo al que se aplica: validación (jul–ago) o prueba (sep)

    def apply(self, current: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        return current


class DemandSurge(Scenario):
    """Un evento o una obra sube la demanda real 60 % en el 30 % de las estaciones: la demanda
    de ayer y de la semana pasada (los rezagos) también la reflejan."""

    def apply(self, current, rng):
        df = current.copy()
        stations = df["estacion_id"].unique()
        hit = rng.choice(stations, size=int(0.3 * len(stations)), replace=False)
        mask = df["estacion_id"].isin(hit)
        for column in ("salidas", "lag_24h", "lag_168h"):
            df.loc[mask, column] = rng.poisson(1.6 * df.loc[mask, column].to_numpy())
        return df


class BrokenFeed(Scenario):
    """Falla la fuente: el 25 % de las filas llega con los rezagos en 0 (la estación no reportó),
    aunque la demanda real no cambió."""

    def apply(self, current, rng):
        df = current.copy()
        mask = rng.random(len(df)) < 0.25
        df.loc[mask, ["lag_24h", "lag_168h"]] = 0
        return df


class StationClosures(Scenario):
    """Obras cierran el 15 % de las estaciones a media semana: su demanda real cae al 10 %, pero
    los rezagos todavía son de la semana normal. Las entradas no cambian (drift de concepto): sólo
    lo delata comparar el pronóstico con la demanda real."""

    def apply(self, current, rng):
        df = current.copy()
        stations = df["estacion_id"].unique()
        hit = rng.choice(stations, size=int(0.15 * len(stations)), replace=False)
        mask = df["estacion_id"].isin(hit)
        df.loc[mask, "salidas"] = rng.binomial(df.loc[mask, "salidas"].astype(int), 0.10)
        return df


# La línea base es el set de validación (jul–ago 2026), como pide la Entrega Final. Los escenarios
# sintéticos lo alteran: desplazamiento de medias, variables faltantes y drift de concepto. Septiembre
# es el cambio estacional real (y trae 58 estaciones nuevas).
SCENARIOS = [
    Scenario("validacion", "Línea base: validación jul–ago 2026 sin cambios"),
    Scenario("septiembre_real", "Cambio estacional real: septiembre 2026 (mes de prueba)", "test"),
    DemandSurge(
        "aumento_demanda", "Desplazamiento de medias: +60 % de demanda en 30 % de estaciones"
    ),
    BrokenFeed("rezagos_rotos", "Variables faltantes: 25 % de filas con los rezagos en 0"),
    StationClosures(
        "cierre_estaciones", "Drift de concepto: la demanda real cae al 10 % en 15 % de estaciones"
    ),
]


def score(model, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out[PREDICTION] = model.predict(out[list(model.feature_names_in_)])
    return out


def performance(df: pd.DataFrame) -> dict:
    """Error del modelo y de la persistencia semanal en el mismo periodo."""
    from mibici.modeling.train import evaluate

    model_m = evaluate(df[TARGET], df[PREDICTION])
    base_mae = float((df[TARGET] - df["lag_168h"]).abs().mean())
    return {
        "mae": model_m["mae"],
        "sesgo": model_m["sesgo"],
        "devianza_poisson": model_m["devianza_poisson"],
        "mae_persistencia": base_mae,
        "mejora_vs_persistencia_pct": 100 * (1 - model_m["mae"] / base_mae),
    }


def alerts(perf: dict, baseline: dict) -> dict:
    """Consecuencias pre-negociadas: compara el periodo contra la línea base (validación)."""
    ratio = perf["mae"] / baseline["mae"]
    return {
        "mae_vs_validacion": round(ratio, 3),
        "alerta_desempeno": bool(ratio > config.MONITOR_MAE_RATIO),
        "alerta_sesgo": bool(abs(perf["sesgo"]) > config.MONITOR_MAX_ABS_BIAS),
        "alerta_persistencia_gana": bool(perf["mae"] >= perf["mae_persistencia"]),
    }


def proposed_action(row: dict) -> str:
    """Acción propuesta según qué se encendió (ver references/monitoreo.md)."""
    performance_alert = row["alerta_desempeno"] or row["alerta_sesgo"]
    if performance_alert and row["drift_etiqueta"]:
        return "reentrenar con los meses recientes (la demanda real cambió)"
    if performance_alert and row["columnas_con_drift"] == 0:
        return "drift de concepto: confirmar con el operador (cierres, obras) y reentrenar"
    if row["columnas_con_drift"] > 0 and not performance_alert:
        return "revisar el pipeline de features y la fuente de datos (las entradas cambiaron)"
    if performance_alert:
        return "revisar el pipeline de features y reentrenar"
    return "sin acción"


def segment_performance(df: pd.DataFrame, seen: set) -> pd.DataFrame:
    """Error por tipo de estación (vista o nueva) con el modelo, la persistencia y la política de
    arranque en frío (persistencia para las estaciones nuevas)."""
    df = df.assign(estacion_nueva=~df["estacion_id"].isin(seen))
    policy = df[PREDICTION].where(~df["estacion_nueva"], df["lag_168h"])
    df = df.assign(
        err=(df[TARGET] - df[PREDICTION]).abs(),
        err_base=(df[TARGET] - df["lag_168h"]).abs(),
        err_politica=(df[TARGET] - policy).abs(),
        bias=df[PREDICTION] - df[TARGET],
    )
    out = df.groupby("estacion_nueva").agg(
        estaciones=("estacion_id", "nunique"), filas=(TARGET, "size"),
        demanda_media=(TARGET, "mean"), mae_modelo=("err", "mean"), sesgo=("bias", "mean"),
        mae_persistencia=("err_base", "mean"), mae_politica=("err_politica", "mean"),
    )  # fmt: skip
    total = df[["err", "err_base", "err_politica"]].mean()
    out.loc["total"] = [df["estacion_id"].nunique(), len(df), df[TARGET].mean(), total["err"],
                        df["bias"].mean(), total["err_base"], total["err_politica"]]  # fmt: skip
    return out.reset_index().round(4)


def monthly_performance(scored: pd.DataFrame) -> pd.DataFrame:
    """MAE y sesgo del campeón y MAE de la persistencia semanal, mes por mes."""
    g = scored.assign(
        err=(scored[TARGET] - scored[PREDICTION]).abs(),
        bias=scored[PREDICTION] - scored[TARGET],
        err_base=(scored[TARGET] - scored["lag_168h"]).abs(),
    ).groupby(["mes", "particion"], sort=True)
    out = g.agg(
        mae=("err", "mean"),
        sesgo=("bias", "mean"),
        mae_persistencia=("err_base", "mean"),
        demanda_media=(TARGET, "mean"),
        filas=(TARGET, "size"),
    ).reset_index()
    out["modelo_gana"] = out["mae"] < out["mae_persistencia"]
    return out.round(4)


# --- Figuras --------------------------------------------------------------------------------------
def plot_monthly(monthly: pd.DataFrame, path: Path) -> Path:
    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(monthly))
    ax.plot(x, monthly["mae"], marker="o", label="HGB Poisson (@champion)")
    ax.plot(x, monthly["mae_persistencia"], marker="s", ls="--", label="Persistencia semanal")
    for i, part in enumerate(monthly["particion"]):
        if part != "train":
            ax.axvspan(i - 0.5, i + 0.5, color="#f0c419" if part == "valid" else "#e8604c",
                       alpha=0.15, lw=0)  # fmt: skip
    ax.set_xticks(x, monthly["mes"], rotation=45, ha="right")
    ax.set_ylabel("MAE (salidas por estación-hora)")
    ax.set_title("Error mensual: entrenamiento, validación (amarillo) y prueba (rojo)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def plot_distributions(reference: pd.DataFrame, current: pd.DataFrame, columns, title, path):
    fig, axes = plt.subplots(1, len(columns), figsize=(4 * len(columns), 3.2))
    for ax, column in zip(np.atleast_1d(axes), columns):
        upper = np.nanquantile(pd.concat([reference[column], current[column]]), 0.99)
        bins = np.linspace(0, max(upper, 1), 30)
        ax.hist(reference[column].clip(upper=upper), bins, alpha=0.55, density=True,
                label="referencia (validación)")  # fmt: skip
        ax.hist(current[column].clip(upper=upper), bins, alpha=0.55, density=True, label="actual")
        ax.set_title(column)
    np.atleast_1d(axes)[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


# --- Comandos -------------------------------------------------------------------------------------
@app.command()
def drift(seed: int = config.SEED, out_dir: Path = MONITORING_DIR):
    """Línea base (validación) contra septiembre y contra cada escenario sintético."""
    from mibici.serving import load_bundle

    out_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_bundle()
    assert bundle.model is not None, "falta models/champion: corre make export-model"
    scored = load_scored(bundle.model)
    periods = {
        p: scored[scored["particion"] == p].reset_index(drop=True) for p in ("valid", "test")
    }
    reference = sample(periods["valid"], seed=seed)
    baseline = performance(periods["valid"])
    september = periods["test"]
    monthly = monthly_performance(scored)
    monthly.to_csv(out_dir / "error_mensual.csv", index=False)
    plot_monthly(monthly, out_dir / "error_mensual.png")
    reference.to_parquet(out_dir / "referencia.parquet", index=False)
    from mibici.serving import known_stations

    segments = segment_performance(september, known_stations(bundle.model))
    segments.to_csv(out_dir / "septiembre_por_tipo_de_estacion.csv", index=False)
    logger.info("Septiembre por tipo de estación:\n" + segments.to_string(index=False))

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_MONITOREO)
    rng = np.random.default_rng(seed)
    summary = []
    for scenario in SCENARIOS:
        current = score(bundle.model, scenario.apply(periods[scenario.base], rng))
        current_sample = sample(current, seed=seed)
        table = drift_table(reference, current_sample, monitored_columns())
        inputs = table[~table["columna"].isin([PREDICTION, TARGET])]
        share = float(inputs["drift"].mean())
        perf = performance(current)
        row = {
            "escenario": scenario.nombre,
            "descripcion": scenario.descripcion,
            "columnas_con_drift": int(inputs["drift"].sum()),
            "proporcion_drift": round(share, 3),
            "drift_dataset": share >= DATASET_DRIFT_SHARE,
            "drift_prediccion": bool(table.set_index("columna").loc[PREDICTION, "drift"]),
            "drift_etiqueta": bool(table.set_index("columna").loc[TARGET, "drift"]),
            **{k: round(v, 4) for k, v in perf.items()},
            **alerts(perf, baseline),
        }
        row["accion"] = proposed_action(row)
        summary.append(row)
        table_path = out_dir / f"drift_{scenario.nombre}.csv"
        table.to_csv(table_path, index=False)
        current_sample.to_parquet(out_dir / f"actual_{scenario.nombre}.parquet", index=False)
        fig_path = plot_distributions(
            reference, current_sample, ["lag_168h", PREDICTION, TARGET],
            f"{scenario.nombre}: {scenario.descripcion}", out_dir / f"dist_{scenario.nombre}.png",
        )  # fmt: skip
        with mlflow.start_run(run_name=f"drift-{scenario.nombre}"):
            mlflow.log_params({"escenario": scenario.nombre, "descripcion": scenario.descripcion,
                               "modelo_version": bundle.version, "referencia": "valid jul-ago",
                               "umbral": THRESHOLD, "filas_muestra": len(current_sample)})  # fmt: skip
            mlflow.log_metrics({k: float(v) for k, v in row.items()
                                if isinstance(v, (int, float, bool, np.bool_))})  # fmt: skip
            mlflow.log_metrics({f"dist_{r.columna}": r.distancia for r in table.itertuples()})
            for artifact in (table_path, fig_path, out_dir / "error_mensual.png",
                             out_dir / "error_mensual.csv",
                             out_dir / "septiembre_por_tipo_de_estacion.csv"):  # fmt: skip
                mlflow.log_artifact(str(artifact))
        logger.info(
            f"{scenario.nombre:>16}: drift en {row['columnas_con_drift']} columnas "
            f"(predicción {row['drift_prediccion']}, etiqueta {row['drift_etiqueta']}); "
            f"MAE {row['mae']:.3f} vs persistencia {row['mae_persistencia']:.3f}"
        )
    summary = pd.DataFrame(summary)
    summary.to_csv(out_dir / "resumen_escenarios.csv", index=False)
    (out_dir / "resumen_escenarios.json").write_text(
        json.dumps(summary.to_dict("records"), indent=2, ensure_ascii=False, default=str)
    )
    logger.success(f"Monitoreo -> {out_dir} y experimento {MLFLOW_EXPERIMENT_MONITOREO}")


@app.command()
def log(log_path: Path = config.PREDICTION_LOG, out_dir: Path = MONITORING_DIR):
    """Drift de las peticiones reales del log de eventos contra la referencia guardada."""
    from mibici.dataset import load_stations
    from mibici.serving import build_features

    events = pd.read_json(log_path, lines=True)
    rows = pd.concat(
        [
            pd.DataFrame(e.entradas).assign(**{PREDICTION: [s["salidas_pred"] for s in e.salidas]})
            for e in events.itertuples()
        ],
        ignore_index=True,
    )
    rows["hora"] = pd.to_datetime(rows["hora"])
    current = build_features(rows, load_stations())
    reference = pd.read_parquet(out_dir / "referencia.parquet")
    columns = {c: k for c, k in monitored_columns().items() if c != TARGET}
    table = drift_table(reference, current, columns)
    table.to_csv(out_dir / "drift_log_peticiones.csv", index=False)
    logger.info(f"{len(events)} peticiones, {len(current)} filas")
    logger.info("\n" + table[["columna", "distancia", "drift", "psi"]].to_string(index=False))


if __name__ == "__main__":
    app()
