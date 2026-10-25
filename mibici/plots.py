"""Figuras del análisis exploratorio (se guardan en reports/figures).

Comando:
    python -m mibici.plots
"""

import json

import joblib
from loguru import logger
import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer

from mibici import config
from mibici.dataset import load_stations

app = typer.Typer()

ACCENT, GRAY, RED = "#1f6f78", "#9aa5ad", "#e4572e"
SEQUENTIAL = LinearSegmentedColormap.from_list("mibici", ["#ffffff", ACCENT, "#0f3d3e"])
DIVERGING = LinearSegmentedColormap.from_list("mibici_div", [ACCENT, "#f2f2f2", RED])
DAYS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]
MONTHS = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]
RULE_LABELS = {
    "duplicados": "Duplicados",
    "fecha_invalida": "Fecha inválida",
    "duracion_no_positiva": "Duración ≤ 0",
    "arranque_fallido": "Misma estación y < 2 min",
    "duracion_menor_1min": "Duración < 1 min",
    "duracion_mayor_180min": "Duración > 180 min",
    "fuera_de_horario": "Fuera de horario (1–4 h)",
    "estacion_fuera_de_catalogo": "Estación fuera del catálogo",
}
plt.rcParams.update({"figure.dpi": 150, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 12, "font.size": 10})  # fmt: skip


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / name, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Figura: {name}")


def plot_cleaning(report: dict) -> None:
    rules = sorted((report[k], label) for k, label in RULE_LABELS.items())
    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.barh([label for _, label in rules], [v for v, _ in rules], color=ACCENT)
    ax.bar_label(bars, labels=[f"{v:,}" for v, _ in rules], padding=3, fontsize=9)
    removed = report["filas_iniciales"] - report["filas_finales"]
    ax.set_title(f"Viajes eliminados por regla ({removed:,} en total, "
                 f"{100 * removed / report['filas_iniciales']:.1f} %)")  # fmt: skip
    ax.set_xlabel("viajes")
    ax.xaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
    ax.set_xlim(0, rules[-1][0] * 1.18)
    save(fig, "fig1_limpieza.png")


def plot_duration(raw: pd.DataFrame) -> None:
    start, end = (pd.to_datetime(raw[c], format="%Y-%m-%d %H:%M:%S", errors="coerce")
                  for c in ("inicio", "fin"))  # fmt: skip
    dur = (end - start).dt.total_seconds() / 60
    dur = dur[dur > 0]
    fig, ax = plt.subplots(figsize=(7, 3.4))
    ax.hist(dur, bins=np.logspace(-2, 4, 120), color=ACCENT)
    ax.set_xscale("log")
    for x, label in [(config.MIN_DURATION_MIN, "1 min"), (config.MAX_DURATION_MIN, "180 min")]:
        ax.axvline(x, color=RED, ls="--", lw=1)
        ax.text(x * 1.1, ax.get_ylim()[1] * 0.9, label, color=RED, fontsize=9)
    ax.set_title(f"Duración de los viajes crudos (mediana {dur.median():.1f} min)")
    ax.set_xlabel("minutos (escala log)")
    ax.set_ylabel("viajes")
    save(fig, "fig2_duracion.png")


def plot_monthly(trips: pd.DataFrame) -> None:
    monthly = trips.groupby(trips["inicio"].dt.to_period("M")).size()
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = [f"{MONTHS[p.month - 1]}\n{p.year % 100}" for p in monthly.index]
    ax.bar(labels, monthly.values / 1000, color=ACCENT)
    ax.set_title("Viajes limpios por mes (miles)")
    ax.set_ylabel("miles de viajes")
    save(fig, "fig3_viajes_mes.png")


def plot_heatmap(panel: pd.DataFrame) -> None:
    grid = panel.groupby([panel["hora"].dt.dayofweek, panel["hora"].dt.hour])["salidas"].sum()
    grid = grid.unstack().reindex(columns=range(24))  # horas sin servicio quedan en blanco
    grid = grid / panel["hora"].dt.normalize().nunique() * 7  # salidas por día de la semana
    fig, ax = plt.subplots(figsize=(8, 3.2))
    im = ax.imshow(grid, aspect="auto", cmap=SEQUENTIAL)
    ax.set_yticks(range(7), DAYS)
    ax.set_xticks(range(0, 24, 2), [f"{h}h" for h in range(0, 24, 2)])
    ax.set_title("Salidas promedio del sistema por día y hora")
    fig.colorbar(im, ax=ax, label="salidas por hora")
    save(fig, "fig4_heatmap_dia_hora.png")


def plot_demand_distribution(panel: pd.DataFrame) -> None:
    counts = panel["salidas"].clip(upper=15).value_counts(normalize=True).sort_index() * 100
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(counts.index, counts.values, color=[RED] + [ACCENT] * (len(counts) - 1))
    ax.set_xticks(range(16), [str(i) for i in range(15)] + ["15+"])
    ax.set_title(f"Salidas por estación-hora: {counts.iloc[0]:.0f} % de las horas son cero")
    ax.set_xlabel("salidas en la hora")
    ax.set_ylabel("% de estación-horas")
    save(fig, "fig5_distribucion_demanda.png")


def plot_top_stations(trips: pd.DataFrame, stations: pd.DataFrame, n: int = 15) -> None:
    total = trips["origen_id"].value_counts()
    names = stations.set_index("estacion_id")["nombre"].str.replace(r"^\S+\s", "", regex=True)
    top = total.head(n)[::-1]
    share = 100 * total.head(int(0.2 * len(total))).sum() / total.sum()
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.barh(names.reindex(top.index).str.slice(0, 38), top.values / 1000, color=ACCENT)
    ax.set_title(f"Top {n} estaciones por salidas (el 20 % de estaciones genera {share:.0f} %)")
    ax.set_xlabel("miles de salidas en 12 meses")
    save(fig, "fig6_top_estaciones.png")


def plot_pca(pca, comps: pd.DataFrame, stations: pd.DataFrame, trips: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    for i, color in [(0, ACCENT), (1, RED)]:
        ax.plot(pca.components_[i], color=color, label=f"PC{i + 1}")
    names = list(pca.feature_names_in_)  # p. ej. "semana_07", "finde_18"
    split = next(i for i, n in enumerate(names) if n.startswith("finde"))
    ticks = [i for i, n in enumerate(names) if int(n[-2:]) in (6, 12, 18)]
    ax.set_xticks(ticks, [f"{int(names[i][-2:])}h" for i in ticks])
    ax.axvline(split - 0.5, color=GRAY, lw=1)
    ax.text(1, ax.get_ylim()[1] * 0.9, "entre semana", color=GRAY)
    ax.text(split + 1, ax.get_ylim()[1] * 0.9, "fin de semana", color=GRAY)
    ax.set_title("Cargas de los componentes sobre el perfil horario")
    ax.legend(frameon=False)
    ax = axes[1]
    ratio = 100 * pca.explained_variance_ratio_
    ax.bar(range(1, len(ratio) + 1), ratio.cumsum(), color=ACCENT)
    ax.axhline(100 * config.PCA_VARIANCE, color=RED, ls="--", lw=1)
    ax.set_title(f"Varianza acumulada ({len(ratio)} PCs)")
    ax.set_xlabel("componentes")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    ax.set_ylabel("%")
    save(fig, "fig7_pca_perfiles.png")

    geo = stations.set_index("estacion_id").join(comps["pc1"], how="inner")
    geo["salidas"] = trips["origen_id"].value_counts()
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    lim = np.abs(geo["pc1"]).quantile(0.95)
    sc = ax.scatter(geo["lon"], geo["lat"], c=geo["pc1"], cmap=DIVERGING, vmin=-lim, vmax=lim,
                    s=8 + geo["salidas"] / geo["salidas"].max() * 120, edgecolor="none")  # fmt: skip
    fig.colorbar(sc, ax=ax, label="PC1 (+ pico vespertino, − pico matutino)", shrink=0.8)
    ax.set_title("PC1 por estación (tamaño = salidas)")
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
    ax.set_xlabel("longitud")
    ax.set_ylabel("latitud")
    ax.set_aspect("equal")
    save(fig, "fig8_mapa_pc1.png")


@app.command()
def main():
    raw = pd.read_parquet(config.INTERIM_DATA_DIR / "trips_raw.parquet", columns=["inicio", "fin"])
    trips = pd.read_parquet(config.INTERIM_DATA_DIR / "trips_clean.parquet")
    panel = pd.read_parquet(config.PROCESSED_DATA_DIR / "demand_panel.parquet")
    stations = load_stations()
    report = json.loads((config.REPORTS_DIR / "reporte_limpieza.json").read_text())
    comps = pd.read_csv(config.MODELS_DIR / "pca_componentes_por_estacion.csv", index_col=0)
    pca = joblib.load(config.MODELS_DIR / "pca_perfiles.joblib")

    plot_cleaning(report)
    plot_duration(raw)
    plot_monthly(trips)
    plot_heatmap(panel)
    plot_demand_distribution(panel)
    plot_top_stations(trips, stations)
    plot_pca(pca, comps, stations, trips)
    logger.success(f"Figuras en {config.FIGURES_DIR}")


if __name__ == "__main__":
    app()
