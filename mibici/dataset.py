"""Ingesta y limpieza de los viajes de MiBici.

Comandos:
    python -m mibici.dataset download   # descarga los 12 CSV mensuales y el catálogo
    python -m mibici.dataset clean      # v0 (crudo) y v1 (limpio), registrados en MLflow
"""

import json
from pathlib import Path
import urllib.request

from loguru import logger
import pandas as pd
import typer

from mibici import config
from mibici.tracking import log_dataset_version

app = typer.Typer()

TRIP_COLUMNS = [
    "viaje_id", "usuario_id", "genero", "anio_nacimiento",
    "inicio", "fin", "origen_id", "destino_id",
]  # fmt: skip


def read_trips_csv(path: Path) -> pd.DataFrame:
    """Lee un CSV mensual como texto. Enero 2026 viene en latin-1; el resto en utf-8."""
    for encoding in ("utf-8", "latin-1"):
        try:
            df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding=encoding)
            break
        except UnicodeDecodeError:
            logger.warning(f"{path.name}: no es utf-8, se lee como latin-1")
    if df.shape[1] != len(TRIP_COLUMNS):
        raise ValueError(f"{path.name}: se esperaban {len(TRIP_COLUMNS)} columnas")
    df.columns = TRIP_COLUMNS
    return df


def load_raw(raw_dir: Path = config.RAW_DATA_DIR) -> pd.DataFrame:
    files = sorted(raw_dir.glob("datos_abiertos_*.csv"))
    if not files:
        raise FileNotFoundError(f"No hay CSV en {raw_dir}; ejecuta `make download`")
    return pd.concat([read_trips_csv(f) for f in files], ignore_index=True)


def load_stations(path: Path = config.STATIONS_FILE) -> pd.DataFrame:
    """Catálogo de estaciones. El municipio viene en el nombre: '(GDL-001) ...'."""
    st = pd.read_csv(path).rename(
        columns={"id": "estacion_id", "name": "nombre", "latitude": "lat",
                 "longitude": "lon", "dpcapacity": "capacidad"}
    )  # fmt: skip
    st["municipio"] = st["nombre"].str.extract(r"(GDL|ZPN|TLQ)", expand=False)
    st["retirada"] = st["nombre"].str.startswith("ex-")
    return st


class TripCleaner:
    """Aplica las reglas de limpieza en orden y registra cuántos viajes elimina cada una."""

    def __init__(self, station_ids):
        self.station_ids = set(station_ids)
        self.report: dict[str, int] = {}

    def _drop(self, rule: str, df: pd.DataFrame, mask: pd.Series) -> pd.DataFrame:
        mask = mask.fillna(False).astype(bool)
        self.report[rule] = int(mask.sum())
        return df.loc[~mask]

    @staticmethod
    def cast_types(raw: pd.DataFrame) -> pd.DataFrame:
        df = raw.copy()
        for col in ("inicio", "fin"):
            df[col] = pd.to_datetime(df[col], format="%Y-%m-%d %H:%M:%S", errors="coerce")
        for col in ("viaje_id", "usuario_id", "anio_nacimiento", "origen_id", "destino_id"):
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        df["duracion_min"] = (df["fin"] - df["inicio"]).dt.total_seconds() / 60
        return df

    def clean(self, raw: pd.DataFrame) -> pd.DataFrame:
        self.report = {"filas_iniciales": len(raw)}
        df = self.cast_types(raw)
        dur = "duracion_min"
        df = self._drop("duplicados", df, df.duplicated("viaje_id"))
        df = self._drop("fecha_invalida", df, df["inicio"].isna() | df["fin"].isna())
        df = self._drop("duracion_no_positiva", df, df[dur] <= 0)
        same = df["origen_id"] == df["destino_id"]
        df = self._drop("arranque_fallido", df, same & (df[dur] < config.SAME_STATION_MAX_MIN))
        df = self._drop("duracion_menor_1min", df, df[dur] < config.MIN_DURATION_MIN)
        df = self._drop("duracion_mayor_180min", df, df[dur] > config.MAX_DURATION_MIN)
        unknown = ~df["origen_id"].isin(self.station_ids) | ~df["destino_id"].isin(
            self.station_ids
        )
        df = self._drop("estacion_fuera_de_catalogo", df, unknown)

        # Correcciones que no eliminan filas
        df["genero"] = df["genero"].replace({"NULL": "Desconocido", "": "Desconocido"})
        lo, hi = config.BIRTH_YEAR_RANGE
        bad_year = df["anio_nacimiento"].notna() & ~df["anio_nacimiento"].between(lo, hi)
        self.report["anio_nacimiento_a_nulo"] = int(bad_year.sum())
        df.loc[bad_year.astype(bool), "anio_nacimiento"] = pd.NA

        self.report["filas_finales"] = len(df)
        check_clean_trips(df, self.station_ids)
        return df.sort_values("inicio").reset_index(drop=True)


def check_clean_trips(df: pd.DataFrame, station_ids) -> None:
    """Validación automática del dataset limpio: falla si alguna regla no se cumple."""
    assert df["viaje_id"].is_unique, "hay viajes duplicados"
    assert (df["fin"] > df["inicio"]).all(), "hay viajes que terminan antes de empezar"
    assert df["duracion_min"].between(config.MIN_DURATION_MIN, config.MAX_DURATION_MIN).all()
    assert df["origen_id"].isin(station_ids).all(), "origen fuera del catálogo"
    assert df["destino_id"].isin(station_ids).all(), "destino fuera del catálogo"


@app.command()
def download(raw_dir: Path = config.RAW_DATA_DIR):
    """Descarga los CSV mensuales y el catálogo de estaciones (omite los que ya existen)."""
    urls = [f"{config.BASE_URL}/{folder}/datos_abiertos_{month}.csv"
            for month, folder in config.TRIP_FILES.items()] + [config.STATIONS_URL]  # fmt: skip
    for url in urls:
        target = raw_dir / url.rsplit("/", 1)[-1]
        if not target.exists():
            logger.info(f"Descargando {target.name}")
            urllib.request.urlretrieve(url, target)
    logger.success(f"{len(urls)} archivos en {raw_dir}")


@app.command()
def clean(
    raw_dir: Path = config.RAW_DATA_DIR,
    interim_dir: Path = config.INTERIM_DATA_DIR,
):
    """Genera v0 (viajes crudos) y v1 (viajes limpios) y los versiona en MLflow."""
    raw = load_raw(raw_dir)
    stations = load_stations()
    raw_path = interim_dir / "trips_raw.parquet"
    raw.to_parquet(raw_path, index=False)
    log_dataset_version(
        raw, raw_path, name="viajes", version="v0",
        description="12 CSV mensuales concatenados, sin cambios (todo como texto)",
        params={"meses": len(config.TRIP_FILES), "fuente": config.BASE_URL},
    )  # fmt: skip

    cleaner = TripCleaner(stations["estacion_id"])
    trips = cleaner.clean(raw)
    clean_path = interim_dir / "trips_clean.parquet"
    trips.to_parquet(clean_path, index=False)
    report_path = config.REPORTS_DIR / "reporte_limpieza.json"
    report_path.write_text(json.dumps(cleaner.report, indent=2, ensure_ascii=False))
    log_dataset_version(
        trips, clean_path, name="viajes", version="v1", parent="v0",
        description="Tipos corregidos y reglas de limpieza aplicadas en orden",
        params={"min_duracion_min": config.MIN_DURATION_MIN,
                "max_duracion_min": config.MAX_DURATION_MIN,
                "arranque_fallido_min": config.SAME_STATION_MAX_MIN,
                "rango_anio_nacimiento": str(config.BIRTH_YEAR_RANGE)},
        metrics=cleaner.report,
        artifacts=[report_path],
    )  # fmt: skip
    for rule, n in cleaner.report.items():
        logger.info(f"{rule:>28}: {n:,}")
    logger.success(f"v1: {len(trips):,} viajes limpios -> {clean_path}")


if __name__ == "__main__":
    app()
