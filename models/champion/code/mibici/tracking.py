"""Versionado de datasets con MLflow.

Cada versión es un run del experimento `mibici-fase1-datos` que guarda:
  * el dataset como input (mlflow.log_input): esquema, nombre, origen y digest
  * parámetros: versión, versión padre, descripción y reglas aplicadas
  * métricas: filas, columnas, % de nulos, tamaño y conteos de la transformación
  * tag sha256 del archivo parquet, para detectar cualquier cambio en los datos
  * artefactos: resumen por columna (JSON) y una muestra de 200 filas (CSV)
"""

import hashlib
import json
from pathlib import Path
import tempfile
import warnings

import mlflow
import pandas as pd

from mibici import config

# Avisos informativos de MLflow que no aplican a datasets locales en parquet
warnings.filterwarnings("ignore", message="The specified dataset source")
warnings.filterwarnings("ignore", message="Hint: Inferred schema contains integer")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def column_summary(df: pd.DataFrame) -> dict:
    return {
        col: {"dtype": str(df[col].dtype), "nulos": int(df[col].isna().sum()),
              "unicos": int(df[col].nunique())}
        for col in df.columns
    }  # fmt: skip


def log_dataset_version(
    df: pd.DataFrame,
    path: Path,
    name: str,
    version: str,
    description: str,
    parent: str | None = None,
    params: dict | None = None,
    metrics: dict | None = None,
    artifacts: list[Path] | None = None,
) -> str:
    """Registra una versión del dataset en MLflow y regresa el run_id."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.MLFLOW_EXPERIMENT)
    with mlflow.start_run(run_name=f"{name}-{version}") as run:
        dataset = mlflow.data.from_pandas(df, source=str(path), name=f"{name}-{version}")
        mlflow.log_input(dataset, context=version)

        mlflow.log_params(
            {
                "version": version,
                "padre": parent or "ninguno",
                "descripcion": description,
                "archivo": path.name,
                **(params or {}),
            }
        )
        mlflow.log_metrics({
            "filas": len(df), "columnas": df.shape[1],
            "pct_nulos": round(100 * df.isna().mean().mean(), 4),
            "tamano_mb": round(path.stat().st_size / 2**20, 2),
            **(metrics or {}),
        })  # fmt: skip
        mlflow.set_tags({"dataset": name, "version": version, "sha256": file_sha256(path)})

        with tempfile.TemporaryDirectory() as tmp:
            summary = Path(tmp) / f"resumen_{name}_{version}.json"
            summary.write_text(json.dumps(column_summary(df), indent=2, ensure_ascii=False))
            sample = Path(tmp) / f"muestra_{name}_{version}.csv"
            df.sample(min(200, len(df)), random_state=0).to_csv(sample, index=False)
            for file in [summary, sample, *(artifacts or [])]:
                mlflow.log_artifact(str(file))
        return run.info.run_id
