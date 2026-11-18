"""Exporta un modelo del Model Registry a una carpeta que la API y el contenedor pueden servir.

El servicio no lee la base de MLflow: recibe una copia exacta del artefacto registrado (el mismo
Pipeline que guardó train.py) y un model_meta.json con su versión, sus métricas y de dónde salió.
Así la imagen del contenedor es autocontenida y cada respuesta de la API dice qué versión la produjo.

Comando:
    python -m mibici.modeling.export                       # @champion -> models/champion
    python -m mibici.modeling.export --alias challenger --out-dir models/challenger
"""

from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

from loguru import logger
import mlflow
from mlflow.tracking import MlflowClient
import typer

from mibici import config

app = typer.Typer()


def model_metadata(alias: str) -> dict:
    """Versión, run, etiquetas (hiperparámetros y métricas) y fechas de la versión con el alias."""
    version = MlflowClient().get_model_version_by_alias(config.REGISTERED_MODEL, alias)
    registered = datetime.fromtimestamp(version.creation_timestamp / 1000, tz=timezone.utc)
    return {
        "nombre": config.REGISTERED_MODEL,
        "alias": alias,
        "version": str(version.version),
        "run_id": version.run_id,
        "registrado": registered.isoformat(timespec="seconds"),
        "exportado": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "datos_entrenamiento_hasta": config.TRAIN_END,
        "horizonte": version.tags.get("horizonte", config.HORIZON),
        "etiquetas": dict(version.tags),
    }


@app.command()
def main(alias: str = config.MODEL_ALIAS, out_dir: Path = config.SERVING_MODEL_DIR):
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    meta = model_metadata(alias)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    mlflow.artifacts.download_artifacts(
        artifact_uri=f"models:/{config.REGISTERED_MODEL}@{alias}", dst_path=str(out_dir)
    )
    (out_dir / "model_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    logger.success(
        f"{config.REGISTERED_MODEL} versión {meta['version']} (@{alias}, "
        f"{meta['etiquetas'].get('modelo', '?')}) -> {out_dir}"
    )


if __name__ == "__main__":
    app()
