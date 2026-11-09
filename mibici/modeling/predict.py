"""Pronósticos con un modelo del Model Registry (por defecto, el alias "champion").

Comando:
    python -m mibici.modeling.predict                    # pronostica septiembre 2026 (prueba)
    python -m mibici.modeling.predict --particion valid --alias champion
"""

from pathlib import Path

from loguru import logger
import mlflow
import pandas as pd
import typer

from mibici import config
from mibici.modeling.train import evaluate

app = typer.Typer()


def load_model(alias: str = config.MODEL_ALIAS):
    """Carga el Pipeline completo (preprocesamiento + modelo) desde el Model Registry."""
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    return mlflow.sklearn.load_model(f"models:/{config.REGISTERED_MODEL}@{alias}")


def predict(model, df: pd.DataFrame) -> pd.Series:
    """Usa solo las columnas con las que se entrenó el Pipeline."""
    return pd.Series(model.predict(df[list(model.feature_names_in_)]), index=df.index)


@app.command()
def main(
    input_path: Path = config.PROCESSED_DATA_DIR / "model_input.parquet",
    output_path: Path = config.PROCESSED_DATA_DIR / "predicciones.parquet",
    particion: str = "test",
    alias: str = config.MODEL_ALIAS,
):
    df = pd.read_parquet(input_path)
    df = df[df["particion"] == particion].reset_index(drop=True)
    df["salidas_pred"] = predict(load_model(alias), df)
    df[["estacion_id", "hora", "salidas", "salidas_pred"]].to_parquet(output_path, index=False)
    metrics = evaluate(df["salidas"], df["salidas_pred"])
    logger.info(", ".join(f"{k}={v:.3f}" for k, v in metrics.items()))
    logger.success(f"{len(df):,} pronósticos ({particion}, @{alias}) -> {output_path}")


if __name__ == "__main__":
    app()
