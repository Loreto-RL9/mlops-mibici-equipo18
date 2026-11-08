"""Prueba de punta a punta del Model Registry con una base de MLflow temporal: registrar el
Pipeline, elegir el campeón y volver a cargarlo para pronosticar (sin descargas ni mlflow.db)."""

import matplotlib
import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pytest

from mibici import config
from mibici.modeling.predict import load_model, predict
from mibici.modeling.train import fit_and_evaluate, log_and_register, set_champion

matplotlib.use("Agg")  # el CI no tiene pantalla


@pytest.fixture
def registry(tmp_path, monkeypatch):
    """MLflow apuntando a una base SQLite en una carpeta temporal, no a la del proyecto."""
    previous, uri = mlflow.get_tracking_uri(), f"sqlite:///{tmp_path / 'mlflow.db'}"
    monkeypatch.setattr(config, "MLFLOW_TRACKING_URI", uri)
    monkeypatch.chdir(tmp_path)  # los artefactos (mlruns/) también quedan en la carpeta temporal
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment("prueba-registry")
    yield MlflowClient()
    mlflow.set_tracking_uri(previous)


def _register(name, params, parts):
    with mlflow.start_run(run_name=name):
        model, metrics, predictions = fit_and_evaluate(name, params, parts, "24h")
        version = log_and_register(model, name, params, metrics, predictions, parts, "24h")
    return model, metrics, version


def test_registered_version_keeps_params_and_metrics(registry, parts):
    _, metrics, version = _register("hgb_poisson", {"max_iter": 30}, parts)
    tags = registry.get_model_version(config.REGISTERED_MODEL, version).tags
    assert tags["modelo"] == "hgb_poisson"
    assert tags["horizonte"] == "24h"
    assert tags["param_max_iter"] == "30"
    assert float(tags["mae_valid"]) == pytest.approx(metrics["mae_valid"], abs=1e-4)


def test_champion_is_the_lowest_validation_deviance_and_predicts_like_the_trained_model(
    registry, parts
):
    glm, glm_metrics, _ = _register("poisson_glm", {"alpha": 1e-3}, parts)
    hgb, hgb_metrics, _ = _register("hgb_poisson", {"max_iter": 30}, parts)
    best_name, best_model = min(
        [("poisson_glm", glm, glm_metrics), ("hgb_poisson", hgb, hgb_metrics)],
        key=lambda m: m[2]["devianza_poisson_valid"],
    )[:2]

    assert set_champion("24h").startswith(best_name)
    champion = load_model("champion")
    # El Pipeline cargado del Registry trae su preprocesamiento: recibe las columnas crudas
    assert np.allclose(predict(champion, parts["test"]), predict(best_model, parts["test"]))
