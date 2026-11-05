"""Pruebas de la selección de modelos con datos sintéticos pequeños (sin MLflow ni descargas)."""

import numpy as np
import optuna
import pandas as pd
import pytest

from mibici.modeling.tune import MODELS, SPACES, evaluate, top_stations


def test_evaluate_known_values_and_zero_predictions():
    m = evaluate([0, 2, 4], [1, 2, 2], top_mask=[False, False, True])
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(np.sqrt(5 / 3))
    assert m["sesgo"] == pytest.approx(-1 / 3)  # subestima en promedio
    assert m["mae_top_estaciones"] == pytest.approx(2.0)
    # Un bosque puede predecir 0 exacto: la devianza de Poisson debe seguir siendo finita
    assert np.isfinite(evaluate([0, 3], [0.0, 3.0])["devianza_poisson"])


def test_top_stations_keeps_highest_demand():
    train = pd.DataFrame({"estacion_id": [1, 1, 2, 3, 4, 5], "salidas": [9, 9, 1, 2, 3, 4]})
    assert top_stations(train, share=0.2) == {1}
    assert top_stations(train, share=0.4) == {1, 5}


@pytest.mark.parametrize("name", list(MODELS))
def test_each_family_fits_with_sampled_params(name):
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(3000, 4)), columns=list("abcd"))
    y = rng.poisson(np.exp(0.5 + 0.8 * X["a"]))
    params = SPACES[name](optuna.trial.FixedTrial(_middle_of_space(name)))
    pred = MODELS[name](params).fit(X, y).predict(X)
    assert pred.shape == (3000,)
    assert (pred >= 0).all()  # conteos: nunca negativos
    assert evaluate(y, pred)["mae"] < np.abs(y - y.mean()).mean()  # mejor que predecir la media


def _middle_of_space(name):
    return {
        "poisson_glm": {"alpha": 1e-3},
        "random_forest": {"n_estimators": 50, "max_depth": 8, "min_samples_leaf": 100,
                          "max_features": 0.8},
        "hgb_poisson": {"learning_rate": 0.1, "max_iter": 200, "max_leaf_nodes": 15,
                        "min_samples_leaf": 20, "l2_regularization": 0.01},
    }[name]  # fmt: skip
