"""Pruebas del monitoreo: las medidas de drift detectan cambios conocidos y no dan falsas alarmas,
y las alertas y acciones siguen las reglas documentadas en references/monitoreo.md."""

import numpy as np
import pandas as pd
import pytest

from mibici import config
from mibici.monitoring import (
    THRESHOLD,
    BrokenFeed,
    DemandSurge,
    StationClosures,
    alerts,
    categorical_drift,
    drift_table,
    numeric_drift,
    proposed_action,
    psi,
    segment_performance,
)

rng = np.random.default_rng(0)


@pytest.fixture
def same():
    return pd.Series(rng.poisson(2, 20_000)), pd.Series(rng.poisson(2, 20_000))


def test_sin_cambio_no_hay_drift(same):
    reference, current = same
    assert numeric_drift(reference, current) < THRESHOLD
    assert psi(reference, current) < 0.1


def test_desplazamiento_de_medias_es_drift(same):
    reference, _ = same
    shifted = pd.Series(rng.poisson(3, 20_000))
    assert numeric_drift(reference, shifted) > THRESHOLD
    assert psi(reference, shifted) > 0.1


def test_drift_categorico():
    reference = pd.Series(["GDL"] * 70 + ["ZPN"] * 20 + ["TLQ"] * 10)
    assert categorical_drift(reference, reference.sample(frac=1, random_state=1)) == 0
    assert categorical_drift(reference, pd.Series(["TLQ"] * 60 + ["GDL"] * 40)) > THRESHOLD


def test_tabla_de_drift_marca_solo_la_columna_que_cambio(data):
    current = data.assign(lag_168h=data["lag_168h"] * 3)
    table = drift_table(data, current, {"lag_168h": "num", "lag_24h": "num", "municipio": "cat"})
    assert table.set_index("columna")["drift"].to_dict() == {
        "lag_168h": True,
        "lag_24h": False,
        "municipio": False,
    }


def test_escenarios_sinteticos(data):
    # 12 estaciones para que el 30 % y el 15 % elijan al menos una
    data = pd.concat(
        [data.assign(estacion_id=data["estacion_id"] + 10 * k) for k in range(4)],
        ignore_index=True,
    )
    gen = np.random.default_rng(config.SEED)
    surge = DemandSurge("x", "").apply(data, gen)
    assert surge["salidas"].mean() > data["salidas"].mean()
    broken = BrokenFeed("x", "").apply(data, gen)
    assert (broken["lag_168h"] == 0).mean() > (data["lag_168h"] == 0).mean()
    assert broken["salidas"].equals(data["salidas"])  # la demanda real no cambia
    closed = StationClosures("x", "").apply(data, gen)
    assert closed["lag_168h"].equals(data["lag_168h"])  # las entradas no cambian
    assert closed["salidas"].sum() <= data["salidas"].sum()


def test_alertas_contra_la_linea_base():
    baseline = {"mae": 1.0, "sesgo": 0.0, "mae_persistencia": 1.2}
    ok = alerts({"mae": 1.1, "sesgo": 0.05, "mae_persistencia": 1.2}, baseline)
    bad = alerts({"mae": 1.3, "sesgo": -0.3, "mae_persistencia": 1.2}, baseline)
    assert not ok["alerta_desempeno"] and not ok["alerta_sesgo"]
    assert bad["alerta_desempeno"] and bad["alerta_sesgo"] and bad["alerta_persistencia_gana"]


@pytest.mark.parametrize(
    "row, expected",
    [
        ({"alerta_desempeno": True, "alerta_sesgo": True, "drift_etiqueta": True,
          "columnas_con_drift": 2}, "reentrenar"),
        ({"alerta_desempeno": False, "alerta_sesgo": True, "drift_etiqueta": False,
          "columnas_con_drift": 0}, "drift de concepto"),
        ({"alerta_desempeno": False, "alerta_sesgo": False, "drift_etiqueta": False,
          "columnas_con_drift": 2}, "pipeline de features"),
        ({"alerta_desempeno": False, "alerta_sesgo": False, "drift_etiqueta": False,
          "columnas_con_drift": 0}, "sin acción"),
    ],
)  # fmt: skip
def test_accion_propuesta(row, expected):
    assert expected in proposed_action(row)


def test_politica_de_arranque_en_frio(data):
    """En las estaciones nuevas la política usa la persistencia; en las vistas, el modelo."""
    df = data.assign(salidas_pred=data["salidas"] + 1.0)
    out = segment_performance(df, seen={1, 2}).set_index("estacion_nueva")
    assert out.loc[False, "mae_politica"] == out.loc[False, "mae_modelo"]
    assert out.loc[True, "mae_politica"] == out.loc[True, "mae_persistencia"]
