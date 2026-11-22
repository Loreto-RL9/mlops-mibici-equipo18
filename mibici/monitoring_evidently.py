"""Reporte de drift con Evidently (DataDriftPreset) sobre las muestras que guarda
`python -m mibici.monitoring drift`.

Corre en un entorno aislado porque Evidently 0.7.23 (la versión más reciente) no es compatible con
pandas 3, que es la versión fijada en uv.lock. Por eso este archivo no importa el paquete mibici:

    make drift-evidently
    # equivale a:
    uv run --no-project --python 3.12 --with "evidently==0.7.23" --with "pandas<3" \
        --with pyarrow python mibici/monitoring_evidently.py

Genera reports/monitoreo/evidently_<escenario>.html y un resumen con la proporción de columnas con
drift según Evidently, para contrastarlo con nuestra implementación (misma regla por omisión).
"""

import json
from pathlib import Path
import sys

from evidently import DataDefinition, Dataset, Report
from evidently.presets import DataDriftPreset
import pandas as pd

NUMERIC = ["capacidad", "lat", "lon", "lag_24h", "lag_168h", "hora_dia", "salidas_pred", "salidas"]
CATEGORICAL = ["municipio", "dia_semana", "fin_de_semana", "festivo"]


def drifted_share(snapshot) -> dict:
    """Busca en el snapshot la métrica DriftedColumnsCount (cuántas y qué proporción)."""
    for metric in snapshot.dict()["metrics"]:
        if "DriftedColumnsCount" in str(metric.get("metric_name", metric.get("metric_id", ""))):
            return metric["value"]
    return {}


def main(folder: Path) -> None:
    reference = pd.read_parquet(folder / "referencia.parquet")
    definition = DataDefinition(numerical_columns=NUMERIC, categorical_columns=CATEGORICAL)
    ref = Dataset.from_pandas(reference[NUMERIC + CATEGORICAL], data_definition=definition)
    summary = {}
    for path in sorted(folder.glob("actual_*.parquet")):
        name = path.stem.removeprefix("actual_")
        current = pd.read_parquet(path)[NUMERIC + CATEGORICAL]
        cur = Dataset.from_pandas(current, data_definition=definition)
        snapshot = Report([DataDriftPreset()]).run(cur, ref)
        snapshot.save_html(str(folder / f"evidently_{name}.html"))
        summary[name] = drifted_share(snapshot)
        print(f"{name:>18}: {summary[name]}")
    (folder / "evidently_resumen.json").write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path("reports/monitoreo"))
