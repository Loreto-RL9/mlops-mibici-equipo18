"""Pipeline de scikit-learn: preprocesamiento + modelo en un solo objeto.

Todo lo que se aprende de los datos (medias del escalado, categorías del one-hot, perfil de
cada estación y su PCA) se ajusta con fit() únicamente sobre entrenamiento. Como el Pipeline
completo se guarda en el Model Registry, la inferencia aplica exactamente las mismas
transformaciones que el entrenamiento.

Etapas:
  1. "ciclicas"         seno y coseno de la hora y del día de la semana
  2. "numericas"        StandardScaler sobre capacidad, ubicación y rezagos
  3. "municipio"        one-hot de GDL, ZPN y TLQ
  4. "perfil_estacion"  StationProfilePCA: la estación como sus componentes PCA
  5. "binarias"         fin de semana y festivo, sin cambios
  6. "modelo"           PoissonRegressor o HistGradientBoostingRegressor (pérdida Poisson)
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import PoissonRegressor
from sklearn.metrics import mean_absolute_error, mean_poisson_deviance, root_mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from mibici import config

CYCLIC_PERIODS = {"hora_dia": 24, "dia_semana": 7}
PROFILE_COLUMNS = ["estacion_id", "fin_de_semana", "hora_dia"]


def numeric_columns(horizon: str = config.HORIZON) -> list[str]:
    """Variables numéricas disponibles al momento de pronosticar. Con 24 h de anticipación
    no se conocen la hora anterior ni la media de las últimas 24 h."""
    columns = ["capacidad", "lat", "lon", "lag_24h", "lag_168h"]
    return columns + ["lag_1h", "media_24h"] if horizon == "1h" else columns


def input_columns(horizon: str = config.HORIZON) -> list[str]:
    """Columnas de v4 que recibe el Pipeline (sin la variable objetivo)."""
    columns = [*CYCLIC_PERIODS, *numeric_columns(horizon), "municipio", *PROFILE_COLUMNS,
               "fin_de_semana", "festivo"]  # fmt: skip
    return list(dict.fromkeys(columns))  # sin duplicados, en orden


def cyclic_encode(X: pd.DataFrame) -> np.ndarray:
    """Seno y coseno de cada columna: las 23 h quedan junto a las 0 h y el domingo junto al lunes."""
    parts = []
    for column, period in CYCLIC_PERIODS.items():
        angle = 2 * np.pi * X[column].to_numpy() / period
        parts += [np.sin(angle), np.cos(angle)]
    return np.column_stack(parts)


def cyclic_names(transformer, input_features) -> list[str]:
    return [f"{c}_{f}" for c in CYCLIC_PERIODS for f in ("sin", "cos")]


class StationProfilePCA(TransformerMixin, BaseEstimator):
    """Representa cada estación con su patrón de uso en lugar de 469 columnas one-hot.

    fit: calcula el perfil de cada estación (salidas medias por hora de servicio, entre semana
    y en fin de semana, divididas entre su media) y lo reduce con PCA hasta explicar `variance`.
    Usa la variable objetivo, igual que una codificación por objetivo, por eso solo debe
    ajustarse con entrenamiento (el Pipeline lo garantiza).
    transform: devuelve los componentes de la estación; una estación nueva recibe el perfil
    promedio (ceros).
    """

    def __init__(self, variance: float = config.PCA_VARIANCE, random_state=config.RANDOM_STATE):
        self.variance = variance
        self.random_state = random_state

    def fit(self, X: pd.DataFrame, y):
        data = X[PROFILE_COLUMNS].assign(salidas=np.asarray(y))
        profiles = data.groupby(PROFILE_COLUMNS)["salidas"].mean().unstack([1, 2])
        profiles.columns = [f"{'finde' if w else 'semana'}_{h:02d}" for w, h in profiles.columns]
        profiles = profiles.div(profiles.mean(axis=1), axis=0).fillna(0)
        self.pca_ = PCA(self.variance, svd_solver="full", random_state=self.random_state)
        codes = self.pca_.fit_transform(profiles)
        self.codes_ = pd.DataFrame(codes, index=profiles.index)
        self.n_components_ = self.pca_.n_components_
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return self.codes_.reindex(X["estacion_id"].to_numpy()).fillna(0).to_numpy()

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array([f"pc{i + 1}" for i in range(self.n_components_)])


def build_preprocessor(horizon: str = config.HORIZON) -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("ciclicas", FunctionTransformer(cyclic_encode, feature_names_out=cyclic_names),
             list(CYCLIC_PERIODS)),
            ("numericas", StandardScaler(), numeric_columns(horizon)),
            ("municipio", OneHotEncoder(handle_unknown="ignore", sparse_output=False),
             ["municipio"]),
            ("perfil_estacion", StationProfilePCA(), PROFILE_COLUMNS),
            ("binarias", "passthrough", ["fin_de_semana", "festivo"]),
        ],
        verbose_feature_names_out=False,
    )  # fmt: skip


# Modelos candidatos: ambos optimizan la verosimilitud Poisson, adecuada para conteos
MODELS = {
    # Lineal generalizado con enlace log: base interpretable y rápida
    "poisson_glm": lambda params: PoissonRegressor(max_iter=500, **params),
    # Árboles con boosting: capturan interacciones (estación x hora x día) y escalan a millones
    # de filas
    "hgb_poisson": lambda params: HistGradientBoostingRegressor(
        loss="poisson", random_state=config.RANDOM_STATE, **params
    ),
}


def build_pipeline(model_name: str, params: dict, horizon: str = config.HORIZON) -> Pipeline:
    return Pipeline(
        [("preprocesamiento", build_preprocessor(horizon)), ("modelo", MODELS[model_name](params))]
    )


def evaluate(y_true, y_pred, top_mask=None) -> dict:
    """Métricas de evaluación. MAE: salidas por hora de error (la métrica de negocio);
    RMSE: castiga los errores grandes en horas pico; desviación Poisson: ajuste del conteo;
    sesgo: si el modelo sobre o subestima en promedio."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred, dtype=float)
    metrics = {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        # la desviación Poisson exige pronósticos > 0 (una persistencia puede dar 0)
        "desviacion_poisson": mean_poisson_deviance(y_true, np.clip(y_pred, 1e-6, None)),
        "sesgo": float(np.mean(y_pred - y_true)),
    }
    if top_mask is not None:  # estaciones con más demanda, donde un error cuesta más
        metrics["mae_top20"] = mean_absolute_error(y_true[top_mask], y_pred[top_mask])
    return metrics
