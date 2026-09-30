"""L3 · Escalera de modelos (§8 del plan).

Se construye en orden de complejidad creciente. Cada peldano solo se justifica
si supera al anterior con significancia estadistica (ver `validate.stats`):

    nivel 0  umbral       regla de alerta tipo NWS sobre el exceso de indice de calor
    nivel 1  logistica    GLM con regularizacion L2
    nivel 2  bosque       Random Forest
    nivel 2  lightgbm     gradient boosting (candidato principal del plan)

El nivel 0 es la hipotesis nula operativa: si nada supera a "alertar cuando el
indice de calor excede el p95 local", el gemelo no aporta sobre lo que ya hay.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260923


class ThresholdRule(ClassifierMixin, BaseEstimator):
    """Nivel 0: alerta si `feature` supera un umbral.

    El umbral se elige en entrenamiento maximizando el indice de Youden
    (sensibilidad + especificidad - 1): es la DECISION de alerta (`predict`).

    La PROBABILIDAD es una logistica de una sola variable sobre `feature`:
    creciente, asi que ordena los dias igual que la regla (mas exceso de calor,
    mas riesgo). Una version anterior devolvia la frecuencia a cada lado del
    umbral; esos dos escalones cambian de altura entre pliegues y, al juntar
    las predicciones fuera de muestra de todos los pliegues, el AUC global
    castigaba a la regla por un artefacto de escala y no por ordenar peor.
    """

    def __init__(self, feature: str = "hi_excess_p95"):
        self.feature = feature

    def fit(self, X, y):  # noqa: N803
        x = np.asarray(X[self.feature], float)
        y = np.asarray(y)
        fpr, tpr, thr = roc_curve(y, x)
        k = int(np.argmax(tpr - fpr))
        self.threshold_ = float(thr[k]) if np.isfinite(thr[k]) else float(np.median(x))
        above = x >= self.threshold_
        self.p_above_ = float(y[above].mean()) if above.any() else float(y.mean())
        self.p_below_ = float(y[~above].mean()) if (~above).any() else float(y.mean())
        self.link_ = LogisticRegression(C=1e6, max_iter=1000).fit(x[:, None], y)
        if self.link_.coef_[0, 0] < 0:  # la regla supone "mas calor, mas riesgo"
            self.link_.coef_[:] = 0.0
            self.link_.intercept_[:] = np.log(y.mean() / (1 - y.mean()))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):  # noqa: N803
        x = np.asarray(X[self.feature], float)
        return self.link_.predict_proba(x[:, None])

    def predict(self, X):  # noqa: N803
        return (np.asarray(X[self.feature], float) >= self.threshold_).astype(int)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    nivel: int
    descripcion: str
    factory: Callable[[], Any]
    #: distribuciones para la busqueda aleatoria; vacio = sin hiperparametros
    param_space: dict[str, Any] = field(default_factory=dict)
    n_iter: int = 20
    #: metodo de explicacion nativo: "coef", "shap_tree" o "regla"
    explicacion: str = "shap_tree"


def _lgbm():
    from lightgbm import LGBMClassifier

    return LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1)


LADDER: tuple[ModelSpec, ...] = (
    ModelSpec(
        "umbral", 0,
        "Regla de alerta: indice de calor por encima de un umbral local (tipo NWS).",
        lambda: ThresholdRule("hi_excess_p95"),
        explicacion="regla",
    ),
    ModelSpec(
        "logistica", 1,
        "Regresion logistica con regularizacion L2 sobre variables estandarizadas.",
        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        {
            "logisticregression__C": loguniform(1e-3, 1e2),
            "logisticregression__class_weight": [None, "balanced"],
        },
        n_iter=15,
        explicacion="coef",
    ),
    ModelSpec(
        "bosque", 2,
        "Random Forest: promedio de arboles de decision sobre remuestreos.",
        lambda: RandomForestClassifier(random_state=SEED, n_jobs=-1),
        {
            "n_estimators": randint(150, 500),
            "max_depth": [4, 6, 8, 12, None],
            "min_samples_leaf": randint(5, 80),
            "max_features": ["sqrt", 0.5, 0.8],
            "class_weight": [None, "balanced_subsample"],
        },
    ),
    ModelSpec(
        "lightgbm", 2,
        "LightGBM: arboles de gradient boosting, candidato principal del plan (§8).",
        _lgbm,
        {
            "n_estimators": randint(100, 600),
            "learning_rate": loguniform(0.01, 0.2),
            "num_leaves": randint(7, 63),
            "min_child_samples": randint(20, 200),
            "subsample": uniform(0.6, 0.4),
            "subsample_freq": [1],
            "colsample_bytree": uniform(0.5, 0.5),
            "reg_lambda": loguniform(1e-3, 10),
        },
    ),
)

LADDER_BY_NAME = {m.name: m for m in LADDER}
