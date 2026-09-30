"""L3 · Explicabilidad (§8, "Explicabilidad").

Tres niveles, del mas global al mas local:

* **Global**: que variables mueven el riesgo en promedio (|SHAP| medio,
  odds ratios de la logistica, importancia por permutacion).
* **Forma del efecto**: como cambia el riesgo al variar una variable
  (dependencia SHAP, dependencia parcial).
* **Local**: por que el modelo marco ESE estado ESE dia (cascada SHAP).

SHAP explica al MODELO, no al mundo: una contribucion alta del indice de calor
significa que el modelo se apoya en el, no que el calor cause la visita. El
reporte lo dice en cada grafico (§15.6).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def shap_values(model: Any, X: pd.DataFrame, max_rows: int | None = 3000, seed: int = 0,
                background: pd.DataFrame | None = None):
    """Valores SHAP de la clase positiva, en escala log-odds.

    Devuelve (Explanation, X_muestra). Usa TreeExplainer (exacto) para arboles
    y LinearExplainer para la logistica. `background` es la referencia contra
    la que se mide cada contribucion (para la logistica, los datos con que se
    entreno); por defecto, las mismas filas explicadas.
    """
    import shap

    big = max_rows is not None and len(X) > max_rows
    Xs = X.sample(max_rows, random_state=seed) if big else X
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if hasattr(est, "coef_"):
        scaler = model.steps[0][1]
        Z = pd.DataFrame(scaler.transform(Xs), columns=Xs.columns, index=Xs.index)
        bg = Z if background is None else pd.DataFrame(
            scaler.transform(background), columns=Xs.columns)
        expl = shap.LinearExplainer(est, bg)(Z)
        expl.data = Xs.to_numpy()  # ejes en unidades originales
    else:
        expl = shap.TreeExplainer(est)(Xs)
        if expl.values.ndim == 3:  # RandomForest: (filas, variables, clases)
            expl = expl[:, :, 1]
    expl.feature_names = list(Xs.columns)
    return expl, Xs


def global_importance(expl) -> pd.Series:
    return pd.Series(np.abs(expl.values).mean(axis=0), index=expl.feature_names) \
        .sort_values(ascending=False)


def odds_ratios(model: Any, feature_names: list[str]) -> pd.DataFrame:
    """Odds ratio por +1 desviacion estandar de cada variable (logistica)."""
    scaler, lr = model.steps[0][1], model.steps[-1][1]
    beta = lr.coef_[0]
    return pd.DataFrame({
        "variable": feature_names,
        "coef_std": beta,
        "odds_ratio_1sd": np.exp(beta),
        "desv_std": scaler.scale_,
    }).sort_values("odds_ratio_1sd", key=lambda s: np.abs(np.log(s)), ascending=False)


def permutation_importance_table(model: Any, X: pd.DataFrame, y: np.ndarray,
                                 n_repeats: int = 10, seed: int = 0) -> pd.DataFrame:
    """Caida del AUC al barajar cada variable: independiente del tipo de modelo."""
    from sklearn.inspection import permutation_importance

    r = permutation_importance(model, X, y, scoring="roc_auc", n_repeats=n_repeats,
                               random_state=seed, n_jobs=1)
    return pd.DataFrame({"variable": X.columns, "caida_auc": r.importances_mean,
                         "desv": r.importances_std}).sort_values("caida_auc", ascending=False)
