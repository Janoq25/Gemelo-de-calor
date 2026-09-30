"""L3 · Entrenamiento, validacion cruzada anidada y seleccion del mejor modelo.

Esquema de validacion (§10.2)
-----------------------------
Validacion cruzada ANIDADA: el ciclo externo mide el desempeno, el interno
elige hiperparametros. Si se eligieran hiperparametros con los mismos datos
con que se reporta el AUC, el AUC saldria optimista.

* Externo, segun los anos con desenlace disponibles:
  - >= 3 anos: origen movil por ano (entrenar con anos < k, probar en k).
  - < 3 anos: dejar fuera una region HHS completa (10 pliegues). Mide si el
    modelo transfiere a estados que nunca vio, que es la pregunta del gemelo.
* Interno: GroupKFold por estado sobre el entrenamiento del pliegue externo,
  para que un mismo estado nunca este a ambos lados de la busqueda.
* Chequeo temporal adicional: entrenar con mayo-julio, dejar 7 dias de hueco
  y probar desde el 8 de agosto. El hueco evita que las medias moviles de 3
  dias crucen la frontera.

La CV aleatoria por filas esta prohibida aqui: pondria el 14 de julio de
Texas en prueba y el 13 y el 15 en entrenamiento.

Regla de seleccion
------------------
Parsimonia con significancia, aplicando los criterios de exito del negocio:

A. AUC global fuera de muestra no significativamente peor que el mejor
   (DeLong + Holm, alfa 0,05).
B. Ranking dentro de cada region no significativamente peor que el mejor
   (Friedman + Nemenyi sobre el AUC por pliegue).
C. Probabilidades calibradas (Spiegelhalter, p >= 0,05).

Gana el modelo MAS SIMPLE que cumple A, B y C. Si ninguno cumple C, el mas
simple que cumple A y B, marcado para recalibracion en L5. Una diferencia que
no supera las pruebas no justifica la complejidad extra.

Nota de trazabilidad: la primera corrida (23/9/2026) seleccionaba solo con A.
B y C se anadieron tras ver que el modelo elegido incumplia el criterio de
calibracion declarado en la ficha del problema; ambos ya eran criterios de
exito, no metricas nuevas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import GroupKFold, RandomizedSearchCV, StratifiedKFold

from xdt.features.panel import Panel
from xdt.models.ladder import LADDER, SEED, ModelSpec
from xdt.validate import stats as st


@dataclass
class Fold:
    name: str
    train: np.ndarray
    test: np.ndarray


@dataclass
class ModelResult:
    spec: ModelSpec
    oof: np.ndarray  # probabilidad fuera de muestra, alineada con el panel
    fold_scores: pd.DataFrame  # una fila por pliegue externo
    best_params: list[dict[str, Any]]  # por pliegue
    search_tables: list[pd.DataFrame]  # cv_results_ de la busqueda interna
    temporal_auc: float = float("nan")
    temporal_pred: np.ndarray | None = None
    #: modelo ajustado en cada pliegue externo; produce la prediccion OOF
    fold_models: list[Any] = field(default_factory=list)


@dataclass
class TrainingRun:
    panel: Panel
    folds: list[Fold]
    cv_scheme: str
    results: dict[str, ModelResult]
    temporal_split: dict[str, Any]
    selection: dict[str, Any] = field(default_factory=dict)
    final_models: dict[str, Any] = field(default_factory=dict)
    final_params: dict[str, dict[str, Any]] = field(default_factory=dict)


def outer_folds(panel: Panel) -> tuple[list[Fold], str]:
    f = panel.frame
    years = sorted(f["year"].unique())
    if len(years) >= 3:
        folds = [
            Fold(str(yr), np.flatnonzero(f["year"] < yr), np.flatnonzero(f["year"] == yr))
            for yr in years[1:]
        ]
        return folds, "origen movil por ano"
    folds = []
    for r in sorted(f["hhs_region"].dropna().unique()):
        te = np.flatnonzero(f["hhs_region"] == r)
        tr = np.flatnonzero(f["hhs_region"] != r)
        if f["y"].iloc[te].nunique() == 2:
            folds.append(Fold(f"region {int(r)}", tr, te))
    return folds, "dejar fuera una region HHS"


def temporal_split(panel: Panel, gap_days: int = 7) -> tuple[np.ndarray, np.ndarray, dict]:
    """Entrenar antes del 1 de agosto (del ultimo ano), probar tras un hueco."""
    d = panel.frame.index.get_level_values("valid_date")
    last = d.year.max()
    cut = pd.Timestamp(f"{last}-08-01")
    start_test = cut + pd.Timedelta(days=gap_days)
    tr = np.flatnonzero(d < cut)
    te = np.flatnonzero(d >= start_test)
    info = {"corte": cut.date().isoformat(), "inicio_prueba": start_test.date().isoformat(),
            "hueco_dias": gap_days, "n_train": len(tr), "n_test": len(te)}
    return tr, te, info


def _fit(spec: ModelSpec, X: pd.DataFrame, y: pd.Series, groups: np.ndarray,
         n_iter: int | None, inner_splits: int) -> tuple[Any, dict, pd.DataFrame | None]:
    est = spec.factory()
    if not spec.param_space:
        return est.fit(X, y), {}, None
    n_groups = len(np.unique(groups))
    # Con un solo estado en entrenamiento no hay grupos que separar; se cae a
    # KFold estratificado (solo ocurre en pruebas con 2 estados).
    cv = (GroupKFold(n_splits=min(inner_splits, n_groups)) if n_groups >= 2
          else StratifiedKFold(inner_splits, shuffle=True, random_state=SEED))
    search = RandomizedSearchCV(
        est, spec.param_space, n_iter=n_iter or spec.n_iter, scoring="roc_auc",
        cv=cv, random_state=SEED,
        n_jobs=1, refit=True, error_score="raise",
    )
    search.fit(X, y, groups=groups if n_groups >= 2 else None)
    table = pd.DataFrame(search.cv_results_)
    return search.best_estimator_, search.best_params_, table


def _scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    return {
        "auc": roc_auc_score(y, p),
        "ap": average_precision_score(y, p),
        "brier": brier_score_loss(y, p),
        "prevalencia": float(np.mean(y)),
    }


def run_training(
    panel: Panel,
    *,
    models: tuple[ModelSpec, ...] = LADDER,
    n_iter: int | None = None,
    inner_splits: int = 3,
    log=print,
) -> TrainingRun:
    X, y = panel.X, panel.y.to_numpy()
    states = panel.frame.index.get_level_values("geoid").to_numpy()
    folds, scheme = outer_folds(panel)
    tr_t, te_t, tinfo = temporal_split(panel)

    results: dict[str, ModelResult] = {}
    for spec in models:
        oof = np.full(len(y), np.nan)
        rows, params, tables, fitted = [], [], [], []
        for fold in folds:
            model, bp, table = _fit(spec, X.iloc[fold.train], y[fold.train],
                                    states[fold.train], n_iter, inner_splits)
            p = model.predict_proba(X.iloc[fold.test])[:, 1]
            oof[fold.test] = p
            rows.append({"pliegue": fold.name, "n_train": len(fold.train),
                         "n_test": len(fold.test), **_scores(y[fold.test], p)})
            params.append(bp)
            fitted.append(model)
            if table is not None:
                tables.append(table.assign(pliegue=fold.name))
        res = ModelResult(spec, oof, pd.DataFrame(rows), params, tables, fold_models=fitted)

        model, _, _ = _fit(spec, X.iloc[tr_t], y[tr_t], states[tr_t], n_iter, inner_splits)
        res.temporal_pred = model.predict_proba(X.iloc[te_t])[:, 1]
        res.temporal_auc = roc_auc_score(y[te_t], res.temporal_pred)
        results[spec.name] = res
        log(f"  {spec.name:<10} AUC CV = {res.fold_scores['auc'].mean():.3f} "
            f"| AUC temporal = {res.temporal_auc:.3f}")

    run = TrainingRun(panel, folds, scheme, results,
                      {**tinfo, "idx_test": te_t, "idx_train": tr_t})
    run.selection = select_best(run)

    # Modelos finales: busqueda interna sobre TODO el panel y reajuste. Son
    # los que se explican (SHAP) y los que el gemelo usaria para predecir.
    for spec in models:
        m, bp, _ = _fit(spec, X, y, states, n_iter, inner_splits)
        run.final_models[spec.name] = m
        run.final_params[spec.name] = bp
    return run


def select_best(run: TrainingRun) -> dict[str, Any]:
    y = run.panel.y.to_numpy()
    covered = ~np.isnan(next(iter(run.results.values())).oof)
    yy = y[covered]
    auc = {k: st.delong_ci(yy, r.oof[covered]) for k, r in run.results.items()}
    best = max(auc, key=lambda k: auc[k][0])
    pvals, tests = {}, {}
    for k, r in run.results.items():
        if k == best:
            continue
        t = st.delong_test(yy, run.results[best].oof[covered], r.oof[covered])
        tests[k] = t
        pvals[f"{best} vs {k}"] = t.p_value
    holm = st.holm(pvals) if pvals else pd.DataFrame()
    not_worse = [best] + [
        k for k in tests
        if not holm.loc[holm["comparacion"] == f"{best} vs {k}", "rechaza_H0"].iloc[0]
    ]

    # Filtro B: ranking DENTRO de cada region (Friedman + Nemenyi). El AUC
    # global premia tambien distinguir estados entre si; este filtro exige
    # ademas no ordenar peor los dias dentro de una region que no se vio.
    fold_auc = pd.DataFrame({k: r.fold_scores.set_index("pliegue")["auc"]
                             for k, r in run.results.items()})
    try:
        fr = st.friedman_nemenyi(fold_auc)
        top_rank = min(fr.mean_ranks.values())
        nemenyi_ok = [k for k, v in fr.mean_ranks.items()
                      if v - top_rank <= fr.critical_difference]
    except ValueError:  # < 3 modelos o sin bloques
        fr, nemenyi_ok = None, list(run.results)

    # Filtro C: calibracion (criterio de exito 2). Spiegelhalter sin rechazo.
    calib = {k: st.calibration(yy, r.oof[covered]) for k, r in run.results.items()}
    calibrated = [k for k, c in calib.items() if c.spiegelhalter_p >= 0.05]

    gate_ab = [k for k in not_worse if k in nemenyi_ok]
    gate_abc = [k for k in gate_ab if k in calibrated]
    pool = gate_abc or gate_ab or not_worse
    chosen = min(pool, key=lambda k: (run.results[k].spec.nivel, -auc[k][0]))
    return {"auc_ci": auc, "mejor_auc": best, "delong": tests, "holm": holm,
            "no_peores": not_worse, "nemenyi_ok": nemenyi_ok, "friedman": fr,
            "calibrados": calibrated, "calibracion": calib,
            "filtro_ab": gate_ab, "filtro_abc": gate_abc,
            "requiere_recalibracion": chosen not in calibrated,
            "elegido": chosen, "mask": covered}
