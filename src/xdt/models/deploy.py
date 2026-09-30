"""L3 -> gemelo: despliegue del modelo elegido (§6).

Cierra el circuito que el reporte CRISP-DM deja abierto:

1. **Persistencia.** El modelo elegido se guarda con una version determinista
   (hash de nombre, hiperparametros, version de codigo y corte bitemporal).
2. **L0.** Cada prediccion se registra con su version, hash de entrada, corte
   bitemporal y, en `extras`, las contribuciones SHAP que la explican.
3. **L4.** Los escenarios de calentamiento (+1, +2, +3 °C) se registran como
   predicciones con `scenario_id`, sobre el mismo modelo y el mismo estado.
4. **Observaciones.** El desenlace observado entra al almacen de L5.

Honestidad retrospectiva
------------------------
Las predicciones son FUERA DE MUESTRA: cada estado-dia lo predice el modelo
del pliegue que no vio su region. Pero se emiten despues de conocer el
desenlace, asi que L5 las cuenta como post-dicciones, no como pronosticos
genuinos. Es lo correcto: la calibracion genuina empieza cuando el gemelo
prediga antes de que EPHT publique.
"""

from __future__ import annotations

import datetime as dt
import json
from pathlib import Path
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from xdt.explain import xai
from xdt.features.panel import FEATURES, OUTCOME_VAR, with_warming
from xdt.hashing import stable_id
from xdt.models.train import TrainingRun
from xdt.storage import code_version
from xdt.twin.registry import ObservationStore, Prediction, PredictionRegistry

#: Cantidad predicha: P(al menos una visita por HRI ese dia en la poblacion VA).
QUANTITY = "p_hri_ed_any_daily_va"
SCENARIOS_C = (1.0, 2.0, 3.0)
TOP_K = 5


def scenario_id(delta_c: float) -> str:
    return f"calentamiento+{delta_c:g}C"


def _input_hash(X: pd.DataFrame) -> str:
    return f"{int(pd.util.hash_pandas_object(X, index=True).sum()) & (2**64 - 1):016x}"


def oof_shap(run: TrainingRun, name: str) -> tuple[np.ndarray, np.ndarray] | None:
    """SHAP fuera de muestra: cada fila explicada por el modelo que la predijo."""
    res = run.results[name]
    if res.spec.explicacion == "regla":
        return None
    X = run.panel.X
    contrib = np.full(X.shape, np.nan)
    base = np.full(len(X), np.nan)
    for fold, model in zip(run.folds, res.fold_models, strict=True):
        expl, _ = xai.shap_values(model, X.iloc[fold.test], max_rows=None,
                                  background=X.iloc[fold.train])
        contrib[fold.test] = expl.values
        base[fold.test] = np.broadcast_to(np.asarray(expl.base_values).reshape(-1),
                                          len(fold.test))
    return contrib, base


def oof_scenarios(run: TrainingRun, name: str,
                  deltas: tuple[float, ...] = SCENARIOS_C) -> dict[float, np.ndarray]:
    """Probabilidad bajo cada escenario, con el mismo modelo de pliegue que la OOF."""
    res = run.results[name]
    out: dict[float, np.ndarray] = {}
    for d in deltas:
        Xd = with_warming(run.panel, d)
        p = np.full(len(Xd), np.nan)
        for fold, model in zip(run.folds, res.fold_models, strict=True):
            p[fold.test] = model.predict_proba(Xd.iloc[fold.test])[:, 1]
        out[d] = p
    return out


def deploy(run: TrainingRun, con: duckdb.DuckDBPyConnection, *, models_dir: Path,
           report_path: str, log=print) -> dict[str, Any]:
    import joblib

    sel = run.selection
    name = sel["elegido"]
    res = run.results[name]
    panel = run.panel
    X = panel.X
    idx = panel.frame.index
    covered = sel["mask"]

    version = stable_id(name, json.dumps(run.final_params.get(name, {}), sort_keys=True,
                                         default=str),
                        code_version(), panel.known_at.isoformat())[:12]
    models_dir.mkdir(parents=True, exist_ok=True)
    model_file = models_dir / f"{name}-{version}.joblib"
    joblib.dump(run.final_models[name], model_file)

    shap_oof = oof_shap(run, name)
    scen = oof_scenarios(run, name)
    in_hash = _input_hash(X)
    known = panel.known_at.to_pydatetime().replace(tzinfo=None)
    feats = list(FEATURES)

    preds: list[Prediction] = []
    fold_of = np.empty(len(X), dtype=object)
    for fold in run.folds:
        fold_of[fold.test] = fold.name
    for i in np.flatnonzero(covered):
        geoid, day = idx[i]
        extras: dict[str, Any] = {"pliegue": fold_of[i]}
        if shap_oof is not None:
            c = shap_oof[0][i]
            order = np.argsort(-np.abs(c))[:TOP_K]
            extras["shap_top"] = [[feats[j], round(float(c[j]), 4)] for j in order]
            extras["shap_base"] = round(float(shap_oof[1][i]), 4)
        common = dict(geo_level="state", geoid=geoid, target_date=day.date(),
                      quantity=QUANTITY, model_name=name, model_version=version,
                      input_hash=in_hash, state_known_at=known, horizon_days=0)
        preds.append(Prediction(value=float(res.oof[i]), extras=extras, **common))
        for d, p in scen.items():
            preds.append(Prediction(value=float(p[i]), scenario_id=scenario_id(d),
                                    extras={"delta_c": d, "pliegue": fold_of[i]}, **common))
    n_pred = PredictionRegistry(con).log(preds)

    obs = pd.DataFrame({
        "geo_level": "state",
        "geoid": idx.get_level_values("geoid"),
        "target_date": pd.to_datetime(idx.get_level_values("valid_date")),
        "quantity": QUANTITY,
        "value": panel.y.to_numpy().astype(float),
    })
    # La observacion se sella con el instante en que el gemelo conocio el
    # desenlace (su primer known_at en L1), NO con la hora de este despliegue:
    # si no, L5 tomaria por pronosticos genuinos predicciones emitidas despues
    # de conocer el resultado.
    first_known = con.execute(
        """
        SELECT geoid, valid_date AS target_date, min(known_at) AS known_at
        FROM twin_state WHERE source = 'epht' AND variable = ? AND geo_level = 'state'
        GROUP BY 1, 2
        """,
        [OUTCOME_VAR],
    ).df()
    first_known["target_date"] = pd.to_datetime(first_known["target_date"])
    obs = obs.merge(first_known, on=["geoid", "target_date"], how="left")
    if obs["known_at"].isna().any():
        raise ValueError("desenlace sin known_at en L1: no se puede sellar la observacion")
    store = ObservationStore(con)
    obs_stats = {"received": 0, "written": 0, "unchanged": 0}
    for ka, grp in obs.groupby("known_at"):
        st_ = store.put(grp.drop(columns="known_at"), source="epht",
                        known_at=pd.Timestamp(ka).to_pydatetime())
        obs_stats = {k: obs_stats[k] + st_[k] for k in obs_stats}

    metrics = {}
    for k, r in run.results.items():
        cal = sel["calibracion"][k]
        metrics[k] = {
            "nivel": r.spec.nivel,
            "auc_global": round(sel["auc_ci"][k][0], 4),
            "auc_global_ic95": [round(sel["auc_ci"][k][1], 4), round(sel["auc_ci"][k][2], 4)],
            "auc_region_media": round(float(r.fold_scores["auc"].mean()), 4),
            "auc_temporal": round(float(r.temporal_auc), 4),
            "brier": round(float(np.mean((r.oof[covered] - panel.y.to_numpy()[covered]) ** 2)), 4),
            "pendiente_calibracion": round(cal.slope, 3),
            "p_spiegelhalter": round(cal.spiegelhalter_p, 4),
        }
    meta = {
        "modelo": name,
        "version": version,
        "archivo": model_file.name,
        "entrenado_en": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "corte_bitemporal": panel.known_at.isoformat(),
        "version_codigo": code_version(),
        "cantidad": QUANTITY,
        "variables": feats,
        "anios_referencia": list(panel.ref_years),
        "esquema_cv": run.cv_scheme,
        "n_estado_dias": int(len(X)),
        "prevalencia": round(float(panel.y.mean()), 4),
        "seleccion": {
            "mejor_auc": sel["mejor_auc"],
            "filtro_a_auc_global": sel["no_peores"],
            "filtro_b_ranking_region": sel["nemenyi_ok"],
            "filtro_c_calibrados": sel["calibrados"],
            "requiere_recalibracion": bool(sel["requiere_recalibracion"]),
        },
        "metricas": metrics,
        "escenarios": [scenario_id(d) for d in SCENARIOS_C],
        "reporte": report_path,
        "registro_l0": {"predicciones_nuevas": n_pred, **obs_stats},
    }
    (models_dir / "latest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    log(f"desplegado {name} v{version}: {n_pred} predicciones nuevas en L0, "
        f"{obs_stats['written']} observaciones")
    return meta
