"""L5 · Calibracion retrospectiva (§6 del plan). "Esta capa define el twin."

Cada vez que llega una observacion nueva (EPHT publica un ano, llega la serie
diaria), el gemelo recupera las predicciones que habia emitido para ese
periodo desde L0, calcula el error, actualiza las metricas de calibracion y
registra si hubo deriva. Sin esto hay un simulador de un solo sentido.

Tres reglas, fijadas aqui y no en cada analisis:

1. **Solo se calibran pronosticos genuinos.** Una prediccion cuenta si el
   estado que uso (`state_known_at`, o `issued_at` si falta) es ANTERIOR a
   la primera vez que el gemelo conocio la observacion. Si no, el modelo
   pudo haberla visto: es una post-diccion, se excluye y se cuenta. Es la
   razon de ser del eje bitemporal de L1 y del campo `state_known_at` de L0.

2. **La censura entra como intervalo, no se descarta.** Una celda suprimida
   dice "el valor esta entre lo y hi". El error de esa pareja es la
   distancia de la prediccion al intervalo (cero si cae dentro). Descartarla
   sesgaria la calibracion hacia las zonas de mayor incidencia (§14 R2).
   El sesgo medio y la pendiente, que necesitan un valor puntual, se
   calculan solo sobre observaciones no censuradas, y se dice cuantas son.

3. **La deriva se declara con reglas fijadas a priori**, no mirando la
   grafica (§17: pre-registro):
     - sesgo sistematico: |sesgo| significativo (|t| > 1.96, n >= 10) Y
       materialmente grande (> `rel_bias_tol` de la media observada). Solo la
       significacion marcaria deriva con cualquier sesgo minusculo en n grande.
     - degradacion: MAE > (1 + `mae_tol`) x mediana del MAE de las ultimas
       `reference_runs` corridas del mismo modelo y cantidad.

Solo se evaluan predicciones del escenario base (`scenario_id IS NULL`): un
contrafactual no tiene observacion posible (§15.6).
"""

from __future__ import annotations

import datetime as dt
import json
from dataclasses import asdict, dataclass, field
from typing import Any

import duckdb
import numpy as np
import pandas as pd

from xdt.hashing import stable_id
from xdt.twin.state import now_utc

Z_95 = 1.96


@dataclass(frozen=True)
class CalibrationResult:
    run_id: str
    ran_at: dt.datetime
    model_name: str
    model_version: str | None
    quantity: str
    window_start: dt.date | None
    window_end: dt.date | None
    obs_known_at: dt.datetime
    n_pairs: int
    n_censored: int
    n_postdiction: int
    metrics: dict[str, float | None]
    drift_reasons: list[str] = field(default_factory=list)

    @property
    def drift_flag(self) -> bool:
        return bool(self.drift_reasons)


class Calibrator:
    """Reune predicciones (L0) con observaciones (L0/L5) y mide la calibracion."""

    def __init__(self, con: duckdb.DuckDBPyConnection):
        self.con = con

    # ------------------------------------------------------------- parejas
    def pairs(
        self,
        *,
        model_name: str,
        quantity: str,
        model_version: str | None = None,
        window_start: dt.date | str | None = None,
        window_end: dt.date | str | None = None,
        obs_known_at: dt.datetime | None = None,
    ) -> tuple[pd.DataFrame, int]:
        """Parejas (prediccion, observacion) validas y numero de post-dicciones.

        `obs_known_at` reproduce una calibracion pasada: solo se usan las
        observaciones (y revisiones) que el gemelo conocia en ese instante.
        Por cada objetivo se conserva el pronostico genuino emitido mas tarde:
        es el que un usuario habria consultado.
        """
        cutoff = obs_known_at or now_utc()
        clauses = ["p.model_name = ?", "p.quantity = ?", "p.scenario_id IS NULL"]
        filt: list[Any] = [model_name, quantity]
        if model_version:
            clauses.append("p.model_version = ?")
            filt.append(model_version)
        if window_start:
            clauses.append("p.target_date >= ?")
            filt.append(pd.to_datetime(window_start).date())
        if window_end:
            clauses.append("p.target_date <= ?")
            filt.append(pd.to_datetime(window_end).date())
        where = " AND ".join(clauses)

        df = self.con.execute(
            f"""
            WITH obs AS (
                SELECT *,
                       min(known_at) OVER (PARTITION BY obs_key) AS first_known_at,
                       row_number() OVER (PARTITION BY obs_key ORDER BY revision DESC) AS rn
                FROM observation
                WHERE known_at <= ?
            ),
            o AS (SELECT * FROM obs WHERE rn = 1),
            joined AS (
                SELECT p.prediction_id, p.geo_level, p.geoid, p.target_date,
                       p.model_version, p.issued_at,
                       COALESCE(p.state_known_at, p.issued_at) AS input_cutoff,
                       p.value AS pred, p.value_lower AS pred_lo, p.value_upper AS pred_hi,
                       o.value AS obs, o.censored, o.censor_lo, o.censor_hi,
                       o.first_known_at, o.revision AS obs_revision
                FROM prediction_registry p
                JOIN o ON o.geo_level = p.geo_level AND o.geoid = p.geoid
                      AND o.target_date = p.target_date AND o.quantity = p.quantity
                WHERE {where} AND p.issued_at <= ?
            )
            SELECT *, input_cutoff < first_known_at AS genuine FROM joined
            """,
            # la prediccion tampoco puede ser posterior al corte que se reproduce
            [cutoff, *filt, cutoff],
        ).df()

        if df.empty:
            return df, 0
        n_post = int((~df["genuine"]).sum())
        df = df[df["genuine"]]
        df = (
            df.sort_values("issued_at")
            .groupby(["geo_level", "geoid", "target_date", "model_version"], as_index=False)
            .tail(1)
            .reset_index(drop=True)
        )
        return df, n_post

    # ------------------------------------------------------------- corrida
    def run(
        self,
        *,
        model_name: str,
        quantity: str,
        model_version: str | None = None,
        window_start: dt.date | str | None = None,
        window_end: dt.date | str | None = None,
        obs_known_at: dt.datetime | None = None,
        reference_runs: int = 3,
        mae_tol: float = 0.25,
        rel_bias_tol: float = 0.10,
        persist: bool = True,
    ) -> CalibrationResult:
        cutoff = obs_known_at or now_utc()
        pairs, n_post = self.pairs(
            model_name=model_name, quantity=quantity, model_version=model_version,
            window_start=window_start, window_end=window_end, obs_known_at=cutoff,
        )
        metrics = calibration_metrics(pairs)

        reasons: list[str] = []
        if metrics["n_point"] and metrics["n_point"] >= 10 and metrics["bias_t"] is not None:
            scale = abs(metrics["obs_mean"] or 0.0)
            if abs(metrics["bias_t"]) > Z_95 and scale > 0 and \
                    abs(metrics["bias"]) > rel_bias_tol * scale:
                reasons.append(
                    f"sesgo sistematico {metrics['bias']:+.3g} "
                    f"(t={metrics['bias_t']:.2f}, {abs(metrics['bias']) / scale:.0%} de la media)"
                )
        ref = self._reference_mae(model_name, quantity, model_version, reference_runs)
        if ref is not None and metrics["mae"] is not None and \
                metrics["mae"] > (1.0 + mae_tol) * ref:
            reasons.append(
                f"MAE {metrics['mae']:.3g} supera en >{mae_tol:.0%} la referencia {ref:.3g}"
            )
        metrics["reference_mae"] = ref

        ran_at = now_utc()
        ws = pd.to_datetime(window_start).date() if window_start else None
        we = pd.to_datetime(window_end).date() if window_end else None
        result = CalibrationResult(
            run_id=stable_id("calibration", model_name, model_version, quantity, ws, we,
                             cutoff, ran_at),
            ran_at=ran_at,
            model_name=model_name,
            model_version=model_version,
            quantity=quantity,
            window_start=ws,
            window_end=we,
            obs_known_at=cutoff,
            n_pairs=len(pairs),
            n_censored=int(pairs["censored"].sum()) if not pairs.empty else 0,
            n_postdiction=n_post,
            metrics=metrics,
            drift_reasons=reasons,
        )
        if persist:
            self._persist(result)
        return result

    def history(self, model_name: str | None = None) -> pd.DataFrame:
        where, params = ("WHERE model_name = ?", [model_name]) if model_name else ("", [])
        return self.con.execute(
            f"SELECT * FROM calibration_run {where} ORDER BY ran_at", params
        ).df()

    # ----------------------------------------------------------- internos
    def _reference_mae(
        self, model_name: str, quantity: str, model_version: str | None, k: int
    ) -> float | None:
        hist = self.history(model_name)
        if hist.empty or k <= 0:
            return None
        meta = hist["metrics_json"].map(json.loads)
        hist = hist.assign(
            quantity=meta.map(lambda m: m.get("quantity")),
            mae=meta.map(lambda m: m.get("mae")),
        )
        hist = hist[(hist["quantity"] == quantity) & hist["mae"].notna()]
        if model_version:
            hist = hist[hist["model_version"] == model_version]
        if hist.empty:
            return None
        return float(np.median(hist.tail(k)["mae"]))

    def _persist(self, r: CalibrationResult) -> None:
        body = {
            **r.metrics,
            "quantity": r.quantity,
            "obs_known_at": r.obs_known_at.isoformat(),
            "n_censored": r.n_censored,
            "n_postdiction": r.n_postdiction,
            "drift_reasons": r.drift_reasons,
        }
        self.con.execute(
            "INSERT INTO calibration_run VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [r.run_id, r.ran_at, r.model_name, r.model_version or "*", r.window_start,
             r.window_end, r.n_pairs, json.dumps(body, sort_keys=True, default=str),
             r.drift_flag],
        )


def calibration_metrics(pairs: pd.DataFrame) -> dict[str, float | None]:
    """Metricas de calibracion para una cantidad continua (tasa o conteo).

    `mae` usa todas las parejas (censuradas como distancia al intervalo);
    `bias`, `rmse`, pendiente e intercepto solo las puntuales.
    """
    empty: dict[str, float | None] = {
        "n_point": 0, "mae": None, "bias": None, "bias_t": None, "rmse": None,
        "obs_mean": None, "calib_slope": None, "calib_intercept": None,
        "poisson_deviance": None, "censored_hit_rate": None, "pi_coverage": None,
    }
    if pairs.empty:
        return empty

    cens = pairs["censored"].fillna(False).astype(bool).to_numpy()
    pred = pairs["pred"].to_numpy(float)
    obs = pairs["obs"].to_numpy(float)
    lo = pairs["censor_lo"].to_numpy(float)
    hi = pairs["censor_hi"].to_numpy(float)

    abs_err = np.abs(pred - obs)
    interval_err = np.maximum(np.nan_to_num(lo, nan=-np.inf) - pred, 0.0) + \
        np.maximum(pred - np.nan_to_num(hi, nan=np.inf), 0.0)
    abs_err[cens] = interval_err[cens]
    point = ~cens & ~np.isnan(obs)

    out = dict(empty)
    out["mae"] = float(np.nanmean(abs_err)) if len(abs_err) else None
    if cens.any():
        out["censored_hit_rate"] = float(np.mean(interval_err[cens] == 0.0))

    n = int(point.sum())
    out["n_point"] = n
    if n:
        resid = pred[point] - obs[point]
        out["bias"] = float(resid.mean())
        out["rmse"] = float(np.sqrt(np.mean(resid**2)))
        out["obs_mean"] = float(obs[point].mean())
        if n >= 2 and resid.std(ddof=1) > 0:
            out["bias_t"] = float(resid.mean() / (resid.std(ddof=1) / np.sqrt(n)))
        if n >= 3 and np.var(pred[point]) > 0:
            slope, intercept = np.polyfit(pred[point], obs[point], 1)
            out["calib_slope"], out["calib_intercept"] = float(slope), float(intercept)
        y, mu = obs[point], pred[point]
        if (y >= 0).all() and (mu > 0).all():
            with np.errstate(divide="ignore", invalid="ignore"):
                term = np.where(y > 0, y * np.log(y / mu), 0.0)
            out["poisson_deviance"] = float(np.mean(2.0 * (term - (y - mu))))
        plo = pairs["pred_lo"].to_numpy(float)[point]
        phi = pairs["pred_hi"].to_numpy(float)[point]
        has_pi = ~np.isnan(plo) & ~np.isnan(phi)
        if has_pi.any():
            yy = obs[point][has_pi]
            out["pi_coverage"] = float(np.mean((yy >= plo[has_pi]) & (yy <= phi[has_pi])))
    return out


def result_dict(r: CalibrationResult) -> dict[str, Any]:
    d = asdict(r)
    d["drift_flag"] = r.drift_flag
    return d
