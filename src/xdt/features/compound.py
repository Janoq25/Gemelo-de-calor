"""M2 · Evento compuesto calor-apagon (§9.2 del plan).

Definicion operativa, fijada aqui para que sea la misma en la estimacion (RERI,
PI2) y en la simulacion (L4):

    dia compuesto = dia caliente  (tmmx > p95 local de temporada calida)
                  ∧ apagon relevante (>= `min_frac` de clientes sin luz
                                      durante >= `min_hours` seguidas)

`min_frac` y `min_hours` son la X y la Y del plan. Sus valores por defecto
(1 %, 4 h) son una eleccion declarada, no estimada; el analisis de
sensibilidad debe barrerlos antes de reportar cualquier efecto.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from xdt.features.thermal import above_threshold, local_percentile, streak

DEFAULT_MIN_FRAC = 0.01
DEFAULT_MIN_HOURS = 4.0


def compound_features(
    frame: pd.DataFrame,
    *,
    ref_start: dt.date | str,
    ref_end: dt.date | str,
    q: float = 95.0,
    min_frac: float = DEFAULT_MIN_FRAC,
    min_hours: float = DEFAULT_MIN_HOURS,
    var_tmax: str = "tmmx",
    var_frac: str = "outage_customers_frac",
    var_hours: str = "outage_max_hours",
) -> pd.DataFrame:
    """Banderas diarias de calor, apagon y evento compuesto.

    Devuelve `hot`, `hot_streak`, `outage`, `compound`. Un dia sin dato de
    apagon queda con `outage`/`compound` en NaN: la ausencia de EAGLE-I no es
    evidencia de que hubo luz.
    """
    missing = [v for v in (var_tmax, var_frac, var_hours) if v not in frame.columns]
    if missing:
        raise KeyError(f"faltan variables {missing} para el evento compuesto")

    thr = local_percentile(frame, var_tmax, q, ref_start=ref_start, ref_end=ref_end)
    hot = above_threshold(frame, var_tmax, thr).rename("hot")
    out = pd.DataFrame({"hot": hot, "hot_streak": streak(hot).to_numpy()}, index=frame.index)

    has_outage_data = frame[var_frac].notna() & frame[var_hours].notna()
    outage = ((frame[var_frac] >= min_frac) & (frame[var_hours] >= min_hours)).astype(float)
    out["outage"] = outage.where(has_outage_data)
    out["compound"] = (out["hot"] * out["outage"]).where(has_outage_data & hot.notna())
    return out


def compound_summary(flags: pd.DataFrame) -> pd.DataFrame:
    """Tabla 2x2 por geografia: dias calientes / con apagon / compuestos.

    Incluye el cociente observado/esperado bajo independencia. Es descriptivo:
    un O/E > 1 dice que calor y apagon coinciden mas de lo que el azar
    predice, no que uno cause el otro ni cuanto riesgo anaden (eso es RERI,
    con desenlace).
    """
    f = flags.dropna(subset=["hot", "outage"])
    g = f.groupby(level="geoid")
    s = pd.DataFrame({
        "days": g.size(),
        "hot_days": g["hot"].sum(),
        "outage_days": g["outage"].sum(),
        "compound_days": g["compound"].sum(),
    })
    expected = s["hot_days"] * s["outage_days"] / s["days"]
    s["expected_compound"] = expected
    s["obs_over_exp"] = s["compound_days"] / expected.where(expected > 0)
    return s
