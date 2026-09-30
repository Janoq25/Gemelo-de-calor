"""M2 · Caracteristicas termicas (§7 del plan).

Operan sobre `EstadoGemelo.frame` (indice (geoid, valid_date), una columna
por variable) y devuelven columnas nuevas con el mismo indice.

Regla de fuga: todo umbral "local" (percentil 95, normal climatica) se
calcula SOLO sobre un periodo de referencia explicito. Calcularlo sobre la
serie completa meteria el clima de 2024 en el umbral con el que se juzga
2022, y la validacion temporal (§10.2) dejaria de medir lo que dice medir.
"""

from __future__ import annotations

import datetime as dt

import numpy as np
import pandas as pd


def heat_index_c(tmax_c: pd.Series, rh_pct: pd.Series) -> pd.Series:
    """Indice de calor del NWS (regresion de Rothfusz con sus ajustes).

    Se alimenta con tmmx y rmin: la humedad minima diaria ocurre cerca de la
    hora de la maxima, asi que el par aproxima el pico de estres del dia.
    Por debajo de ~27 °C el NWS usa la formula simple de Steadman, que aqui
    se respeta para no inventar indices de calor en dias frescos.
    """
    t = tmax_c.to_numpy(float) * 9.0 / 5.0 + 32.0
    rh = rh_pct.to_numpy(float)

    simple = 0.5 * (t + 61.0 + (t - 68.0) * 1.2 + rh * 0.094)
    full = (
        -42.379 + 2.04901523 * t + 10.14333127 * rh - 0.22475541 * t * rh
        - 0.00683783 * t**2 - 0.05481717 * rh**2 + 0.00122874 * t**2 * rh
        + 0.00085282 * t * rh**2 - 0.00000199 * t**2 * rh**2
    )
    with np.errstate(invalid="ignore"):
        dry = (rh < 13) & (t >= 80) & (t <= 112)
        full = np.where(dry, full - ((13 - rh) / 4) * np.sqrt((17 - np.abs(t - 95)) / 17), full)
    humid = (rh > 85) & (t >= 80) & (t <= 87)
    full = np.where(humid, full + ((rh - 85) / 10) * ((87 - t) / 5), full)

    hi_f = np.where((simple + t) / 2.0 < 80.0, simple, full)
    return pd.Series((hi_f - 32.0) * 5.0 / 9.0, index=tmax_c.index, name="heat_index_c")


def local_percentile(
    frame: pd.DataFrame,
    var: str,
    q: float,
    *,
    ref_start: dt.date | str,
    ref_end: dt.date | str,
    months: tuple[int, ...] | None = (5, 6, 7, 8, 9),
) -> pd.Series:
    """Percentil `q` (0-100) de `var` por geografia, en el periodo de referencia.

    `months` restringe a la temporada calida (mayo-septiembre por defecto): el
    p95 de todo el ano en Phoenix mezcla enero con julio y queda por debajo de
    lo que la poblacion local considera un dia extremo.
    """
    dates = frame.index.get_level_values("valid_date")
    mask = (dates >= pd.Timestamp(ref_start)) & (dates <= pd.Timestamp(ref_end))
    if months:
        mask &= dates.month.isin(months)
    ref = frame.loc[mask, var]
    if ref.empty:
        raise ValueError(f"periodo de referencia {ref_start}..{ref_end} sin datos de {var}")
    return ref.groupby(level="geoid").quantile(q / 100.0).rename(f"{var}_p{q:g}")


def above_threshold(frame: pd.DataFrame, var: str, threshold: pd.Series) -> pd.Series:
    """1.0 si `var` supera el umbral de su geografia, 0.0 si no, NaN sin dato."""
    thr = threshold.reindex(frame.index.get_level_values("geoid")).to_numpy()
    vals = frame[var].to_numpy(float)
    out = np.where(np.isnan(vals) | np.isnan(thr), np.nan, (vals > thr).astype(float))
    return pd.Series(out, index=frame.index, name=f"{var}_hot")


def streak(flag: pd.Series) -> pd.Series:
    """Dias consecutivos con la bandera activa, hasta e incluyendo el dia.

    Un hueco de fechas rompe la racha: dos dias calientes separados por un
    dia sin dato no son una ola de dos dias.
    """
    out = pd.Series(0.0, index=flag.index, name=f"{flag.name}_streak")
    for _, s in flag.groupby(level="geoid"):
        d = s.index.get_level_values("valid_date").values
        contiguous = np.r_[False, np.diff(d).astype("timedelta64[D]").astype(int) == 1]
        run, vals = 0, np.empty(len(s))
        for k, (v, cont) in enumerate(zip(s.to_numpy() == 1.0, contiguous, strict=True)):
            run = (run + 1 if cont else 1) if v else 0
            vals[k] = run
        out.loc[s.index] = vals
    return out


def first_heatwave_of_season(streak_days: pd.Series, min_days: int = 2) -> pd.Series:
    """1.0 en los dias de la primera ola (>= `min_days`) de cada ano y geografia.

    Es la feature de "no aclimatacion" de §7 M2: la primera ola de la
    temporada golpea mas que una igual en agosto. Marca la ola entera, desde
    su primer dia, no solo a partir del dia en que alcanza `min_days`.
    """
    idx = streak_days.index
    out = pd.Series(0.0, index=idx, name="first_heatwave")
    keys = [idx.get_level_values("geoid"), idx.get_level_values("valid_date").year]
    for _, s in streak_days.groupby(keys):
        vals = s.to_numpy()
        hits = np.flatnonzero(vals >= min_days)
        if hits.size == 0:
            continue
        start = hits[0] - int(vals[hits[0]]) + 1
        end = hits[0]
        while end + 1 < len(vals) and vals[end + 1] == vals[end] + 1:
            end += 1
        out.loc[s.index[start : end + 1]] = 1.0
    return out
