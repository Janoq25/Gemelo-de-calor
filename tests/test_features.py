"""Pruebas de M2: features termicas y evento compuesto."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from xdt.features.compound import compound_features, compound_summary
from xdt.features.thermal import (
    above_threshold,
    first_heatwave_of_season,
    heat_index_c,
    local_percentile,
    streak,
)


def _f_to_c(f: float) -> float:
    return (f - 32) * 5 / 9


@pytest.mark.parametrize(
    ("t_f", "rh", "hi_f"),
    [(90, 50, 95), (100, 40, 109), (86, 90, 105), (80, 40, 80)],  # tabla del NWS
)
def test_indice_de_calor_reproduce_la_tabla_del_nws(t_f, rh, hi_f):
    hi = heat_index_c(pd.Series([_f_to_c(t_f)]), pd.Series([float(rh)]))
    assert hi.iloc[0] == pytest.approx(_f_to_c(hi_f), abs=0.9)  # ~1.6 °F: la tabla viene redondeada


def _frame(values: dict[str, list[float]], start="2020-06-01", var="tmmx") -> pd.DataFrame:
    rows = []
    for geoid, vals in values.items():
        for d, v in zip(pd.date_range(start, periods=len(vals)), vals, strict=True):
            rows.append({"geoid": geoid, "valid_date": d, var: v})
    return pd.DataFrame(rows).set_index(["geoid", "valid_date"]).sort_index()


def test_percentil_local_no_mira_fuera_del_periodo_de_referencia():
    """Regla de fuga: el umbral de 2020 no puede contener el clima de 2021."""
    f = pd.concat([_frame({"04": [40.0] * 30}, "2020-06-01"),
                   _frame({"04": [50.0] * 30}, "2021-06-01")])
    thr = local_percentile(f, "tmmx", 95, ref_start="2020-01-01", ref_end="2020-12-31")
    assert thr["04"] == pytest.approx(40.0)


def test_percentil_local_es_por_geografia():
    f = _frame({"04": list(np.linspace(30, 45, 100)), "36": list(np.linspace(20, 32, 100))})
    thr = local_percentile(f, "tmmx", 95, ref_start="2020-01-01", ref_end="2020-12-31")
    assert thr["04"] > 44 and 30 < thr["36"] < 32


def test_racha_se_rompe_con_hueco_de_fechas():
    f = _frame({"04": [1.0, 1.0, 0.0, 1.0, 1.0, 1.0]}, var="hot")
    f = f.drop(index=("04", pd.Timestamp("2020-06-05")))  # hueco a mitad de ola
    s = streak(f["hot"])
    assert s.tolist() == [1, 2, 0, 1, 1]


def test_primera_ola_de_temporada_marca_la_ola_entera():
    f = _frame({"04": [0, 1, 0, 1, 1, 1, 0, 1, 1, 0]}, var="hot").astype(float)
    first = first_heatwave_of_season(streak(f["hot"]), min_days=2)
    assert first.tolist() == [0, 0, 0, 1, 1, 1, 0, 0, 0, 0]


def test_evento_compuesto_y_resumen():
    tmmx = [30.0] * 8 + [45.0, 46.0]
    f = _frame({"04": tmmx})
    f["outage_customers_frac"] = [0.0] * 8 + [0.05, 0.0]
    f["outage_max_hours"] = [0.0] * 8 + [6.0, 0.0]
    flags = compound_features(f, ref_start="2020-01-01", ref_end="2020-12-31", q=75)
    assert flags["hot"].tolist()[-2:] == [1.0, 1.0]
    assert flags["compound"].tolist() == [0.0] * 8 + [1.0, 0.0]

    s = compound_summary(flags).loc["04"]
    assert s["compound_days"] == 1 and s["hot_days"] == 2 and s["outage_days"] == 1
    assert s["obs_over_exp"] == pytest.approx(1 / (2 * 1 / 10))


def test_sin_dato_de_apagon_no_es_ausencia_de_apagon():
    f = _frame({"04": [30.0, 45.0]})
    f["outage_customers_frac"] = [np.nan, np.nan]
    f["outage_max_hours"] = [np.nan, np.nan]
    flags = compound_features(f, ref_start="2020-01-01", ref_end="2020-12-31", q=50)
    assert flags["compound"].isna().all()


def test_umbral_nan_da_nan_no_falso():
    f = _frame({"04": [30.0, 45.0]})
    out = above_threshold(f, "tmmx", pd.Series({"36": 40.0}))
    assert out.isna().all()
