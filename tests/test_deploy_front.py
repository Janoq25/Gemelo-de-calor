"""Escenario de calentamiento (L4 -> variables de L3) e inyeccion en la consola."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from xdt.export import inject_html
from xdt.features.panel import FEATURES, Panel, with_warming
from xdt.features.thermal import heat_index_c


def _panel() -> Panel:
    days = pd.date_range("2023-07-01", periods=10)
    idx = pd.MultiIndex.from_product([["04", "22"], days], names=["geoid", "valid_date"])
    rng = np.random.default_rng(0)
    f = pd.DataFrame(index=idx)
    f["tmax"] = 33 + rng.normal(0, 2, len(idx))
    f["tmin"] = 22 + rng.normal(0, 1, len(idx))
    f["rh_min"] = 30 + rng.normal(0, 5, len(idx))
    f["heat_index"] = heat_index_c(f["tmax"], f["rh_min"])
    p95 = 36.0
    f["hi_excess_p95"] = f["heat_index"] - p95
    g = f.groupby(level="geoid")
    f["hi_lag1"] = g["heat_index"].shift(1).fillna(f["heat_index"])
    f["hi_mean_3d"] = g["heat_index"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    f["tmin_mean_3d"] = g["tmin"].transform(lambda s: s.rolling(3, min_periods=1).mean())
    f["hot_streak"] = 0.0
    f["first_heatwave"] = 0.0
    f["hi_clim_ref"] = 30.0
    f["doy_sin"] = f["doy_cos"] = f["weekend"] = 0.0
    f["y"] = 0
    return Panel(frame=f, ref_years=(2019,), known_at=pd.Timestamp("2026-09-23"))


def test_warming_zero_is_identity():
    p = _panel()
    pd.testing.assert_frame_equal(with_warming(p, 0.0), p.X)


def test_warming_propagates_to_derived_features():
    p = _panel()
    w = with_warming(p, 3.0)
    assert list(w.columns) == list(FEATURES)
    np.testing.assert_allclose(w["tmax"] - p.X["tmax"], 3.0)
    assert (w["heat_index"] > p.X["heat_index"]).all()
    dhi = w["heat_index"] - p.X["heat_index"]
    # el exceso sobre el p95 local se mueve igual que el indice: el umbral no cambia
    np.testing.assert_allclose(w["hi_excess_p95"] - p.X["hi_excess_p95"], dhi)
    # la climatologia de referencia no se toca
    assert (w["hi_clim_ref"] == p.X["hi_clim_ref"]).all()
    assert w["hot_streak"].sum() >= p.X["hot_streak"].sum()


def test_inject_html_replaces_only_payload_line(tmp_path):
    html = tmp_path / "consola.html"
    html.write_text("<script>\nconst PAYLOAD = {\"a\":1};\nconst PATHS = {};\n</script>\n",
                    encoding="utf-8")
    inject_html({"b": "</script><x>"}, html)
    text = html.read_text(encoding="utf-8")
    line = next(ln for ln in text.splitlines() if ln.startswith("const PAYLOAD"))
    assert "</script>" not in line  # escapado: no cierra el bloque
    assert json.loads(line[len("const PAYLOAD = "):-1].replace("<\\/", "</")) == \
        {"b": "</script><x>"}
    assert "const PATHS = {};" in text


def test_inject_html_requires_payload_line(tmp_path):
    html = tmp_path / "otra.html"
    html.write_text("<p>sin datos</p>", encoding="utf-8")
    with pytest.raises(ValueError):
        inject_html({}, html)
