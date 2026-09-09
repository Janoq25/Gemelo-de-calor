from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from xdt.config import Settings, get_settings
from xdt.storage import connect


@pytest.fixture(autouse=True)
def _isolated_settings(tmp_path, monkeypatch):
    """Cada test corre con su propia raiz de datos, nunca sobre data/ real."""
    get_settings.cache_clear()
    monkeypatch.setenv("XDT_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setenv("XDT_EPHT_TOKEN", "")
    monkeypatch.setenv("XDT_HTTP_BACKOFF_BASE_S", "0.01")
    monkeypatch.setenv("XDT_HTTP_BACKOFF_MAX_S", "0.05")
    monkeypatch.setenv("XDT_OFFLINE", "0")
    yield
    get_settings.cache_clear()


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings(data_root=tmp_path / "data", http_backoff_base_s=0.01, http_backoff_max_s=0.05)
    s.ensure_dirs()
    return s


@pytest.fixture
def con():
    c = connect(":memory:")
    yield c
    c.close()


@pytest.fixture
def facts_day1() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"geo_level": "state", "geoid": "04", "valid_date": "2023-01-01",
             "variable": "hri_rate", "value": 48.1},
            {"geo_level": "state", "geoid": "22", "valid_date": "2023-01-01",
             "variable": "hri_rate", "value": 57.7},
            {"geo_level": "state", "geoid": "36", "valid_date": "2023-01-01",
             "variable": "hri_rate", "value": 6.3},
        ]
    )


@pytest.fixture
def t0() -> dt.datetime:
    return dt.datetime(2024, 1, 1, 12, 0, 0)


@pytest.fixture
def t1() -> dt.datetime:
    return dt.datetime(2025, 6, 1, 12, 0, 0)


@pytest.fixture
def panel_state() -> pd.DataFrame:
    """Panel corto (2 geografias x 10 dias) para probar escenarios."""
    dates = pd.date_range("2023-07-01", periods=10, freq="D").date
    rows = []
    for geoid, base in (("04", 38.0), ("36", 28.0)):
        for i, d in enumerate(dates):
            rows.append(
                {"geo_level": "state", "geoid": geoid, "valid_date": d,
                 "variable": "tmmx", "value": base + (i % 4)}
            )
            rows.append(
                {"geo_level": "state", "geoid": geoid, "valid_date": d,
                 "variable": "tmmn", "value": base - 10 + (i % 3)}
            )
    return pd.DataFrame(rows)
