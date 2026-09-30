"""Tests de contrato contra el THREDDS vivo de gridMET.

Fijan los hechos verificados el 23/9/2026 sobre los que descansa el conector.
Si alguno falla, gridMET cambio algo y la ingesta debe revisarse antes de
volver a correr: no son fallos de red, son fallos de supuesto.

    uv run pytest -m contract tests/contract/test_gridmet_contract.py
"""

from __future__ import annotations

import re

import pytest

from xdt.ingest.gridmet import (
    DLAT,
    DLON,
    LAT0,
    LON0,
    NLAT,
    NLON,
    VARIABLES,
    GridmetConnector,
    grid_cell,
    parse_ascii,
    parse_das,
    unpack,
)

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def conn():
    c = GridmetConnector()
    yield c
    c.close()


def _ascii(conn, short: str, t0: int, t1: int, i: int, j: int) -> str:
    v = VARIABLES[short]
    return conn.client.get(
        f"/{v.file}.ascii?{v.ncvar}[{t0}:1:{t1}][{i}:1:{i}][{j}:1:{j}]",
        resource=f"contract:{short}:{i}:{j}:{t0}-{t1}",
        refresh=True,
    ).text()


def test_malla_no_cambio(conn):
    dds = conn.client.get(
        f"/{VARIABLES['tmmx'].file}.dds", resource="contract:dds", refresh=True
    ).text()
    assert f"[lat = {NLAT}]" in dds and f"[lon = {NLON}]" in dds
    assert int(re.search(r"\[day = (\d+)\]", dds).group(1)) > 17000


@pytest.mark.parametrize("short", list(VARIABLES))
def test_empaquetado_y_unidades_de_cada_variable(conn, short):
    v = VARIABLES[short]
    das = conn.client.get(f"/{v.file}.das", resource=f"contract:das:{short}", refresh=True)
    p = parse_das(das.text(), v.ncvar)
    assert p.units == v.units
    assert p.fill == 32767
    assert p.scale == pytest.approx(0.1)


def test_offsets_de_temperatura_difieren(conn):
    """Trampa 2 del conector: si algun dia se igualan, este test lo dira."""
    offs = {}
    for short in ("tmmx", "tmmn"):
        v = VARIABLES[short]
        das = conn.client.get(f"/{v.file}.das", resource=f"contract:das:{short}", refresh=True)
        offs[short] = parse_das(das.text(), v.ncvar).offset
    assert offs == {"tmmx": 220.0, "tmmn": 210.0}


def test_phoenix_1_julio_2023(conn):
    """Valor de referencia verificado: 971 empaquetado = 43.95 °C."""
    i, j = grid_cell(33.45, -112.07)
    t = 16252  # 2023-07-01
    s = parse_ascii(_ascii(conn, "tmmx", t, t, i, j), VARIABLES["tmmx"].ncvar,
                    expect_cell=(i, j))
    assert str(s["valid_date"].iloc[0]) == "2023-07-01"
    assert s["raw"].iloc[0] == 971
    das = conn.client.get(f"/{VARIABLES['tmmx'].file}.das", resource="contract:das:tmmx")
    p = parse_das(das.text(), VARIABLES["tmmx"].ncvar)
    assert unpack(s["raw"].to_numpy(), p, VARIABLES["tmmx"])[0] == pytest.approx(43.95)


def test_oceano_devuelve_relleno_con_http_200(conn):
    """Trampa 3: celda enmascarada -> 32767, no un error."""
    i = int(round((33.0 - LAT0) / DLAT))
    j = int(round((-119.5 - LON0) / DLON))
    s = parse_ascii(_ascii(conn, "tmmx", 16252, 16252, i, j), VARIABLES["tmmx"].ncvar)
    assert s["raw"].iloc[0] == 32767
