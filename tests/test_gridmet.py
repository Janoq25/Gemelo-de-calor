"""Pruebas del conector gridMET y de la agregacion ponderada (M0)."""

from __future__ import annotations

import re

import httpx
import numpy as np
import pandas as pd
import pytest
import respx

from xdt.harmonize.aggregate import population_weighted
from xdt.harmonize.centroids import CENPOP_COUNTY_URL
from xdt.ingest.gridmet import (
    VARIABLES,
    GridmetConnector,
    Packing,
    _check_opendap_body,
    cell_center,
    grid_cell,
    on_grid,
    parse_ascii,
    parse_das,
    unpack,
)
from xdt.ingest.http import InvalidBody

# Extracto real del DAS de agg_met_tmmx / agg_met_tmmn (23/9/2026).
DAS_TMMX = """Attributes {
    lat {
        String units "degrees_north";
    }
    daily_maximum_temperature {
        Int16 _FillValue 32767;
        String units "K";
        String description "Daily Maximum Temperature (2m)";
        Int16 missing_value 32767;
        Float64 scale_factor 0.1;
        Float64 add_offset 220.0;
        String _Unsigned "true";
    }
}
"""
DAS_TMMN = DAS_TMMX.replace("daily_maximum_temperature", "daily_minimum_temperature").replace(
    "add_offset 220.0", "add_offset 210.0"
)


def ascii_body(ncvar: str, i: int, j: int, raw: list[int], day0: float = 44925.0) -> str:
    """Respuesta `.ascii` de OPeNDAP con la forma exacta del servicio."""
    n = len(raw)
    lat, lon = cell_center(i, j)
    rows = "\n".join(f"[{k}][0], {v}" for k, v in enumerate(raw))
    days = ", ".join(f"{day0 + k:.1f}" for k in range(n))
    return (
        "Dataset {\n    Grid {\n     ARRAY:\n"
        f"        UInt16 {ncvar}[day = {n}][lat = 1][lon = 1];\n"
        "    } " + ncvar + ";\n} agg.nc;\n"
        "---------------------------------------------\n"
        f"{ncvar}.{ncvar}[{n}][1][1]\n{rows}\n\n"
        f"{ncvar}.day[{n}]\n{days}\n\n"
        f"{ncvar}.lat[1]\n{lat}\n\n{ncvar}.lon[1]\n{lon}\n\n"
    )


# ------------------------------------------------------------ empaquetado
def test_das_se_lee_por_variable_porque_los_offsets_difieren():
    """Trampa 2: tmmx y tmmn NO comparten add_offset."""
    assert parse_das(DAS_TMMX, "daily_maximum_temperature").offset == 220.0
    assert parse_das(DAS_TMMN, "daily_minimum_temperature").offset == 210.0


def test_unpack_phoenix_verificado():
    """Trampa 1: 971 empaquetado = 317.1 K = 43.95 °C (Phoenix, 1/7/2023)."""
    p = parse_das(DAS_TMMX, "daily_maximum_temperature")
    out = unpack(np.array([971.0]), p, VARIABLES["tmmx"])
    assert out[0] == pytest.approx(43.95, abs=1e-6)


def test_valor_de_relleno_del_oceano_es_nan_no_3496_kelvin():
    """Trampa 3: el oceano responde 200 con 32767."""
    p = parse_das(DAS_TMMX, "daily_maximum_temperature")
    out = unpack(np.array([32767.0, 971.0]), p, VARIABLES["tmmx"])
    assert np.isnan(out[0]) and not np.isnan(out[1])


def test_unidad_inesperada_detiene_la_ingesta():
    p = Packing(scale=0.1, offset=0.0, fill=32767, units="degC")
    with pytest.raises(InvalidBody, match="unidad"):
        unpack(np.array([1.0]), p, VARIABLES["tmmx"])


def test_das_sin_la_variable_falla_claro():
    with pytest.raises(InvalidBody, match="no describe"):
        parse_das(DAS_TMMX, "daily_mean_wind_speed")


# ------------------------------------------------------------ parser ASCII
def test_parse_ascii_fechas_y_valores():
    i, j = grid_cell(33.45, -112.07)
    s = parse_ascii(ascii_body("daily_maximum_temperature", i, j, [672, 666, 688]),
                    "daily_maximum_temperature", expect_cell=(i, j))
    assert list(s["raw"]) == [672, 666, 688]
    assert str(s["valid_date"].iloc[0]) == "2023-01-01"  # 44925 dias desde 1900-01-01
    assert str(s["valid_date"].iloc[-1]) == "2023-01-03"


def test_respuesta_truncada_no_pasa_por_serie_corta():
    i, j = grid_cell(33.45, -112.07)
    body = ascii_body("daily_maximum_temperature", i, j, [672, 666, 688])
    body = body.replace("[2][0], 688\n", "")
    with pytest.raises(InvalidBody, match="truncada"):
        parse_ascii(body, "daily_maximum_temperature", expect_cell=(i, j))


def test_celda_devuelta_distinta_de_la_pedida_falla():
    """Si la malla cambia, no se asigna clima al lugar equivocado."""
    i, j = grid_cell(33.45, -112.07)
    body = ascii_body("daily_maximum_temperature", i + 3, j, [672])
    with pytest.raises(InvalidBody, match="malla"):
        parse_ascii(body, "daily_maximum_temperature", expect_cell=(i, j))


def test_documento_error_opendap_se_detecta():
    body = b'Error {\n    code = 3;\n    message = "stop >= size: 99999:17431";\n};'
    with pytest.raises(InvalidBody, match="stop >= size"):
        _check_opendap_body(body)
    _check_opendap_body(b"Dataset {\n}")  # respuesta valida: no lanza


# ------------------------------------------------------------------ malla
def test_grid_cell_ida_y_vuelta():
    i, j = grid_cell(33.45, -112.07)
    lat, lon = cell_center(i, j)
    assert abs(lat - 33.45) <= 1 / 48 and abs(lon + 112.07) <= 1 / 48


def test_condados_fuera_de_la_malla_se_excluyen_sin_romper():
    """Monroe (FL) tiene el centroide en 24.75 N; la malla empieza en 25.06 N."""
    c = pd.DataFrame({
        "geoid": ["04013", "12087", "02020"],
        "state_fips": ["04", "12", "02"],
        "lat": [33.5, 24.747427, 61.2],
        "lon": [-112.1, -81.244796, -149.9],
    })
    assert on_grid(c)["geoid"].tolist() == ["04013"]


# ------------------------------------------------------ agregacion (M0)
WEIGHTS = pd.DataFrame({
    "geoid": ["04013", "04012", "04001"],
    "population": [800, 150, 50],
    "state_fips": ["04", "04", "04"],
})


def _facts(values: dict[str, float]) -> pd.DataFrame:
    return pd.DataFrame([
        {"geoid": g, "valid_date": "2023-07-15", "variable": "tmmx", "value": v}
        for g, v in values.items()
    ])


def test_ponderacion_por_poblacion():
    out = population_weighted(
        _facts({"04013": 46.0, "04012": 48.0, "04001": 34.0}), WEIGHTS,
        parent_col="state_fips", parent_level="state",
    )
    assert len(out) == 1
    # 0.80*46 + 0.15*48 + 0.05*34 = 45.7 ; la media simple seria 42.67
    assert out["value"].iloc[0] == pytest.approx(45.7)
    assert out["geo_level"].iloc[0] == "state"


def test_estado_con_cobertura_insuficiente_no_se_emite():
    """Un estado con condados sin dato NO es "el estado"."""
    out = population_weighted(
        _facts({"04013": 46.0, "04012": 48.0}), WEIGHTS,  # falta 5 % del peso
        parent_col="state_fips", parent_level="state",
    )
    assert out.empty

    laxo = population_weighted(
        _facts({"04013": 46.0, "04012": 48.0}), WEIGHTS,
        parent_col="state_fips", parent_level="state", min_weight_coverage=0.9,
    )
    assert laxo["weight_coverage"].iloc[0] == pytest.approx(0.95)
    assert laxo["value"].iloc[0] == pytest.approx((800 * 46 + 150 * 48) / 950)


# ------------------------------------------------------ extremo a extremo
CENPOP = (
    "﻿STATEFP,COUNTYFP,COUNAME,STNAME,POPULATION,LATITUDE,LONGITUDE\n"
    "04,013,Maricopa,Arizona,800,+33.450000,-112.070000\n"
    "04,012,La Paz,Arizona,150,+33.700000,-114.000000\n"
    "04,001,Apache,Arizona,50,+35.300000,-109.500000\n"
    "02,020,Anchorage,Alaska,290000,+61.200000,-149.900000\n"
)
RAW_BY_COUNTY = {"04013": [971, 977], "04012": [990, 991], "04001": [850, 860]}


def _mock_gridmet() -> respx.Route:
    respx.get(CENPOP_COUNTY_URL).mock(
        return_value=httpx.Response(200, content=CENPOP.encode("utf-8"))
    )
    respx.get(url__regex=r".*agg_met_tmmx.*\.das$").mock(
        return_value=httpx.Response(200, text=DAS_TMMX)
    )
    by_cell = {
        grid_cell(33.45, -112.07): RAW_BY_COUNTY["04013"],
        grid_cell(33.70, -114.00): RAW_BY_COUNTY["04012"],
        grid_cell(35.30, -109.50): RAW_BY_COUNTY["04001"],
    }

    def serve(request: httpx.Request) -> httpx.Response:
        q = httpx.URL(str(request.url)).query.decode()
        q = q.replace("%5B", "[").replace("%5D", "]").replace("%3A", ":")
        t0, t1, i, j = map(int, re.search(
            r"\[(\d+):1:(\d+)\]\[(\d+):1:\d+\]\[(\d+):1:\d+\]", q).groups())
        return httpx.Response(200, text=ascii_body(
            "daily_maximum_temperature", i, j, by_cell[(i, j)], day0=44925.0 + 195))

    return respx.get(url__regex=r".*agg_met_tmmx.*\.ascii.*").mock(side_effect=serve)


@respx.mock
def test_gridmet_extremo_a_extremo(settings, con):
    route = _mock_gridmet()
    conn = GridmetConnector(settings=settings)
    stats = conn.run(con, years=[2023], variables=["tmmx"], states=["04", "02"])

    assert route.call_count == 3  # una por celda; Alaska ni se pide
    rows = con.execute(
        "SELECT geo_level, geoid, valid_date, value FROM twin_state ORDER BY 1, 2, 3"
    ).df()
    county = rows[rows["geo_level"] == "county"]
    assert set(county["geoid"]) == {"04013", "04012", "04001"}
    maricopa = county[county["geoid"] == "04013"]["value"].iloc[0]
    assert maricopa == pytest.approx(43.95)

    state = rows[rows["geo_level"] == "state"]
    assert list(state["geoid"].unique()) == ["04"]
    esperado = (800 * 43.95 + 150 * (99.0 + 220 - 273.15) + 50 * (85.0 + 220 - 273.15)) / 1000
    assert state["value"].iloc[0] == pytest.approx(esperado)
    assert stats["written"] == 3 * 2 + 2

    # Idempotencia: segunda corrida sale de cache y no escribe nada.
    stats2 = conn.run(con, years=[2023], variables=["tmmx"], states=["04"])
    assert route.call_count == 3
    assert stats2["written"] == 0
    conn.close()


@respx.mock
def test_gridmet_normalize_es_reproducible_desde_la_cache(settings, con):
    """normalize() solo necesita los crudos: nada fuera de data/raw."""
    _mock_gridmet()
    conn = GridmetConnector(settings=settings)
    arts = conn.fetch(years=[2023], variables=["tmmx"], states=["04"])
    a = conn.normalize(arts)
    b = conn.normalize(list(reversed(arts)))
    key = ["geo_level", "geoid", "valid_date", "variable"]
    pd.testing.assert_frame_equal(
        a.sort_values(key).reset_index(drop=True), b.sort_values(key).reset_index(drop=True)
    )
    conn.close()


def test_gridmet_exige_acotar_la_seleccion(settings):
    conn = GridmetConnector(settings=settings)
    with pytest.raises(ValueError, match="states"):
        conn.fetch(years=[2023])
    with pytest.raises(ValueError, match="desconocidas"):
        conn.fetch(years=[2023], variables=["pr"], states=["04"])
    conn.close()
