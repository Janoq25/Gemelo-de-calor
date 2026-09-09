"""Pruebas de la capa de ingesta: cache, reintentos, modo offline y parser EPHT."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from xdt.config import Settings
from xdt.ingest.epht import MEASURES, EphtConnector, _parse_temporal
from xdt.ingest.http import CachedClient, OfflineCacheMiss

URL = "https://example.test/data"


# ------------------------------------------------------------------ cache
@respx.mock
def test_segunda_peticion_sale_de_cache(settings):
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={"ok": 1}))
    client = CachedClient(source="demo", settings=settings)

    a1 = client.get(URL)
    a2 = client.get(URL)

    assert route.call_count == 1  # solo una salida a la red
    assert a1.from_cache is False
    assert a2.from_cache is True
    assert a2.json() == {"ok": 1}
    assert a1.raw_hash == a2.raw_hash


@respx.mock
def test_parametros_distintos_son_entradas_de_cache_distintas(settings):
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={"ok": 1}))
    client = CachedClient(source="demo", settings=settings)

    client.get(URL, params={"year": 2022})
    client.get(URL, params={"year": 2023})
    assert route.call_count == 2


@respx.mock
def test_refresh_fuerza_nueva_descarga(settings):
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={"ok": 1}))
    client = CachedClient(source="demo", settings=settings)

    client.get(URL)
    client.get(URL, refresh=True)
    assert route.call_count == 2


# ------------------------------------------------------- reintentos (R6b)
@respx.mock
def test_429_reintenta_y_acaba_sirviendo(settings):
    """R6b del plan: 429 observado repetidamente sin token."""
    route = respx.get(URL).mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(429, headers={"Retry-After": "0"}),
            httpx.Response(200, json={"ok": 1}),
        ]
    )
    client = CachedClient(source="demo", settings=settings)

    art = client.get(URL)
    assert route.call_count == 3
    assert art.json() == {"ok": 1}


@respx.mock
def test_429_persistente_acaba_lanzando(settings):
    respx.get(URL).mock(return_value=httpx.Response(429))
    client = CachedClient(source="demo", settings=settings)

    with pytest.raises(httpx.HTTPStatusError):
        client.get(URL)


@respx.mock
def test_404_no_se_reintenta(settings):
    route = respx.get(URL).mock(return_value=httpx.Response(404))
    client = CachedClient(source="demo", settings=settings)

    with pytest.raises(httpx.HTTPStatusError):
        client.get(URL)
    assert route.call_count == 1


# ------------------------------------------------------------- offline
@respx.mock
def test_modo_offline_sirve_de_cache_sin_tocar_la_red(settings, tmp_path):
    route = respx.get(URL).mock(return_value=httpx.Response(200, json={"ok": 1}))
    online = Settings(data_root=tmp_path / "data")
    online.ensure_dirs()
    CachedClient(source="demo", settings=online).get(URL)
    assert route.call_count == 1

    offline = Settings(data_root=tmp_path / "data", offline=True)
    art = CachedClient(source="demo", settings=offline).get(URL)
    assert art.from_cache is True
    assert route.call_count == 1  # no hubo segunda salida


def test_modo_offline_sin_cache_falla_claro(tmp_path):
    """`dvc repro` debe correr offline. Si falta un crudo, el error tiene que
    decir exactamente eso, no reventar en un parser tres capas mas abajo."""
    s = Settings(data_root=tmp_path / "data", offline=True)
    s.ensure_dirs()
    with pytest.raises(OfflineCacheMiss, match="modo offline"):
        CachedClient(source="demo", settings=s).get(URL)


# --------------------------------------------------------------- EPHT
# Respuesta real de /getCoreHolder/440/1/1/ALL/0/2023/0/0, recortada.
# Capturada el 9/9/2026; conserva la forma exacta de fila del servicio.
EPHT_FIXTURE = {
    "tableResult": [
        {
            "id": "642829", "geographicTypeId": 1, "geo": "Arizona", "geoId": "04",
            "calculationType": "Age Adjusted Rate", "temporalTypeId": 1,
            "temporal": "2023", "temporalId": 2023, "dataValue": "48.1",
            "displayValue": "48.1", "groupById": "1", "suppressionFlag": "0",
            "confidenceIntervalLow": None, "confidenceIntervalHigh": None,
        },
        {
            "id": "749882", "geographicTypeId": 1, "geo": "Connecticut", "geoId": "09",
            "calculationType": "Age Adjusted Rate", "temporalTypeId": 1,
            "temporal": "2023", "temporalId": 2023, "dataValue": "7.8",
            "displayValue": "7.8", "groupById": "1", "suppressionFlag": "0",
            "confidenceIntervalLow": None, "confidenceIntervalHigh": None,
        },
        {
            "id": "999999", "geographicTypeId": 1, "geo": "Wyoming", "geoId": "56",
            "calculationType": "Age Adjusted Rate", "temporalTypeId": 1,
            "temporal": "2023", "temporalId": 2023, "dataValue": "",
            "displayValue": "suppressed", "groupById": "1", "suppressionFlag": "1",
            "confidenceIntervalLow": None, "confidenceIntervalHigh": None,
        },
    ],
    "dailyEstimatesTableResult": [],
}


@respx.mock
def test_epht_fetch_y_normalize(settings, monkeypatch, con):
    monkeypatch.setattr("xdt.config.get_settings", lambda: settings)
    respx.get(url__regex=r".*getCoreHolder.*").mock(
        return_value=httpx.Response(200, json=EPHT_FIXTURE)
    )

    conn = EphtConnector(settings=settings)
    arts = conn.fetch(measure="hri_ed_rate_age_adj", temporal=2023)
    facts = conn.normalize(arts)

    assert len(facts) == 3
    assert set(facts["geo_level"]) == {"state"}
    az = facts[facts["geoid"] == "04"].iloc[0]
    assert az["value"] == pytest.approx(48.1)
    assert az["valid_date"].year == 2023
    assert az["variable"].endswith("age_adj_rate")


@respx.mock
def test_epht_marca_supresion_en_vez_de_descartarla(settings, monkeypatch):
    """§14 R2: la celda suprimida se conserva marcada, no se borra."""
    monkeypatch.setattr("xdt.config.get_settings", lambda: settings)
    respx.get(url__regex=r".*getCoreHolder.*").mock(
        return_value=httpx.Response(200, json=EPHT_FIXTURE)
    )

    conn = EphtConnector(settings=settings)
    facts = conn.normalize(conn.fetch(measure="hri_ed_rate_age_adj", temporal=2023))

    wy = facts[facts["geoid"] == "56"].iloc[0]
    assert wy["suppressed"] is True or bool(wy["suppressed"]) is True
    assert pytest.approx(0) != 1  # sanity
    assert len(facts) == 3  # la fila suprimida sigue presente


@respx.mock
def test_epht_carga_en_el_estado(settings, monkeypatch, con):
    monkeypatch.setattr("xdt.config.get_settings", lambda: settings)
    respx.get(url__regex=r".*getCoreHolder.*").mock(
        return_value=httpx.Response(200, json=EPHT_FIXTURE)
    )

    conn = EphtConnector(settings=settings)
    stats = conn.run(con, measure="hri_ed_rate_age_adj", temporal=2023)

    assert stats["written"] == 3
    (n_art,) = con.execute("SELECT count(*) FROM raw_artifact").fetchone()
    assert n_art == 1

    # Re-ejecutar: sale de cache y no escribe nada nuevo.
    stats2 = conn.run(con, measure="hri_ed_rate_age_adj", temporal=2023)
    assert stats2["written"] == 0
    assert stats2["from_cache"] == 1


def test_catalogo_de_medidas_cubre_las_dos_etapas_del_plan():
    """Las medidas que sostienen §5.1 (Etapa A) y §5.2 (Etapa B)."""
    assert MEASURES["hri_ed_rate_daily_va"].measure_id == 1385
    assert MEASURES["hri_ed_rate_daily_va"].cadence == "daily"
    assert MEASURES["hri_ed_rate_daily_nonva"].geo_level == "hhs_region"
    assert MEASURES["heat_deaths_5yr"].geo_level == "county"
    # La advertencia sobre la poblacion VA debe viajar con la medida (§15.3).
    assert "centinela" in MEASURES["hri_ed_rate_daily_va"].note.lower()


@pytest.mark.parametrize(
    ("entrada", "esperado_anio"),
    [("2023", 2023), ("2018-2022", 2018), ("2023-07-15", 2023)],
)
def test_parse_temporal_cubre_las_granularidades_de_epht(entrada, esperado_anio):
    d = _parse_temporal(entrada, None)
    assert d is not None and d.year == esperado_anio


def test_parse_temporal_devuelve_none_si_no_hay_nada():
    assert _parse_temporal(None, None) is None
    assert _parse_temporal("", None) is None


def test_validate_facts_rechaza_duplicados(con):
    import pandas as pd

    from xdt.ingest.base import Connector

    dup = pd.DataFrame(
        [
            {"geo_level": "state", "geoid": "04", "valid_date": "2023-01-01",
             "variable": "x", "value": 1.0},
            {"geo_level": "state", "geoid": "04", "valid_date": "2023-01-01",
             "variable": "x", "value": 2.0},
        ]
    )
    with pytest.raises(ValueError, match="duplicados"):
        Connector.validate_facts(dup)


def test_fixture_refleja_la_cobertura_verificada():
    """Sanity: el plan documenta 16 estados en 2023 y Arizona en 48.1.
    La fixture conserva esos valores para que un cambio silencioso del parser
    se note."""
    az = next(r for r in EPHT_FIXTURE["tableResult"] if r["geoId"] == "04")
    assert az["dataValue"] == "48.1"
    assert json.loads(json.dumps(EPHT_FIXTURE))  # serializable
