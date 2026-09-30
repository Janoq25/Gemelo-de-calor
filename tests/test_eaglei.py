"""Pruebas del conector EAGLE-I y de la descarga en streaming."""

from __future__ import annotations

import datetime as dt
import hashlib
import json

import httpx
import pandas as pd
import pytest
import respx

from xdt.ingest.eaglei import (
    FIGSHARE_API,
    FIGSHARE_VERSION,
    EagleiConnector,
    parse_coverage,
    parse_mcc,
)
from xdt.ingest.http import CachedClient, InvalidBody, OfflineCacheMiss, RawArtifact

MCC = "﻿County_FIPS,Customers\n4013,1000\n4012,200\n6037,5000\n"
COVERAGE = (
    "year,state,total_customers,min_covered,max_covered,min_pct_covered,max_pct_covered\n"
    "1/1/18,AZ,3000000,2000000,2900000,0.67,0.97\n"
    "1/1/18,AL,2552741,556498,2029174,0.22,0.79\n"
)


def _outages(rows: list[tuple[str, int, str]]) -> str:
    # California (06) marca los extremos UTC del ano: el recorte de dias debe
    # salir del archivo completo, no de los estados filtrados.
    base = [("06037", 1, "2023-01-01 00:00:00"), ("06037", 1, "2023-12-31 23:45:00")]
    lines = ["fips_code,county,state,customers_out,run_start_time"]
    for fips, n, ts in base + rows:
        lines.append(f"{fips},X,Y,{n},{ts}")
    return "\n".join(lines) + "\n"


def _art(tmp_path, resource: str, text: str) -> RawArtifact:
    p = tmp_path / (resource.replace(":", "_") + ".bin")
    p.write_text(text, encoding="utf-8")
    return RawArtifact(
        raw_hash=hashlib.sha256(text.encode()).hexdigest(), request_key="k", source="eaglei",
        resource=resource, params={}, path=p, n_bytes=len(text), content_type="text/csv",
        fetched_at=dt.datetime(2026, 9, 23), from_cache=True,
    )


def _normalize(tmp_path, settings, rows, states=("04",)) -> tuple[pd.DataFrame, EagleiConnector]:
    conn = EagleiConnector(settings=settings)
    conn.states = list(states)
    arts = [
        _art(tmp_path, "eaglei:MCC.csv", MCC),
        _art(tmp_path, "eaglei:coverage_history.csv", COVERAGE),
        _art(tmp_path, f"eaglei:outages:2023:v{FIGSHARE_VERSION}", _outages(rows)),
    ]
    return conn.normalize(arts), conn


def _val(df, geo_level, geoid, day, var):
    sel = df[(df["geo_level"] == geo_level) & (df["geoid"] == geoid)
             & (df["valid_date"] == dt.date.fromisoformat(day)) & (df["variable"] == var)]
    assert len(sel) == 1, sel
    return sel["value"].iloc[0]


# Un apagon en Maricopa (MCC 1000) de 12:00 a 14:45 hora de Phoenix (UTC-7):
# 12 instantes al 5 %, mas un instante aislado a las 20:00.
EVENTO = [("04013", 50, f"2023-07-15 {h:02d}:{m:02d}:00")
          for h in range(19, 22) for m in (0, 15, 30, 45)] + [("04013", 50, "2023-07-16 03:00:00")]


def test_metricas_diarias_en_hora_local(tmp_path, settings):
    df, _ = _normalize(tmp_path, settings, EVENTO)
    assert _val(df, "county", "04013", "2023-07-15", "outage_customers_frac") == pytest.approx(0.05)
    assert _val(df, "county", "04013", "2023-07-15", "outage_max_hours") == pytest.approx(3.0)
    assert _val(df, "county", "04013", "2023-07-15", "outage_customer_hours_frac") == \
        pytest.approx(13 * 0.05 * 0.25)


def test_utc_se_lleva_al_dia_local_no_al_dia_utc(tmp_path, settings):
    """03:00 UTC del 16/7 son las 20:00 del 15/7 en Phoenix (trampa 2)."""
    df, _ = _normalize(tmp_path, settings, [("04013", 50, "2023-07-16 03:00:00")])
    assert _val(df, "county", "04013", "2023-07-15", "outage_customers_frac") > 0
    assert _val(df, "county", "04013", "2023-07-16", "outage_customers_frac") == 0


def test_serie_densa_con_ceros_y_ultimo_dia_local_recortado(tmp_path, settings):
    """El 31/12 local de Phoenix termina en el archivo de 2024: no se inventa."""
    df, conn = _normalize(tmp_path, settings, EVENTO)
    county = df[(df["geo_level"] == "county") & (df["variable"] == "outage_max_hours")]
    assert set(county["geoid"]) == {"04013", "04012"}  # La Paz sin eventos: ceros
    assert county["valid_date"].min() == dt.date(2023, 1, 1)
    assert county["valid_date"].max() == dt.date(2023, 12, 30)
    assert len(county) == 2 * 364
    assert conn.report["local_days"]["04"] == ("2023-01-01", "2023-12-30")


def test_agregado_estatal_usa_el_mcc_del_estado_entero(tmp_path, settings):
    df, _ = _normalize(tmp_path, settings, EVENTO)
    # 50 de 1200 clientes modelados en AZ
    assert _val(df, "state", "04", "2023-07-15", "outage_customers_frac") == \
        pytest.approx(50 / 1200)


def test_mcc_recupera_ceros_a_la_izquierda():
    """Trampa 4: '4013' debe casar con '04013'."""
    mcc = parse_mcc(_art_mem(MCC))
    assert "04013" in set(mcc["geoid"]) and "4013" not in set(mcc["geoid"])


def test_fraccion_se_recorta_a_uno(tmp_path, settings):
    df, conn = _normalize(tmp_path, settings, [("04012", 900, "2023-07-15 19:00:00")])
    assert _val(df, "county", "04012", "2023-07-15", "outage_customers_frac") == 1.0
    assert conn.report["snapshots_clipped_to_mcc"] == 1


def test_formato_de_fecha_del_articulo_tambien_se_acepta(tmp_path, settings):
    """Trampa 3: el articulo documenta MM/DD/YY; los archivos usan ISO."""
    df, _ = _normalize(tmp_path, settings, [("04013", 50, "7/15/23 19:00")])
    assert _val(df, "county", "04013", "2023-07-15", "outage_customers_frac") > 0


def test_fecha_irreconocible_detiene_la_ingesta(tmp_path, settings):
    with pytest.raises(InvalidBody, match="formato desconocido"):
        _normalize(tmp_path, settings, [("04013", 50, "15.07.2023 19h")])


def test_cobertura_por_estado_y_anio():
    cov = parse_coverage(_art_mem(COVERAGE))
    al = cov[(cov["geoid"] == "01") & (cov["variable"] == "eaglei_cov_min_pct")]
    assert al["value"].iloc[0] == pytest.approx(22.0)
    assert al["valid_date"].iloc[0] == dt.date(2018, 1, 1)


def test_normalize_carga_en_el_estado(tmp_path, settings, con):
    from xdt.twin.state import StateStore

    df, conn = _normalize(tmp_path, settings, EVENTO)
    conn.validate_facts(df)
    StateStore(con).put_facts(df, source="eaglei")
    snap = StateStore(con).snapshot(
        geo_level="county", valid_from="2023-07-15", geoids=["04013"]
    )
    assert snap.frame["outage_max_hours"].iloc[0] == pytest.approx(3.0)


def _art_mem(text: str) -> RawArtifact:
    class _A:
        def bytes(self):  # noqa: D401 - imita RawArtifact.bytes
            return text.encode("utf-8")

    return _A()  # type: ignore[return-value]


# -------------------------------------------------- descarga en streaming
BIG = b"fips_code,county\n" + b"01001,Autauga\n" * 5000


@respx.mock
def test_download_streaming_cachea_y_hashea(settings):
    route = respx.get("https://files.test/big.csv").mock(
        return_value=httpx.Response(200, content=BIG)
    )
    c = CachedClient(source="demo", settings=settings)
    a1 = c.download("https://files.test/big.csv", resource="demo:big")
    a2 = c.download("https://files.test/big.csv", resource="demo:big")
    assert route.call_count == 1
    assert a1.raw_hash == hashlib.sha256(BIG).hexdigest() == a2.raw_hash
    assert a2.from_cache and a1.path.read_bytes() == BIG
    assert not list(settings.raw_dir.rglob("*.tmp"))


@respx.mock
def test_download_reintenta_503(settings):
    route = respx.get("https://files.test/big.csv").mock(
        side_effect=[httpx.Response(503), httpx.Response(200, content=BIG)]
    )
    art = CachedClient(source="demo", settings=settings).download(
        "https://files.test/big.csv", resource="demo:big"
    )
    assert route.call_count == 2 and art.n_bytes == len(BIG)


def test_download_offline_sin_cache_falla_claro(tmp_path):
    from xdt.config import Settings

    s = Settings(data_root=tmp_path / "data", offline=True)
    s.ensure_dirs()
    with pytest.raises(OfflineCacheMiss):
        CachedClient(source="demo", settings=s).download("https://x.test/a", resource="r")


@respx.mock
def test_md5_distinto_borra_la_descarga_corrupta(settings):
    meta = {"files": [
        {"name": "MCC.csv", "download_url": "https://files.test/mcc"},
        {"name": "coverage_history.csv", "download_url": "https://files.test/cov"},
        {"name": "eaglei_outages_2023.csv", "download_url": "https://files.test/2023",
         "computed_md5": "0" * 32},
    ]}
    respx.get(f"{FIGSHARE_API}/versions/{FIGSHARE_VERSION}").mock(
        return_value=httpx.Response(200, content=json.dumps(meta).encode())
    )
    respx.get("https://files.test/mcc").mock(return_value=httpx.Response(200, text=MCC))
    respx.get("https://files.test/cov").mock(return_value=httpx.Response(200, text=COVERAGE))
    respx.get("https://files.test/2023").mock(
        return_value=httpx.Response(200, text=_outages(EVENTO))
    )
    conn = EagleiConnector(settings=settings)
    with pytest.raises(InvalidBody, match="MD5"):
        conn.fetch(years=[2023], states=["04"])
    bins = list((settings.raw_dir / "eaglei").rglob("*.bin"))
    assert not [p for p in bins if p.read_bytes().startswith(b"fips_code")]
