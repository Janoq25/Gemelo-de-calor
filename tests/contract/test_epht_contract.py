"""Tests de contrato contra la API viva de EPHT.

§12.2 del plan: "los conectores fallan silenciosamente cuando cambia un
endpoint; los tests son la unica defensa".

No corren por defecto. Para ejecutarlos:

    uv run pytest -m contract

Conviene correrlos en un cron semanal, no en cada commit: sin token la API
devuelve 429 tras ~8 peticiones, y estos tests gastan varias.
"""

from __future__ import annotations

import pytest

from xdt.ingest.epht import MEASURES, EphtConnector

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def conn():
    c = EphtConnector()
    yield c
    c.close()


def test_endpoint_geographic_types_sigue_vivo(conn):
    """El reemplazo de /geographicLevels/, que devuelve 410 Gone (R6)."""
    got = conn.client.get("/geographicTypes/440").json()
    assert isinstance(got, list) and got
    assert {"id", "geographicTypeId", "geographicType"} <= set(got[0])


def test_geographic_levels_sigue_retirado(conn):
    """Si algun dia revive, conviene enterarse: el plan documenta su retirada."""
    import httpx

    try:
        body = conn.client.get("/geographicLevels/440").json()
    except httpx.HTTPStatusError as exc:
        assert exc.response.status_code == 410
        return
    assert body.get("code") == 410, "el endpoint retirado volvio a estar activo"


def test_stratification_level_de_la_medida_440_sigue_siendo_1(conn):
    """El id que espera getCoreHolder. Si cambia, la ingesta devolveria
    resultados vacios sin lanzar ningun error."""
    levels = conn.client.get("/stratificationlevel/440/1/0").json()
    ids = {lv["id"] for lv in levels}
    assert MEASURES["hri_ed_rate_age_adj"].strat_level_id in ids


def test_medida_440_devuelve_datos_estatales_de_2023(conn):
    """Contrato de forma y de contenido a la vez.

    El plan documenta que en 2023 solo 16 estados reportaron y que Arizona
    estuvo en 48.1. Si el numero de estados cambia, no es un fallo del test:
    es una revision de la fuente que hay que incorporar al manuscrito (§15.2).
    """
    arts = conn.fetch(measure="hri_ed_rate_age_adj", temporal=2023)
    facts = conn.normalize(arts)

    assert not facts.empty, "getCoreHolder devolvio vacio: revisa strat_level_id"
    assert set(facts["geo_level"]) == {"state"}

    az = facts[facts["geoid"] == "04"]
    assert len(az) == 1
    assert az.iloc[0]["value"] == pytest.approx(48.1, abs=0.05)

    n_estados = facts["geoid"].nunique()
    assert n_estados == 16, (
        f"la cobertura de 2023 cambio de 16 a {n_estados} estados; "
        "actualizar la limitacion §15.2 del plan"
    )


def test_la_disyuntiva_de_resolucion_sigue_vigente(conn):
    """La restriccion dura sobre la que se construyo todo el §5 del plan:
    ninguna medida ofrece a la vez temporalidad fina y geografia sub-estatal.

    Si esto dejara de ser cierto, el diseno de fusion de dos escalas seria
    innecesario y habria que replantear el trabajo. Merece un test.
    """
    diaria_va = conn.client.get("/geographicTypes/1385").json()
    assert [g["geographicType"] for g in diaria_va] == ["State"]

    diaria_nonva = conn.client.get("/geographicTypes/1238").json()
    assert [g["geographicType"] for g in diaria_nonva] == ["HHS Regions"]

    # La unica con nivel condado es mortalidad agregada a 5 anos.
    quinquenal = conn.client.get("/geographicTypes/1034").json()
    assert "County" in [g["geographicType"] for g in quinquenal]
