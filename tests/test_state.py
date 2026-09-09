"""Pruebas de L1: idempotencia, revisiones y viaje en el tiempo."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from xdt.twin.state import StateStore


def test_put_facts_inserta_y_es_idempotente(con, facts_day1, t0):
    store = StateStore(con)

    first = store.put_facts(facts_day1, source="epht", known_at=t0)
    assert first["written"] == 3
    assert first["revisions"] == 0

    # Re-ejecutar la misma ingesta no debe escribir nada: requisito de §7 M1.
    second = store.put_facts(facts_day1, source="epht", known_at=t0)
    assert second["written"] == 0
    assert second["unchanged"] == 3

    (total,) = con.execute("SELECT count(*) FROM twin_state").fetchone()
    assert total == 3


def test_valor_revisado_crea_revision_no_sobrescribe(con, facts_day1, t0, t1):
    store = StateStore(con)
    store.put_facts(facts_day1, source="epht", known_at=t0)

    revised = facts_day1.copy()
    revised.loc[revised["geoid"] == "04", "value"] = 49.4
    stats = store.put_facts(revised, source="epht", known_at=t1)

    assert stats["written"] == 1
    assert stats["revisions"] == 1

    hist = store.history(
        geo_level="state", geoid="04", valid_date="2023-01-01", variable="hri_rate"
    )
    assert len(hist) == 2
    assert hist["value"].tolist() == [48.1, 49.4]


def test_time_travel_devuelve_el_valor_vigente_en_su_momento(con, facts_day1, t0, t1):
    """El nucleo del diseno bitemporal.

    Una prediccion emitida en t0 debe poder reevaluarse con el estado que
    existia en t0, no con el revisado despues. Si esto falla, toda la
    validacion temporal del plan (§10.2) tendria fuga de informacion.
    """
    store = StateStore(con)
    store.put_facts(facts_day1, source="epht", known_at=t0)

    revised = facts_day1.copy()
    revised.loc[revised["geoid"] == "04", "value"] = 49.4
    store.put_facts(revised, source="epht", known_at=t1)

    # Hoy sabemos 49.4
    ahora = store.snapshot(geo_level="state", valid_from="2023-01-01")
    assert ahora.frame.loc[("04", pd.Timestamp("2023-01-01")), "hri_rate"] == pytest.approx(49.4)

    # Pero en t0 el gemelo creia 48.1
    entonces = store.snapshot(
        geo_level="state",
        valid_from="2023-01-01",
        known_at=t0 + dt.timedelta(seconds=1),
    )
    assert entonces.frame.loc[("04", pd.Timestamp("2023-01-01")), "hri_rate"] == pytest.approx(48.1)


def test_known_at_anterior_a_toda_ingesta_da_estado_vacio(con, facts_day1, t0):
    store = StateStore(con)
    store.put_facts(facts_day1, source="epht", known_at=t0)

    vacio = store.snapshot(
        geo_level="state", valid_from="2023-01-01", known_at=t0 - dt.timedelta(days=1)
    )
    assert vacio.frame.empty
    assert vacio.variables == []


def test_fuentes_distintas_no_colisionan(con, facts_day1, t0):
    """La fuente forma parte de la clave natural: dos fuentes pueden discrepar
    sobre el mismo hecho sin pisarse. Es necesario para contrastar gridMET
    contra GHCN (§4.2) en vez de que una sobrescriba a la otra."""
    store = StateStore(con)
    store.put_facts(facts_day1, source="epht", known_at=t0)

    otra = facts_day1.copy()
    otra["value"] = otra["value"] * 1.1
    store.put_facts(otra, source="tracking_california", known_at=t0)

    (total,) = con.execute("SELECT count(*) FROM twin_state").fetchone()
    assert total == 6

    long = store.as_of(geo_level="state", valid_from="2023-01-01")
    assert set(long["source"]) == {"epht", "tracking_california"}


def test_rechaza_geo_level_desconocido(con, facts_day1):
    store = StateStore(con)
    bad = facts_day1.copy()
    bad["geo_level"] = "manzana"
    with pytest.raises(ValueError, match="geo_level no reconocido"):
        store.put_facts(bad, source="epht")


def test_rechaza_columnas_faltantes(con):
    store = StateStore(con)
    with pytest.raises(ValueError, match="faltan columnas"):
        store.put_facts(pd.DataFrame({"geoid": ["04"]}), source="epht")


def test_duplicados_en_el_lote_se_colapsan(con, t0):
    store = StateStore(con)
    dup = pd.DataFrame(
        [
            {"geo_level": "state", "geoid": "04", "valid_date": "2023-01-01",
             "variable": "hri_rate", "value": 1.0},
            {"geo_level": "state", "geoid": "04", "valid_date": "2023-01-01",
             "variable": "hri_rate", "value": 2.0},
        ]
    )
    stats = store.put_facts(dup, source="epht", known_at=t0)
    assert stats["duplicates_collapsed"] == 1
    assert stats["written"] == 1

    long = store.as_of(geo_level="state", valid_from="2023-01-01")
    assert long["value"].tolist() == [2.0]  # gana el ultimo


def test_snapshot_es_ancho_y_ordenado(con, panel_state, t0):
    store = StateStore(con)
    store.put_facts(panel_state, source="gridmet", known_at=t0)

    snap = store.snapshot(
        geo_level="state", valid_from="2023-07-01", valid_to="2023-07-10"
    )
    assert snap.variables == ["tmmn", "tmmx"]
    assert snap.geoids == ["04", "36"]
    assert len(snap.frame) == 20
    assert snap.is_baseline
    assert snap.frame.index.names == ["geoid", "valid_date"]


def test_input_hash_es_estable_e_independiente_del_orden(con, panel_state, t0):
    store = StateStore(con)
    store.put_facts(panel_state, source="gridmet", known_at=t0)
    snap = store.snapshot(geo_level="state", valid_from="2023-07-01", valid_to="2023-07-10")

    h1 = snap.input_hash()
    barajado = snap.derive(snap.frame.sample(frac=1.0, random_state=0), "shuffle")
    assert barajado.input_hash() == h1


def test_coverage_reporta_huecos(con, facts_day1, t0):
    store = StateStore(con)
    con_nulo = pd.concat(
        [
            facts_day1,
            pd.DataFrame([{"geo_level": "state", "geoid": "99",
                           "valid_date": "2023-01-01", "variable": "hri_rate",
                           "value": None}]),
        ]
    )
    store.put_facts(con_nulo, source="epht", known_at=t0)

    cov = store.coverage()
    assert len(cov) == 1
    row = cov.iloc[0]
    assert row["n_geoids"] == 4
    assert row["n_null"] == 1


def test_rechaza_revision_antedatada(con, facts_day1, t0, t1):
    """Una revision con known_at anterior al de la revision que reemplaza
    corromperia el eje bitemporal: `as_of` con un corte intermedio devolveria
    el valor nuevo para un instante en que el gemelo aun no lo conocia."""
    store = StateStore(con)
    store.put_facts(facts_day1, source="epht", known_at=t1)

    antedatada = facts_day1.copy()
    antedatada.loc[antedatada["geoid"] == "04", "value"] = 49.4
    with pytest.raises(ValueError, match="antedatar|anterior al de la revision"):
        store.put_facts(antedatada, source="epht", known_at=t0)


def test_reingesta_identica_con_known_at_anterior_no_falla(con, facts_day1, t0, t1):
    """La guarda solo aplica a cambios de valor. Reprocesar el mismo crudo con
    un known_at antiguo es inofensivo y no debe romper el pipeline."""
    store = StateStore(con)
    store.put_facts(facts_day1, source="epht", known_at=t1)
    stats = store.put_facts(facts_day1, source="epht", known_at=t0)
    assert stats["written"] == 0


def test_rechaza_known_at_en_el_futuro(con, facts_day1):
    """Un hecho fechado en el futuro se escribiria sin error pero seria
    invisible para toda consulta que corte en "ahora". Fallo silencioso."""
    from xdt.twin.state import now_utc

    store = StateStore(con)
    futuro = now_utc() + dt.timedelta(days=1)
    with pytest.raises(ValueError, match="futuro"):
        store.put_facts(facts_day1, source="epht", known_at=futuro)
