"""Pruebas de L0: registro de predicciones y observaciones censuradas."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from xdt.twin.registry import ObservationStore, Prediction, PredictionRegistry


def _pred(**kw) -> Prediction:
    base = dict(
        geo_level="state",
        geoid="04",
        target_date=dt.date(2023, 7, 15),
        quantity="risk_score",
        value=72.5,
        model_name="lgbm",
        model_version="0.1.0",
        input_hash="abc123",
        state_known_at=dt.datetime(2023, 7, 14, 6, 0),
        seed=42,
    )
    base.update(kw)
    return Prediction(**base)


def test_log_es_idempotente(con):
    reg = PredictionRegistry(con)
    assert reg.log([_pred()]) == 1
    assert reg.log([_pred()]) == 0  # misma prediccion, no se duplica

    (n,) = con.execute("SELECT count(*) FROM prediction_registry").fetchone()
    assert n == 1


def test_cambio_de_version_de_modelo_es_otra_prediccion(con):
    reg = PredictionRegistry(con)
    reg.log([_pred()])
    reg.log([_pred(model_version="0.2.0")])

    (n,) = con.execute("SELECT count(*) FROM prediction_registry").fetchone()
    assert n == 2


def test_cambio_de_input_hash_es_otra_prediccion(con):
    """Mismo modelo y misma fecha, pero otros datos de entrada: son dos
    predicciones distintas y ambas deben quedar registradas."""
    reg = PredictionRegistry(con)
    reg.log([_pred()])
    reg.log([_pred(input_hash="def456", value=80.0)])

    (n,) = con.execute("SELECT count(*) FROM prediction_registry").fetchone()
    assert n == 2


def test_audit_recupera_todo_lo_necesario_para_reproducir(con):
    reg = PredictionRegistry(con)
    p = _pred(extras={"features": 12, "escenario": None})
    reg.log([p])

    rec = reg.audit(p.prediction_id())
    assert rec is not None
    assert rec["model_version"] == "0.1.0"
    assert rec["input_hash"] == "abc123"
    assert rec["seed"] == 42
    # El corte bitemporal usado: sin el, no se puede demostrar ausencia de fuga.
    assert rec["state_known_at"] == dt.datetime(2023, 7, 14, 6, 0)
    assert rec["extras"]["features"] == 12
    assert rec["code_version"]


def test_audit_de_id_inexistente_devuelve_none(con):
    assert PredictionRegistry(con).audit("no-existe") is None


def test_filtro_por_escenario_separa_contrafactuales_del_baseline(con):
    reg = PredictionRegistry(con)
    reg.log([_pred()])
    reg.log([_pred(scenario_id="scn-apagon", value=95.0)])

    base = reg.get(baseline_only=True)
    assert len(base) == 1
    assert base.iloc[0]["value"] == pytest.approx(72.5)

    scn = reg.get(scenario_id="scn-apagon")
    assert len(scn) == 1
    assert scn.iloc[0]["value"] == pytest.approx(95.0)


def test_log_frame(con):
    reg = PredictionRegistry(con)
    df = pd.DataFrame(
        {
            "geoid": ["04", "36"],
            "target_date": ["2023-07-15", "2023-07-15"],
            "value": [72.5, 31.0],
        }
    )
    n = reg.log_frame(
        df, model_name="lgbm", model_version="0.1.0",
        quantity="risk_score", input_hash="h", geo_level="state",
    )
    assert n == 2


# --------------------------------------------------------------------- L5
def test_observacion_suprimida_se_guarda_como_censura_no_como_nulo(con):
    """§14 R2: la supresion de celdas es censura por intervalo.

    Una celda con <10 casos no es un hueco; es "el valor esta entre 0 y 10".
    Descartarla sesgaria la calibracion hacia las zonas de mayor incidencia.
    """
    obs = ObservationStore(con)
    df = pd.DataFrame(
        [
            {"geo_level": "county", "geoid": "04013", "target_date": "2023-01-01",
             "quantity": "heat_deaths", "value": 24.0, "censored": False},
            {"geo_level": "county", "geoid": "04003", "target_date": "2023-01-01",
             "quantity": "heat_deaths", "value": None, "censored": True,
             "censor_lo": 0.0, "censor_hi": 10.0},
        ]
    )
    obs.put(df, source="epht")

    got = obs.latest(quantity="heat_deaths")
    supr = got[got["censored"]].iloc[0]
    assert pd.isna(supr["value"])
    assert supr["censor_lo"] == 0.0
    assert supr["censor_hi"] == 10.0


def test_observacion_revisada_crea_revision(con):
    obs = ObservationStore(con)
    df = pd.DataFrame(
        [{"geo_level": "state", "geoid": "04", "target_date": "2023-01-01",
          "quantity": "hri_rate", "value": 48.1}]
    )
    obs.put(df, source="epht", known_at=dt.datetime(2024, 1, 1))

    df2 = df.copy()
    df2["value"] = 49.4
    stats = obs.put(df2, source="epht", known_at=dt.datetime(2025, 1, 1))
    assert stats["written"] == 1

    # Corte pasado: la observacion vieja sigue siendo recuperable.
    vieja = obs.latest(quantity="hri_rate", known_at=dt.datetime(2024, 6, 1))
    assert vieja.iloc[0]["value"] == pytest.approx(48.1)

    nueva = obs.latest(quantity="hri_rate")
    assert nueva.iloc[0]["value"] == pytest.approx(49.4)
