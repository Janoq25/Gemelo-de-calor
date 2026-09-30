"""Pruebas de L5: calibracion retrospectiva y deteccion de deriva."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from xdt.twin.calibrate import Calibrator, calibration_metrics
from xdt.twin.registry import ObservationStore, Prediction, PredictionRegistry

T_STATE = dt.datetime(2023, 7, 14, 6, 0)  # estado usado por la prediccion
T_ISSUED = dt.datetime(2023, 7, 14, 7, 0)
T_OBS = dt.datetime(2024, 3, 1)  # EPHT publica el dato meses despues


def _pred(geoid="04", value=50.0, day=15, **kw) -> Prediction:
    base = dict(
        geo_level="state", geoid=geoid, target_date=dt.date(2023, 7, day),
        quantity="hri_rate", value=value, model_name="lgbm", model_version="0.1.0",
        input_hash=f"h-{geoid}-{day}-{value}", state_known_at=T_STATE, issued_at=T_ISSUED,
    )
    base.update(kw)
    return Prediction(**base)


def _obs(con, rows, known_at=T_OBS):
    ObservationStore(con).put(
        pd.DataFrame([
            {"geo_level": "state", "quantity": "hri_rate", "target_date": f"2023-07-{d:02d}",
             "geoid": g, "value": v, **extra}
            for g, d, v, extra in rows
        ]),
        source="epht", known_at=known_at,
    )


def test_pareja_genuina_y_error(con):
    PredictionRegistry(con).log([_pred(value=50.0)])
    _obs(con, [("04", 15, 48.0, {})])

    r = Calibrator(con).run(model_name="lgbm", quantity="hri_rate", persist=False)
    assert r.n_pairs == 1 and r.n_postdiction == 0
    assert r.metrics["mae"] == pytest.approx(2.0)
    assert r.metrics["bias"] == pytest.approx(2.0)


def test_postdiccion_se_excluye_y_se_cuenta(con):
    """Una prediccion hecha con un estado posterior a la observacion pudo
    haberla visto: no es un pronostico y no puede calibrar nada."""
    reg = PredictionRegistry(con)
    reg.log([_pred(geoid="04", value=50.0)])
    reg.log([_pred(geoid="22", value=57.0, state_known_at=dt.datetime(2024, 4, 1),
                   issued_at=dt.datetime(2024, 4, 1))])
    _obs(con, [("04", 15, 48.0, {}), ("22", 15, 57.7, {})])

    r = Calibrator(con).run(model_name="lgbm", quantity="hri_rate", persist=False)
    assert r.n_pairs == 1
    assert r.n_postdiction == 1


def test_revision_posterior_no_convierte_en_postdiccion(con):
    """Lo que cuenta es la PRIMERA vez que se conocio la observacion: una
    revision de 2025 no invalida un pronostico emitido en 2023."""
    PredictionRegistry(con).log([_pred(value=50.0)])
    _obs(con, [("04", 15, 48.0, {})], known_at=T_OBS)
    _obs(con, [("04", 15, 49.0, {})], known_at=dt.datetime(2025, 1, 1))

    r = Calibrator(con).run(model_name="lgbm", quantity="hri_rate", persist=False)
    assert r.n_pairs == 1 and r.n_postdiction == 0
    assert r.metrics["mae"] == pytest.approx(1.0)  # contra la revision vigente


def test_calibracion_pasada_es_reproducible(con):
    """Con obs_known_at se recupera exactamente lo que el gemelo sabia."""
    PredictionRegistry(con).log([_pred(value=50.0)])
    _obs(con, [("04", 15, 48.0, {})], known_at=T_OBS)
    _obs(con, [("04", 15, 49.0, {})], known_at=dt.datetime(2025, 1, 1))

    cal = Calibrator(con)
    antes = cal.run(model_name="lgbm", quantity="hri_rate", persist=False,
                    obs_known_at=dt.datetime(2024, 6, 1))
    assert antes.metrics["mae"] == pytest.approx(2.0)

    nada = cal.run(model_name="lgbm", quantity="hri_rate", persist=False,
                   obs_known_at=dt.datetime(2023, 12, 1))
    assert nada.n_pairs == 0  # la observacion aun no existia


def test_censura_entra_como_intervalo(con):
    """§14 R2: la celda suprimida aporta 'el valor esta entre 0 y 10'."""
    PredictionRegistry(con).log([_pred(geoid="56", value=5.0),
                                 _pred(geoid="50", value=12.0)])
    cens = {"censored": True, "censor_lo": 0.0, "censor_hi": 10.0}
    _obs(con, [("56", 15, None, cens), ("50", 15, None, cens)])

    r = Calibrator(con).run(model_name="lgbm", quantity="hri_rate", persist=False)
    assert r.n_pairs == 2 and r.n_censored == 2
    assert r.metrics["mae"] == pytest.approx((0.0 + 2.0) / 2)
    assert r.metrics["censored_hit_rate"] == pytest.approx(0.5)
    assert r.metrics["bias"] is None  # sin observaciones puntuales


def test_contrafactuales_nunca_se_calibran(con):
    PredictionRegistry(con).log([_pred(value=50.0),
                                 _pred(value=90.0, scenario_id="scn-apagon")])
    _obs(con, [("04", 15, 48.0, {})])
    r = Calibrator(con).run(model_name="lgbm", quantity="hri_rate", persist=False)
    assert r.n_pairs == 1 and r.metrics["mae"] == pytest.approx(2.0)


def test_se_usa_el_pronostico_mas_reciente_por_objetivo(con):
    PredictionRegistry(con).log([
        _pred(value=60.0, issued_at=dt.datetime(2023, 7, 10), state_known_at=T_STATE),
        _pred(value=49.0, issued_at=dt.datetime(2023, 7, 14, 8), state_known_at=T_STATE),
    ])
    _obs(con, [("04", 15, 48.0, {})])
    r = Calibrator(con).run(model_name="lgbm", quantity="hri_rate", persist=False)
    assert r.n_pairs == 1 and r.metrics["mae"] == pytest.approx(1.0)


# ------------------------------------------------------------------ deriva
def _panel(con, bias: float, day0: int = 1, n: int = 12):
    PredictionRegistry(con).log([
        _pred(geoid="04", day=day0 + k, value=40.0 + k + bias + (0.3 if k % 2 else -0.3))
        for k in range(n)
    ])
    _obs(con, [("04", day0 + k, 40.0 + k, {}) for k in range(n)])


def test_sesgo_sistematico_material_marca_deriva(con):
    _panel(con, bias=8.0)
    r = Calibrator(con).run(model_name="lgbm", quantity="hri_rate", persist=False)
    assert r.drift_flag
    assert "sesgo" in r.drift_reasons[0]


def test_sesgo_minusculo_no_marca_deriva_aunque_sea_significativo(con):
    _panel(con, bias=0.5)  # |t| alto, pero ~1 % de la media
    r = Calibrator(con).run(model_name="lgbm", quantity="hri_rate", persist=False)
    assert abs(r.metrics["bias_t"]) > 1.96
    assert not r.drift_flag


def test_degradacion_respecto_al_historial(con):
    cal = Calibrator(con)
    _panel(con, bias=0.0, day0=1, n=12)
    ok = cal.run(model_name="lgbm", quantity="hri_rate", window_end="2023-07-12")
    assert not ok.drift_flag

    _panel(con, bias=0.0, day0=13, n=12)  # mismos errores: sin deriva
    PredictionRegistry(con).log([_pred(geoid="22", day=20, value=80.0)])
    _obs(con, [("22", 20, 40.0, {})])  # un error grande en la ventana nueva
    malo = cal.run(model_name="lgbm", quantity="hri_rate", window_start="2023-07-13")
    assert malo.metrics["reference_mae"] == pytest.approx(ok.metrics["mae"])
    assert any("MAE" in m for m in malo.drift_reasons)

    hist = cal.history("lgbm")
    assert len(hist) == 2 and hist["drift_flag"].tolist() == [False, True]


def test_metricas_de_poisson_y_pendiente():
    pairs = pd.DataFrame({
        "pred": [1.0, 2.0, 3.0, 4.0], "obs": [1.0, 2.0, 3.0, 4.0],
        "censored": False, "censor_lo": None, "censor_hi": None,
        "pred_lo": [0.5, 1.5, 3.5, 3.0], "pred_hi": [1.5, 2.5, 4.5, 5.0],
    })
    m = calibration_metrics(pairs)
    assert m["calib_slope"] == pytest.approx(1.0)
    assert m["calib_intercept"] == pytest.approx(0.0, abs=1e-9)
    assert m["poisson_deviance"] == pytest.approx(0.0, abs=1e-12)
    assert m["pi_coverage"] == pytest.approx(0.75)  # 3.0 cae fuera de [3.5, 4.5]
