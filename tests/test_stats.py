"""Pruebas estadisticas de M5: cada una contra un resultado conocido."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

pytest.importorskip("sklearn")
pytest.importorskip("statsmodels")

from xdt.harmonize.centroids import top_counties_by_population  # noqa: E402
from xdt.models.ladder import ThresholdRule  # noqa: E402
from xdt.validate import stats as st  # noqa: E402


@pytest.fixture
def scored():
    rng = np.random.default_rng(1)
    n = 800
    y = rng.binomial(1, 0.3, n)
    good = y * 1.0 + rng.normal(0, 0.8, n)
    bad = y * 0.3 + rng.normal(0, 0.8, n)
    return y, good, bad


def test_delong_auc_matches_sklearn(scored):
    y, good, bad = scored
    r = st.delong_test(y, good, bad)
    assert r.auc_a == pytest.approx(roc_auc_score(y, good))
    assert r.auc_b == pytest.approx(roc_auc_score(y, bad))
    assert r.p_value < 1e-6 and r.ci_low > 0


def test_delong_identical_models_not_different(scored):
    y, good, _ = scored
    r = st.delong_test(y, good, good + 1e-9 * np.arange(len(good)))
    assert r.p_value > 0.5


def test_delong_ci_contains_auc(scored):
    y, good, _ = scored
    auc, lo, hi = st.delong_ci(y, good)
    assert lo < auc < hi


def test_holm_is_monotone_and_bounded():
    t = st.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert list(t["p_holm"]) == pytest.approx([0.03, 0.06, 0.06])
    assert list(t["rechaza_H0"]) == [True, False, False]


def test_nadeau_bengio_inflates_variance():
    diffs = [0.02, 0.03, 0.01, 0.025, 0.015]
    t_naive = np.mean(diffs) / (np.std(diffs, ddof=1) / np.sqrt(5))
    t_nb, _ = st.corrected_resampled_t(diffs, n_train=900, n_test=100)
    assert 0 < t_nb < t_naive


def test_permutation_detects_signal_and_its_absence(scored):
    y, good, _ = scored
    groups = np.repeat(np.arange(8), 100)
    _, p_sig = st.circular_shift_permutation_test(y, good, groups, roc_auc_score, n_perm=99)
    noise = np.random.default_rng(3).normal(size=len(y))
    _, p_null = st.circular_shift_permutation_test(y, noise, groups, roc_auc_score, n_perm=99)
    assert p_sig <= 0.02 and p_null > 0.05


def test_calibration_of_true_probabilities():
    rng = np.random.default_rng(2)
    p = rng.uniform(0.05, 0.6, 5000)
    y = rng.binomial(1, p)
    c = st.calibration(y, p)
    assert c.slope == pytest.approx(1.0, abs=0.15)
    assert abs(c.intercept) < 0.1 and c.spiegelhalter_p > 0.01


def test_diebold_mariano_sign():
    idx = pd.date_range("2023-05-01", periods=120)
    rng = np.random.default_rng(4)
    la = pd.Series(rng.normal(0.10, 0.02, 120), idx)
    lb = pd.Series(rng.normal(0.14, 0.02, 120), idx)
    dm, p = st.diebold_mariano(la, lb)
    assert dm < 0 and p < 0.01


def test_friedman_ranks_and_cd():
    s = pd.DataFrame({"a": [0.8, 0.82, 0.79, 0.81, 0.83],
                      "b": [0.7, 0.72, 0.71, 0.69, 0.7],
                      "c": [0.6, 0.61, 0.62, 0.6, 0.59]})
    r = st.friedman_nemenyi(s)
    assert r.mean_ranks == {"a": 1.0, "b": 2.0, "c": 3.0}
    assert r.p_value < 0.05 and r.critical_difference > 0


def test_block_bootstrap_wider_than_row_bootstrap():
    rng = np.random.default_rng(5)
    blocks = np.repeat(np.arange(40), 25)
    # Dependencia fuerte dentro del bloque, en el desenlace y en la puntuacion.
    shift = rng.normal(0, 1.5, 40)[blocks]
    y = rng.binomial(1, rng.uniform(0.05, 0.8, 40)[blocks])
    p = y * 0.5 + shift + rng.normal(0, 0.5, len(blocks))
    _, lo_b, hi_b = st.block_bootstrap_ci(y, p, blocks, roc_auc_score, n_boot=300)
    _, lo_r, hi_r = st.block_bootstrap_ci(y, p, np.arange(len(y)), roc_auc_score, n_boot=300)
    assert (hi_b - lo_b) > (hi_r - lo_r)


def test_threshold_rule_picks_youden_cut():
    x = pd.DataFrame({"hi_excess_p95": np.r_[np.linspace(-10, -1, 50), np.linspace(1, 10, 50)]})
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    m = ThresholdRule().fit(x, y)
    assert -1 < m.threshold_ <= 1
    assert m.p_above_ == 1.0 and m.p_below_ == 0.0
    assert roc_auc_score(y, m.predict_proba(x)[:, 1]) == 1.0


def test_top_counties_reach_coverage():
    cen = pd.DataFrame({
        "geoid": ["01001", "01002", "01003", "02001", "02002"],
        "state_fips": ["01", "01", "01", "02", "02"],
        "population": [60, 30, 10, 50, 50],
    })
    out = top_counties_by_population(cen, states=["01", "02"], coverage=0.8)
    assert out == ["01001", "01002", "02001", "02002"]
