"""M5 · Pruebas estadisticas robustas para validar los modelos del gemelo (§10).

Por que estas y no las de manual
--------------------------------
El panel es estado-dia: los dias consecutivos de un mismo estado se parecen
(autocorrelacion) y los estados de una region comparten clima (dependencia
espacial). Casi todas las pruebas clasicas suponen observaciones
independientes y, con estos datos, dan p-valores demasiado pequenos e
intervalos demasiado estrechos. Cada funcion de aqui corrige una de esas
violaciones:

=========================  ==================================================
Pregunta                   Prueba
=========================  ==================================================
AUC de A vs AUC de B       DeLong (curvas ROC pareadas sobre los mismos dias)
IC de cualquier metrica    Bootstrap por bloques (estado x semana)
k modelos en n bloques     Friedman + Nemenyi (Demsar, 2006)
A vs B sobre pliegues CV   t corregida de Nadeau-Bengio (pliegues solapados)
Aprende algo real?         Permutacion por desplazamiento circular por estado
Probabilidad honesta?      Pendiente/intercepto de calibracion + z Spiegelhalter
Serie de perdidas A vs B   Diebold-Mariano con varianza HAC (Newey-West)
Muchas comparaciones       Holm-Bonferroni
=========================  ==================================================
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import stats

# ------------------------------------------------------------------ DeLong


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    n = len(x)
    ranks = np.empty(n)
    i = 0
    while i < n:
        j = i
        while j < n and xs[j] == xs[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n)
    out[order] = ranks
    return out


def _delong_components(y: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUCs y matriz de covarianza de DeLong (Sun & Xu, 2014, version rapida)."""
    pos, neg = scores[:, y == 1], scores[:, y == 0]
    m, n = pos.shape[1], neg.shape[1]
    k = scores.shape[0]
    tx = np.array([_midrank(p) for p in pos])
    ty = np.array([_midrank(q) for q in neg])
    tz = np.array([_midrank(np.r_[p, q]) for p, q in zip(pos, neg, strict=True)])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01).reshape(k, k)
    sy = np.cov(v10).reshape(k, k)
    return aucs, sx / m + sy / n


@dataclass(frozen=True)
class DelongResult:
    auc_a: float
    auc_b: float
    diff: float
    ci_low: float
    ci_high: float
    z: float
    p_value: float


def delong_test(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray) -> DelongResult:
    """Prueba de DeLong para dos AUC correlacionadas (mismos casos)."""
    y = np.asarray(y).astype(int)
    aucs, cov = _delong_components(y, np.vstack([p_a, p_b]).astype(float))
    diff = aucs[0] - aucs[1]
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    se = np.sqrt(max(var, 1e-300))
    z = diff / se
    return DelongResult(
        auc_a=float(aucs[0]), auc_b=float(aucs[1]), diff=float(diff),
        ci_low=float(diff - 1.96 * se), ci_high=float(diff + 1.96 * se),
        z=float(z), p_value=float(2 * stats.norm.sf(abs(z))),
    )


def delong_ci(y: np.ndarray, p: np.ndarray, level: float = 0.95) -> tuple[float, float, float]:
    aucs, cov = _delong_components(np.asarray(y).astype(int), np.asarray(p, float)[None, :])
    se = np.sqrt(cov[0, 0])
    zq = stats.norm.ppf(0.5 + level / 2)
    return float(aucs[0]), float(aucs[0] - zq * se), float(aucs[0] + zq * se)


# ------------------------------------------------------ bootstrap por bloques


def block_bootstrap_ci(
    y: np.ndarray,
    p: np.ndarray,
    blocks: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int = 1000,
    level: float = 0.95,
    seed: int = 0,
) -> tuple[float, float, float]:
    """IC percentil remuestreando bloques completos (p.ej. estado x semana).

    Remuestrear filas sueltas trataria como independientes dias que no lo son
    y estrecharia el intervalo. Remuestrear bloques conserva la dependencia
    dentro de cada bloque.
    """
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    codes, uniq = pd.factorize(blocks)
    idx_by_block = [np.flatnonzero(codes == k) for k in range(len(uniq))]
    point = metric(y, p)
    vals = []
    for _ in range(n_boot):
        pick = rng.integers(0, len(idx_by_block), len(idx_by_block))
        ii = np.concatenate([idx_by_block[k] for k in pick])
        if y[ii].min() == y[ii].max():
            continue
        vals.append(metric(y[ii], p[ii]))
    a = (1 - level) / 2
    return float(point), float(np.quantile(vals, a)), float(np.quantile(vals, 1 - a))


# ------------------------------------------------------ Friedman + Nemenyi


@dataclass(frozen=True)
class FriedmanResult:
    statistic: float
    p_value: float
    mean_ranks: dict[str, float]
    critical_difference: float
    n_blocks: int


def friedman_nemenyi(scores: pd.DataFrame, alpha: float = 0.05) -> FriedmanResult:
    """`scores`: filas = bloques (p.ej. ano x region), columnas = modelos.

    Rango 1 = mejor (metrica mayor). Dos modelos difieren si sus rangos medios
    se separan mas que la diferencia critica (CD) de Nemenyi.
    """
    s = scores.dropna()
    k, n = s.shape[1], s.shape[0]
    stat, p = stats.friedmanchisquare(*[s[c].to_numpy() for c in s.columns])
    ranks = s.rank(axis=1, ascending=False).mean()
    q = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q * np.sqrt(k * (k + 1) / (6.0 * n))
    return FriedmanResult(float(stat), float(p), ranks.round(3).to_dict(), float(cd), n)


# ---------------------------------------------------- Nadeau-Bengio


def corrected_resampled_t(
    diffs: Sequence[float], n_train: int, n_test: int
) -> tuple[float, float]:
    """t de Student corregida (Nadeau & Bengio, 2003) sobre diferencias por pliegue.

    Los pliegues de CV comparten datos de entrenamiento, asi que sus
    diferencias estan correlacionadas; la correccion infla la varianza por
    n_test/n_train. Devuelve (t, p bilateral).
    """
    d = np.asarray(diffs, float)
    k = len(d)
    var = d.var(ddof=1)
    if var == 0:
        return float("inf") if d.mean() else 0.0, 0.0 if d.mean() else 1.0
    t = d.mean() / np.sqrt((1.0 / k + n_test / n_train) * var)
    return float(t), float(2 * stats.t.sf(abs(t), k - 1))


# ------------------------------------------------------ permutacion


def circular_shift_permutation_test(
    y: np.ndarray,
    p: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_perm: int = 500,
    seed: int = 0,
) -> tuple[float, float]:
    """H0: la prediccion no guarda relacion con el desenlace.

    En vez de barajar dias sueltos (que destruiria la autocorrelacion y haria
    la prueba demasiado facil de rechazar), desplaza circularmente la serie de
    predicciones de cada estado una cantidad aleatoria. Conserva la estructura
    temporal de ambas series y rompe solo su alineacion.
    Devuelve (metrica observada, p-valor).
    """
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p, float)
    obs = metric(y, p)
    idx_groups = [np.flatnonzero(groups == g) for g in pd.unique(groups)]
    hits = 0
    for _ in range(n_perm):
        pp = p.copy()
        for ii in idx_groups:
            if len(ii) > 1:
                pp[ii] = np.roll(p[ii], rng.integers(1, len(ii)))
        hits += metric(y, pp) >= obs
    return float(obs), float((hits + 1) / (n_perm + 1))


# ------------------------------------------------------ calibracion


@dataclass(frozen=True)
class CalibrationResult:
    slope: float
    intercept: float
    spiegelhalter_z: float
    spiegelhalter_p: float
    brier: float


def calibration(y: np.ndarray, p: np.ndarray) -> CalibrationResult:
    """Pendiente e intercepto de calibracion (Cox) + prueba z de Spiegelhalter.

    Pendiente 1 e intercepto 0 = calibracion perfecta. Pendiente < 1 indica
    predicciones demasiado extremas (sobreajuste). Spiegelhalter contrasta
    H0: el modelo esta calibrado; a diferencia de Hosmer-Lemeshow no depende
    de como se agrupen los deciles.
    """
    import statsmodels.api as sm

    y = np.asarray(y, float)
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    lp = np.log(p / (1 - p))
    fit = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    slope = float(fit.params[1])
    fit0 = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(), offset=lp).fit()
    intercept = float(fit0.params[0])
    num = np.sum((y - p) * (1 - 2 * p))
    den = np.sqrt(np.sum((1 - 2 * p) ** 2 * p * (1 - p)))
    z = float(num / den)
    return CalibrationResult(slope, intercept, z, float(2 * stats.norm.sf(abs(z))),
                             float(np.mean((p - y) ** 2)))


# ------------------------------------------------------ Diebold-Mariano


def diebold_mariano(
    loss_a: pd.Series, loss_b: pd.Series, lag: int | None = None
) -> tuple[float, float]:
    """DM sobre dos series temporales de perdida (indice = fecha).

    d_t = perdida_A - perdida_B. Negativo favorece a A. La varianza de la
    media usa Newey-West (HAC) porque los errores de dias consecutivos estan
    correlacionados. Devuelve (estadistico, p bilateral).
    """
    d = (loss_a - loss_b).dropna().to_numpy()
    t = len(d)
    lag = lag if lag is not None else int(np.floor(t ** (1 / 3)))
    dc = d - d.mean()
    gamma0 = np.dot(dc, dc) / t
    var = gamma0
    for h in range(1, lag + 1):
        w = 1 - h / (lag + 1)
        var += 2 * w * np.dot(dc[h:], dc[:-h]) / t
    dm = d.mean() / np.sqrt(var / t)
    return float(dm), float(2 * stats.norm.sf(abs(dm)))


# ------------------------------------------------------ Holm


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> pd.DataFrame:
    """Correccion de Holm-Bonferroni: controla el error de familia."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    rows, running = [], 0.0
    for i, (name, p) in enumerate(items):
        adj = min(1.0, max(running, (m - i) * p))
        running = adj
        rows.append({"comparacion": name, "p": p, "p_holm": adj, "rechaza_H0": adj < alpha})
    return pd.DataFrame(rows)


def as_dict(obj: object) -> dict:
    return asdict(obj)  # type: ignore[call-overload]
