from __future__ import annotations

import numpy as np
import pytest
from scipy import stats

from toolbox.distributions import fit_moments, rectnorm, conditional


def _sample_mean_std(x: np.ndarray, ddof: int = 1) -> tuple[float, float]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(x.mean()), float(x.std(ddof=ddof))


@pytest.mark.parametrize(
    "key, dist",
    [
        ("norm", stats.norm),
        ("laplace", stats.laplace),
        ("uniform", stats.uniform),
        ("gamma", stats.gamma),
    ],
)
def test_fit_moments_accepts_key_and_dist_object(key: str, dist: stats.rv_continuous) -> None:
    rng = np.random.default_rng(123)

    # create a "true" frozen distribution for sampling
    if key == "gamma":
        true = dist(3.0, loc=0.0, scale=2.0)
    elif key == "uniform":
        true = dist(loc=-2.0, scale=5.0)
    else:
        true = dist(loc=1.25, scale=0.75)

    x = true.rvs(size=50_000, random_state=rng)
    mean, std = _sample_mean_std(x, ddof=1)

    # Fit via key
    fitted_from_key = fit_moments.fit(key, x, ddof=1)
    assert abs(float(fitted_from_key.mean()) - mean) < 0.05
    assert abs(float(fitted_from_key.std()) - std) < 0.05

    # Fit via SciPy dist object
    fitted_from_obj = fit_moments.fit(dist, x, ddof=1)
    assert abs(float(fitted_from_obj.mean()) - mean) < 0.05
    assert abs(float(fitted_from_obj.std()) - std) < 0.05


def test_fit_moments_raises_on_non_finite() -> None:
    x = np.array([1.0, 2.0, np.inf], dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite"):
        fit_moments.fit("norm", x)

    y = np.array([1.0, np.nan, 3.0], dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite"):
        fit_moments.fit(stats.norm, y)


def test_rectnorm_fit_moments_roundtrip_mean_std() -> None:
    rng = np.random.default_rng(456)

    # sample from rectnorm(mu, sigma)
    true = rectnorm(mu=0.5, sigma=1.5)
    x = true.rvs(size=80_000, random_state=rng)

    mean, std = _sample_mean_std(x, ddof=1)

    # fit via key
    fitted = fit_moments.fit("rectnorm", x, ddof=1)

    # The numeric inversion is approximate; allow a slightly looser tolerance than built-ins.
    assert abs(float(fitted.mean()) - mean) < 0.06
    assert abs(float(fitted.std()) - std) < 0.06


def test_conditional_rvs_zero_fraction() -> None:
    rng = np.random.default_rng(789)

    # zero-inflated normal: P(0) = 1-p
    p = 0.3
    base = stats.norm
    dist = conditional(base, name="norm_cond")
    frozen = dist(p, 0.0, 1.0)  # p, loc, scale

    x = frozen.rvs(size=200_000, random_state=rng)
    zero_frac = float((np.asarray(x) == 0.0).mean())

    # should be close to 1-p (base is continuous so P(base==0) ~ 0)
    assert abs(zero_frac - (1.0 - p)) < 0.01
    