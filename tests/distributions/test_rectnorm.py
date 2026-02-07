from __future__ import annotations

import numpy as np
import pytest

from toolbox.distributions import rectnorm


@pytest.mark.parametrize(
    "mean,std",
    [
        (0.1, 0.2),
        (0.5, 0.8),
        (1.0, 1.0),
        (2.0, 1.5),
    ],
)
def test_rectnorm_params_from_mean_std_roundtrip(mean: float, std: float) -> None:
    mu, sigma = rectnorm.params_from_mean_std(mean, std)
    assert sigma > 0.0

    rv = rectnorm(mu, sigma)

    # rectnorm.stats() is implemented analytically; check roundtrip at distribution level
    m = float(rv.mean())
    s = float(rv.std())

    # Numeric inversion + floating math => allow modest tolerance
    assert abs(m - mean) < 1e-3
    assert abs(s - std) < 1e-3


def test_rectnorm_argcheck_sigma_positive() -> None:
    assert not rectnorm._argcheck(0.0, 0.0)
    assert not rectnorm._argcheck(0.0, -1.0)
    assert rectnorm._argcheck(0.0, 1.0)
