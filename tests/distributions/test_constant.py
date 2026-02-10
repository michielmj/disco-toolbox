from __future__ import annotations

import numpy as np
import pytest

from toolbox.distributions.constant import constant


def test_constant_mean_scalar_param() -> None:
    m = constant.mean(2.0)
    assert float(m) == 2.0


def test_constant_std_scalar_param_is_zero() -> None:
    s = constant.std(2.0)
    assert float(s) == 0.0


def test_constant_cdf_vector_x_scalar_param() -> None:
    x = np.asarray([-1.0, 2.0, 3.0], dtype=np.float64)
    y = constant.cdf(x, 2.0)
    assert isinstance(y, np.ndarray)
    assert y.shape == x.shape
    assert np.allclose(y, [0.0, 1.0, 1.0])


def test_constant_pdf_vector_x_scalar_param() -> None:
    x = np.asarray([1.0, 2.0, 2.0, 3.0], dtype=np.float64)
    y = constant.pdf(x, 2.0)
    assert y.shape == x.shape
    assert np.allclose(y, [0.0, 1.0, 1.0, 0.0])


def test_constant_rvs_size_none_scalar_param() -> None:
    y = constant.rvs(5.0, size=None)
    # SciPy may return a scalar or 0-d array; accept both
    assert float(np.asarray(y)) == 5.0


def test_constant_rvs_size_scalar_param() -> None:
    rng = np.random.default_rng(0)
    y = constant.rvs(5.0, size=10, random_state=rng)
    assert isinstance(y, np.ndarray)
    assert y.shape == (10,)
    assert np.allclose(y, 5.0)


def test_constant_params_from_mean_std() -> None:
    (c,) = constant.params_from_mean_std(2.5, 0.0)
    assert c == 2.5

    with pytest.raises(ValueError):
        constant.params_from_mean_std(2.5, 0.1)

    with pytest.raises(ValueError):
        constant.params_from_mean_std(2.5, -1.0)


def test_constant_multi_c_via_explicit_loop() -> None:
    # If you want "vector parameters", do it explicitly/predictably.
    cs = [1.0, 2.0, 3.0]
    means = np.asarray([constant.mean(c) for c in cs], dtype=np.float64)
    assert np.allclose(means, cs)
