from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm as _norm
from scipy.stats import rv_continuous


def _rectnorm_mean_std(mu: np.ndarray | float, sigma: np.ndarray | float) -> tuple[np.ndarray | float, np.ndarray | float]:
    z = -mu / sigma
    phi = _norm.pdf(z)
    Phi = _norm.cdf(z)

    ex = sigma * phi + mu * (1.0 - Phi)
    ex2 = sigma * mu * phi + (sigma**2 + mu**2) * (1.0 - Phi)
    var = ex2 - ex**2
    std = np.sqrt(np.maximum(var, 0.0))
    return ex, std


def _obj(params: np.ndarray, target: np.ndarray) -> float:
    mu = float(params[0])
    sigma = float(params[1])
    if sigma <= 0.0:
        return 1e30

    ex, sd = _rectnorm_mean_std(mu, sigma)
    d0 = float(ex) - float(target[0])
    d1 = float(sd) - float(target[1])
    return float(d0 * d0 + d1 * d1)


class rectnorm_gen(rv_continuous):
    """
    Rectified normal distribution: X = max(0, Z), Z ~ Normal(mu, sigma).
    """

    def _argcheck(self, *args: Any) -> bool:
        if len(args) != 2:
            return False
        sigma = np.asarray(args[1])
        return bool(np.all(sigma > 0))

    def _rvs(
            self,
            *args: Any,
            size: int | tuple[int, ...] | None = None,
            random_state: Any = None,
    ) -> np.ndarray:
        if len(args) != 2:
            raise ValueError("rectnorm requires parameters (mu, sigma)")
        mu: np.ndarray | float = args[0]
        sigma: np.ndarray | float = args[1]

        y = _norm.rvs(loc=mu, scale=sigma, size=size, random_state=random_state)
        y = np.asarray(y, dtype=np.float64)
        y[y < 0.0] = 0.0
        return y

    def _pdf(self, x: Any, *args: Any) -> np.ndarray:
        if len(args) != 2:
            raise ValueError("rectnorm requires parameters (mu, sigma)")
        mu: np.ndarray | float = args[0]
        sigma: np.ndarray | float = args[1]

        x = np.asarray(x, dtype=np.float64)
        y = np.zeros_like(x, dtype=np.float64)

        y[x == 0.0] = _norm.cdf(0.0, loc=mu, scale=sigma)
        m = x > 0.0
        if np.any(m):
            y[m] = _norm.pdf(x[m], loc=mu, scale=sigma)
        return y

    def _cdf(self, x: Any, *args: Any) -> np.ndarray:
        if len(args) != 2:
            raise ValueError("rectnorm requires parameters (mu, sigma)")
        mu: np.ndarray | float = args[0]
        sigma: np.ndarray | float = args[1]

        x = np.asarray(x, dtype=np.float64)
        y = np.zeros_like(x, dtype=np.float64)

        m = x >= 0.0
        if np.any(m):
            y[m] = _norm.cdf(x[m], loc=mu, scale=sigma)
        return y

    def _stats(self, *args: Any) -> tuple[np.ndarray | float, np.ndarray | float, None, None]:
        if len(args) != 2:
            raise ValueError("rectnorm requires parameters (mu, sigma)")
        mu: np.ndarray | float = args[0]
        sigma: np.ndarray | float = args[1]

        ex, sd = _rectnorm_mean_std(mu, sigma)
        return ex, sd * sd, None, None

    @staticmethod
    def params_from_mean_std(mean: float, std: float) -> tuple[float, float]:
        target = np.asarray([mean, std], dtype=np.float64)
        x0 = target.copy()

        res = minimize(_obj, x0, args=(target,))
        if res.success:
            mu = float(res.x[0])
            sigma = float(res.x[1])
            if sigma > 0.0:
                return mu, sigma

        return float(target[0]), max(float(target[1]), 1e-12)


rectnorm = rectnorm_gen(name="rectnorm", shapes="mu,sigma")
