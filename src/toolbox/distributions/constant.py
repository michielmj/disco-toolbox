from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rv_continuous


class constant_gen(rv_continuous):
    """
    Degenerate (Dirac) distribution with all mass at a single value `c`.

    Parameters (shape):
        c: float

    Usage:
        rv0 = constant(0.0)
        rv5 = constant(5.0)
        x = rv5.rvs(size=10, random_state=rng)
    """

    def _argcheck(self, *args: Any) -> bool:
        # Single shape parameter c (can be any finite real number).
        return len(args) == 1 and np.isfinite(np.asarray(args[0])).all()

    def _rvs(self, *args: Any, size: int | tuple[int, ...] | None = None, random_state: Any = None) -> np.ndarray:
        c = float(args[0])
        if size is None:
            return np.asarray(c, dtype=np.float64)
        return np.full(size, c, dtype=np.float64)

    def _pdf(self, x: Any, *args: Any) -> np.ndarray:
        c = float(args[0])
        xx = np.asarray(x, dtype=np.float64)
        y = np.zeros_like(xx, dtype=np.float64)
        y[xx == c] = 1.0
        return y

    def _cdf(self, x: Any, *args: Any) -> np.ndarray:
        c = float(args[0])
        xx = np.asarray(x, dtype=np.float64)
        y = np.zeros_like(xx, dtype=np.float64)
        y[xx >= c] = 1.0
        return y

    def _stats(self, *args: Any) -> tuple[float, float, float, float]:
        c = float(args[0])
        return c, 0.0, 0.0, 0.0

    @staticmethod
    def params_from_mean_std(mean: float, std: float) -> tuple[float]:
        # For a constant distribution std should be 0 (within tolerance).
        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError("mean/std must be finite.")
        if std < 0:
            raise ValueError("std must be non-negative.")
        if std > 1e-12:
            # Keep this strict to avoid silently producing nonsense fits.
            raise ValueError("constant distribution requires std == 0.")
        return (float(mean),)


constant = constant_gen(name="constant", shapes="c")
