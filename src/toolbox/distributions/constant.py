from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rv_continuous


class constant_gen(rv_continuous):
    """
    Degenerate (Dirac) distribution with all mass at value(s) `c`.

    Shape parameter:
        c : float or array-like of floats

    Supports SciPy-style vector parameters, e.g.:
        constant.mean([1,2,3]) -> array([1,2,3])
    """

    def _argcheck(self, *args: Any) -> bool:
        # Single shape parameter c; accept scalars and arrays; require finite.
        if len(args) != 1:
            return False
        c = np.asarray(args[0])
        return bool(np.all(np.isfinite(c)))

    def _rvs(self,
             *args: Any,
             size: int | tuple[int, ...] | None = None,
             random_state: Any = None,
        ) -> np.ndarray:
        c = np.asarray(args[0], dtype=np.float64)

        # SciPy convention: if size is None, return broadcasted parameter shape.
        if size is None:
            # Return c with its natural (possibly broadcasted) shape
            return np.array(c, copy=True, dtype=np.float64)

        # If size is provided, it overrides. Broadcast c to requested size.
        # If incompatible, numpy will raise ValueError, matching SciPy-style behavior.
        return np.broadcast_to(c, size).astype(np.float64, copy=False)

    def _pdf(self, x: Any, *args: Any) -> np.ndarray:
        c = np.asarray(args[0], dtype=np.float64)
        xx = np.asarray(x, dtype=np.float64)

        y = np.zeros(np.broadcast(xx, c).shape, dtype=np.float64)
        # Broadcasted comparison
        y[np.equal(xx, c)] = 1.0
        return y

    def _cdf(self, x: Any, *args: Any) -> np.ndarray:
        c = np.asarray(args[0], dtype=np.float64)
        xx = np.asarray(x, dtype=np.float64)

        y = np.zeros(np.broadcast(xx, c).shape, dtype=np.float64)
        y[np.greater_equal(xx, c)] = 1.0
        return y

    def _stats(self, *args: Any) -> tuple[np.ndarray | float, np.ndarray | float, np.ndarray | float, np.ndarray | float]:
        c = np.asarray(args[0], dtype=np.float64)

        if c.ndim == 0:
            cc = float(c.item())
            return cc, 0.0, 0.0, 0.0

        z = np.zeros_like(c, dtype=np.float64)
        return c, z, z, z

    @staticmethod
    def params_from_mean_std(mean: float, std: float) -> tuple[float]:
        # For constant, std must be ~0 (strict), and c := mean.
        if not np.isfinite(mean) or not np.isfinite(std):
            raise ValueError("mean/std must be finite.")
        if std < 0:
            raise ValueError("std must be non-negative.")
        if std > 1e-12:
            raise ValueError("constant distribution requires std == 0.")
        return (float(mean),)


constant = constant_gen(name="constant", shapes="c")
