from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np
from scipy.stats import rv_continuous


def _ensure_finite_1d(data: Sequence[float] | np.ndarray) -> np.ndarray:
    x = np.asarray(data, dtype=np.float64).reshape(-1)
    if not np.isfinite(x).all():
        raise ValueError("The data contains non-finite values.")
    return x


# We return "frozen distributions", but SciPy's rv_frozen type is not a stable public typing target.
FrozenDist = Any

ParamsFromMomentsFn = Callable[[float, float, Mapping[str, Any]], tuple[float, ...]]
FreezeFn = Callable[[tuple[float, ...]], FrozenDist]


@dataclass(frozen=True, slots=True)
class FitSpec:
    params_from_mean_std: ParamsFromMomentsFn
    freeze: FreezeFn


class FitMomentsFactory:
    """
    Generic moment-fit entrypoint.

    - Takes a SciPy distribution object (e.g. scipy.stats.gamma) OR a string key.
    - Uses a registry to select a moments->params implementation.
    - Returns a SciPy frozen distribution (or a frozen-like object) from the spec's freeze().
    """

    def __init__(self, registry: Mapping[str, FitSpec]) -> None:
        self._registry: dict[str, FitSpec] = dict(registry)

    def register(self, key: str, spec: FitSpec) -> None:
        self._registry[key] = spec

    def fit(
        self,
        dist: str | rv_continuous,
        data: Sequence[float] | np.ndarray,
        *,
        ddof: int = 1,
        **kwargs: Any,
    ) -> FrozenDist:
        x = _ensure_finite_1d(data)
        mean = float(x.mean())
        std = float(x.std(ddof=int(ddof)))

        spec = self._registry[self._key(dist)]
        params = spec.params_from_mean_std(mean, std, kwargs)
        return spec.freeze(params)

    @staticmethod
    def _key(dist: str | rv_continuous) -> str:
        if isinstance(dist, str):
            return dist
        return str(dist.name)
    