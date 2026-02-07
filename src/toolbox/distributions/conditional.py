from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import rv_continuous


def _split_inner(
    args: tuple[Any, ...],
    n_base_shapes: int,
) -> tuple[float, tuple[Any, ...], float, float]:
    """
    Args layout:
        p,
        <base shapes...> (n_base_shapes),
        inner_loc,
        inner_scale
    """
    if len(args) != 1 + n_base_shapes + 2:
        raise ValueError(
            f"conditional distribution expected {1 + n_base_shapes + 2} args "
            f"(p, {n_base_shapes} base-shapes, inner_loc, inner_scale) but got {len(args)}"
        )

    p = float(args[0])
    base_shapes = tuple(args[1 : 1 + n_base_shapes])
    inner_loc = float(args[1 + n_base_shapes])
    inner_scale = float(args[2 + n_base_shapes])
    return p, base_shapes, inner_loc, inner_scale


class conditional_gen(rv_continuous):
    """
    Zero-inflated wrapper around a base continuous distribution.

    With probability (1-p): return 0
    With probability p: draw from base(*base_shapes, loc=inner_loc, scale=inner_scale)

    Parameters (positional):
        p, <base shapes...>, inner_loc, inner_scale
    """

    def __init__(self, base: rv_continuous, **kwds: Any) -> None:
        self._base: rv_continuous = base
        super().__init__(**kwds)

        # Cache base shape count for unpacking.
        base_shapes = getattr(self._base, "shapes", None)
        if base_shapes:
            self._n_base_shapes = len([s.strip() for s in base_shapes.split(",") if s.strip()])
        else:
            self._n_base_shapes = 0

    def _updated_ctor_param(self) -> dict[str, Any]:
        # SciPy's typing for _updated_ctor_param is loose (often Any), so normalize.
        params_any = super()._updated_ctor_param()
        params: dict[str, Any] = dict(params_any)
        params["base"] = self._base
        return params

    def _argcheck(self, *args: Any) -> bool:
        if len(args) != 1 + self._n_base_shapes + 2:
            return False
        p = np.asarray(args[0])
        if np.any(p < 0.0) or np.any(p > 1.0):
            return False
        inner_scale = np.asarray(args[-1])
        if np.any(inner_scale <= 0.0):
            return False
        return True

    def _rvs(
        self,
        *args: Any,
        size: int | tuple[int, ...] | None = None,
        random_state: np.random.Generator | np.random.RandomState | None = None,
    ) -> np.ndarray:
        p, base_shapes, inner_loc, inner_scale = _split_inner(args, self._n_base_shapes)

        # SciPy normally passes a RNG here, but keep this robust.
        if random_state is None:
            random_state = np.random.default_rng()

        # Normalize size for np.zeros; SciPy may pass None for scalar draws.
        size_norm: int | tuple[int, ...]
        if size is None:
            size_norm = 1
            scalar = True
        else:
            size_norm = size
            scalar = False

        y = np.zeros(size_norm, dtype=np.float64)

        # random_state can be RandomState or Generator
        if isinstance(random_state, np.random.RandomState):
            draw = random_state.random_sample(size_norm) < p
        else:
            draw = random_state.random(size_norm) < p

        n_draw = int(np.sum(draw))
        if n_draw > 0:
            y[draw] = self._base.rvs(
                *base_shapes,
                loc=inner_loc,
                scale=inner_scale,
                size=n_draw,
                random_state=random_state,
            )

        if scalar:
            # Return an array of shape (1,) – SciPy will squeeze upstream if needed.
            return y.reshape((1,))
        return y

    def _pdf(self, x: Any, *args: Any) -> np.ndarray:
        p, base_shapes, inner_loc, inner_scale = _split_inner(args, self._n_base_shapes)

        xx = np.asarray(x, dtype=np.float64)
        y = np.zeros_like(xx, dtype=np.float64)

        # Convention: treat atom at 0 as "pdf" value at x==0
        y[xx == 0.0] = 1.0 - p

        m = xx != 0.0
        if np.any(m):
            y[m] = p * self._base.pdf(
                xx[m],
                *base_shapes,
                loc=inner_loc,
                scale=inner_scale,
            )
        return y

    def _cdf(self, x: Any, *args: Any) -> np.ndarray:
        p, base_shapes, inner_loc, inner_scale = _split_inner(args, self._n_base_shapes)

        xx = np.asarray(x, dtype=np.float64)
        y = np.zeros_like(xx, dtype=np.float64)

        mneg = xx < 0.0
        if np.any(mneg):
            y[mneg] = p * self._base.cdf(
                xx[mneg],
                *base_shapes,
                loc=inner_loc,
                scale=inner_scale,
            )

        mpos = ~mneg
        if np.any(mpos):
            y[mpos] = (1.0 - p) + p * self._base.cdf(
                xx[mpos],
                *base_shapes,
                loc=inner_loc,
                scale=inner_scale,
            )

        return y

    def _munp(self, n: int, *args: Any) -> float:
        p, base_shapes, inner_loc, inner_scale = _split_inner(args, self._n_base_shapes)
        return float(
            p
            * self._base.moment(
                n,
                *base_shapes,
                loc=inner_loc,
                scale=inner_scale,
            )
        )

    def fit(self, data: Any, **kwargs: Any) -> tuple[float, ...]:
        x = np.asarray(data, dtype=np.float64).reshape(-1)
        if not np.isfinite(x).all():
            raise ValueError("The data contains non-finite values.")

        p = float((x != 0.0).mean())
        inner = self._base.fit(x[x != 0.0], **kwargs)
        return (p, *inner)


def conditional(base: rv_continuous, *, name: str) -> conditional_gen:
    base_shapes = getattr(base, "shapes", None)
    if base_shapes:
        base_shapes = ",".join([s.strip() for s in base_shapes.split(",") if s.strip()])
        shapes = f"p,{base_shapes},inner_loc,inner_scale"
    else:
        shapes = "p,inner_loc,inner_scale"

    return conditional_gen(base=base, name=name, shapes=shapes)
