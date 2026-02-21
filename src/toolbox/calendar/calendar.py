import numpy as np
from typing import Sequence, Union, Optional
from ._exceptions import CalendarError   # ← add at the top


try:
    from ._core import elapse_core as _elapse_core
    _HAVE_CYTHON = True
except ImportError:
    _HAVE_CYTHON = False

ArrayLike = Union[float, "np.ndarray"]


class Calendar:
    """
    Compiled calendar: dense weight + prefix-sum array.
    elapse() is accelerated by a Cython/OpenMP extension when available,
    falling back to a pure-NumPy implementation otherwise.
    """

    _DEFAULT_BUFFER: int = 365 * 3

    def __init__(
        self,
        pattern: Sequence[float],
        holidays: Optional[dict[int, float]] = None,
        horizon: Optional[int] = None,
    ) -> None:
        if not pattern:
            raise CalendarError("Pattern must not be empty.")

        self._pattern: list[float] = list(pattern)
        self._n: int = len(self._pattern)
        self._np_pattern: np.ndarray = np.array(self._pattern, dtype=float)

        pp = np.zeros(self._n + 1, dtype=float)
        for i, w in enumerate(self._pattern):
            if w < 0.0:
                raise CalendarError(f"Pattern weights must be non-negative; got {w}.")
            pp[i + 1] = pp[i] + w
        self._pattern_prefix: np.ndarray = pp
        self._cycle_work: float = float(pp[self._n])

        max_hol = max(holidays.keys()) if holidays else -1
        if horizon is None:
            horizon = max(max_hol + 1, 0) + self._DEFAULT_BUFFER
        self._horizon: int = max(horizon, max_hol + 1)

        self._weights: np.ndarray = self._np_pattern[
            np.arange(self._horizon, dtype=np.int64) % self._n
        ].copy()

        if holidays:
            for day, w in holidays.items():
                if day < 0:
                    raise CalendarError(f"Holiday day must be >= 0; got {day}.")
                self._weights[day] = float(w)

        self._build_prefix()

    # ── prefix management ────────────────────────────────────────────────

    def _build_prefix(self) -> None:
        self._prefix = np.empty(self._horizon + 1, dtype=float)
        self._prefix[0] = 0.0
        np.cumsum(self._weights, out=self._prefix[1:])

    def _rebuild_prefix_from(self, day: int) -> None:
        self._prefix[day + 1:] = (
            self._prefix[day] + np.cumsum(self._weights[day:])
        )

    def _extend_to(self, new_horizon: int) -> None:
        old = self._horizon
        new_days = np.arange(old, new_horizon, dtype=np.int64)
        self._weights = np.concatenate(
            [self._weights, self._np_pattern[new_days % self._n]]
        )
        self._horizon = new_horizon
        self._build_prefix()

    # ── holiday management ───────────────────────────────────────────────

    def add_holiday(self, day: int, weight: float = 0.0) -> None:
        if day >= self._horizon:
            self._extend_to(day + 1 + self._DEFAULT_BUFFER)
        self._weights[day] = float(weight)
        self._rebuild_prefix_from(day)

    def remove_holiday(self, day: int) -> None:
        if day < self._horizon:
            self._weights[day] = float(self._np_pattern[day % self._n])
            self._rebuild_prefix_from(day)

    # ── public elapse ────────────────────────────────────────────────────

    def elapse(self, start: ArrayLike, duration: ArrayLike) -> ArrayLike:
        scalar = np.ndim(start) == 0 and np.ndim(duration) == 0
        s = np.atleast_1d(np.asarray(start,    dtype=np.float64))
        d = np.atleast_1d(np.asarray(duration, dtype=np.float64))
        s, d = np.broadcast_arrays(s, d)

        # Ensure C-contiguous flat arrays for the Cython core.
        shape = s.shape
        s = np.ascontiguousarray(s.ravel())
        d = np.ascontiguousarray(d.ravel())

        self._ensure_horizon(s, d)
        result = self._elapse_array(s, d).reshape(shape)
        return float(result.flat[0]) if scalar else result

    # ── horizon guard (shared by both backends) ───────────────────────────

    def _ensure_horizon(
        self, s: np.ndarray, d: np.ndarray
    ) -> None:
        max_start = int(np.floor(s.max())) if s.size else 0
        if max_start >= self._horizon:
            self._extend_to(max_start + 1 + self._DEFAULT_BUFFER)

        max_w   = float(self._np_pattern.max()) or 1.0
        slack   = float(d.max()) if d.size else 0.0
        needed  = self._prefix[self._horizon] + slack
        if needed > self._prefix[self._horizon]:
            extra = int(np.ceil(slack / max_w)) + self._n + self._DEFAULT_BUFFER
            self._extend_to(self._horizon + extra)

    # ── backends ─────────────────────────────────────────────────────────

    def _elapse_array(self, s: np.ndarray, d: np.ndarray) -> np.ndarray:
        if self._cycle_work == 0.0:
            raise CalendarError(
                "All pattern weights are zero; duration can never be consumed."
            )
        if _HAVE_CYTHON:
            return _elapse_core(s, d, self._weights, self._prefix)
        return self._elapse_numpy(s, d)

    def _elapse_numpy(self, s: np.ndarray, d: np.ndarray) -> np.ndarray:
        """Pure-NumPy fallback (single-threaded)."""
        result = np.where(d <= 0.0, s, 0.0)
        mask   = d > 0.0
        if not mask.any():
            return result
        sm, dm  = s[mask], d[mask]
        day0    = np.floor(sm).astype(np.int64)
        frac    = sm - day0.astype(float)
        base    = self._prefix[day0] + self._weights[day0] * frac
        target  = base + dm
        fi      = np.searchsorted(self._prefix, target, side="left")
        fi      = np.clip(fi, 1, self._horizon)
        fd      = fi - 1
        result[mask] = fd.astype(float) + (target - self._prefix[fd]) / self._weights[fd]
        return result

    # ── properties / repr ────────────────────────────────────────────────

    @property
    def cycle_work(self) -> float:
        return self._cycle_work

    @property
    def horizon(self) -> int:
        return self._horizon

    @property
    def holidays(self) -> dict[int, float]:
        pattern_w = self._np_pattern[
            np.arange(self._horizon, dtype=np.int64) % self._n
        ]
        diff_days = np.where(self._weights != pattern_w)[0]
        return {int(d): float(self._weights[d]) for d in diff_days}

    def __repr__(self) -> str:
        backend = "cython+openmp" if _HAVE_CYTHON else "numpy"
        return (
            f"Calendar(pattern={self._pattern}, "
            f"cycle_work={self._cycle_work}, "
            f"horizon={self._horizon}, "
            f"holidays={len(self.holidays)}, "
            f"backend={backend!r})"
        )
