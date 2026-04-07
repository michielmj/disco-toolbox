import heapq
from typing import Tuple, Callable
import numpy as np

from toolbox.calendar import Calendar


class Capacity:

    def __init__(self,
                 capacity: int,
                 calc_finish: Callable[[float, float], float] | None = None
                 ) -> None:
        if capacity < 1:
            raise ValueError("Capacity must be at least 1.")
        self._heap: list[float] = [0.0] * capacity
        heapq.heapify(self._heap)
        self._calc_finish: Callable[[float, float], float] | None = calc_finish

    def process(
        self,
        epoch: float,
        duration: float | np.ndarray,
    ) -> Tuple[float, float] | Tuple[np.ndarray, np.ndarray]:
        if np.ndim(duration) == 0:
            return self._process_one(epoch, float(duration))

        durations = np.asarray(duration, dtype=float)
        n = len(durations)
        starts   = np.empty(n, dtype=float)
        finishes = np.empty(n, dtype=float)

        for i in range(n):
            starts[i], finishes[i] = self._process_one(epoch, durations[i])

        return starts, finishes

    def _process_one(self, epoch: float, duration: float) -> Tuple[float, float]:
        earliest_free: float = heapq.heappop(self._heap)
        start: float = max(epoch, earliest_free)
        finish: float = (
            float(self._calc_finish(start, duration))
            if self._calc_finish is not None
            else start + duration
        )
        heapq.heappush(self._heap, finish)
        return start, finish

    @property
    def capacity(self) -> int:
        return len(self._heap)

    def __repr__(self) -> str:
        return (
            f"Capacity(capacity={self.capacity}, "
            f"finish_callback={self._calc_finish is not None}, "
            f"token_finish_times={sorted(self._heap)})"
        )
