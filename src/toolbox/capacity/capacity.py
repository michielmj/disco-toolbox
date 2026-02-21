import heapq
from typing import Tuple
import numpy as np

from toolbox.calendar import Calendar


class Capacity:

    def __init__(self, capacity: int, calendar: Calendar | None = None) -> None:
        if capacity < 1:
            raise ValueError("Capacity must be at least 1.")
        self._heap: list[float] = [0.0] * capacity
        heapq.heapify(self._heap)
        self._calendar = calendar

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
            float(self._calendar.elapse(start, duration))
            if self._calendar is not None
            else start + duration
        )
        heapq.heappush(self._heap, finish)
        return start, finish

    @property
    def capacity(self) -> int:
        return len(self._heap)

    @property
    def calendar(self) -> Calendar | None:
        return self._calendar

    def __repr__(self) -> str:
        return (
            f"Capacity(capacity={self.capacity}, "
            f"calendar={self._calendar is not None}, "
            f"token_finish_times={sorted(self._heap)})"
        )
