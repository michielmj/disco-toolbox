from __future__ import annotations

from typing import List, Tuple
import numpy as np
import numpy.typing as npt

class Orderbook:
    def __init__(self) -> None: ...

    @property
    def size(self) -> int: ...

    @property
    def length(self) -> int: ...

    def append(self, key: bytes, arr: npt.ArrayLike) -> None: ...
    def remove(self, index: int) -> None: ...
    def get_at_index(self, index: int) -> tuple[bytes, npt.NDArray[np.float64]]: ...

    # Mutates `values` in-place; must be writable
    def allocate_greedy(
        self,
        values: npt.NDArray[np.float64],
    ) -> List[Tuple[bytes, npt.NDArray[np.float64]]]: ...
