# 🧰 disco-toolbox

**A toolbox for `disco` simulation models.**

[![PyPI](https://img.shields.io/pypi/v/disco-toolbox.svg)](https://pypi.org/project/disco-toolbox/)
[![License: Apache-2.0](https://img.shields.io/badge/license-Apache%20License%202.0-blue.svg)](LICENSE)
[![Build](https://github.com/michielmj/disco-toolbox/actions/workflows/build.yml/badge.svg)](https://github.com/michielmj/disco-toolbox/actions)
[![Tests](https://github.com/michielmj/disco-toolbox/actions/workflows/test.yml/badge.svg)](https://github.com/michielmj/disco-toolbox/actions)

`disco-toolbox` contains reusable utilities intended to be embedded in **disco** models and supporting code.
The toolbox is deliberately small and focused: each subpackage should solve one clear problem well, with
an emphasis on correctness, testability, and performance.

---

## 🧭 Overview

- **Python ≥ 3.11**
- **Key dependencies:** `python-graphblas`, `numpy`, `scipy` (orderbook uses dense `numpy` arrays; GraphBLAS is used elsewhere in the toolbox)
- Includes optional **C++/pybind11** and **Cython** extensions for hot paths (built with **scikit-build-core**).

---

## 📦 Installation

```bash
pip install disco-toolbox
```

---

## 📚 Subpackages

### `toolbox.orderbook`

A compact order book for simulation loops (fast C++/pybind11 core):

- Store outstanding orders as **dense** `numpy.ndarray` (float64) vectors
- Keys are stored as opaque `bytes` (so you can pack identifiers however you like)
- Allocate available capacity/stock against orders with a greedy strategy
- Remove fully fulfilled orders during allocation
- Supports pickling / unpickling for checkpointing

#### Quick example

```python
import numpy as np
from toolbox.orderbook import Orderbook

ob = Orderbook()
ob.append(b"abc", np.array([1, 2, 3], dtype=np.double))
ob.append(b"def", np.array([4, 5, 6], dtype=np.double))

stock = np.array([5, 5, 5], dtype=np.double)
allocations = ob.allocate_greedy(stock)

for key, alloc in allocations:
    print(key, alloc)

print("remaining:", stock)
```

---

### `toolbox.calendar`

Calendar-aware time arithmetic for simulation models (Cython-accelerated hot path):

- Define working-time patterns via a **cyclic weight array** — full days, non-working days, or partial days
- Add **non-repeating holiday overrides** by absolute day index
- Compute `elapse(start, duration) -> finish`: the calendar time at which `duration` working units have elapsed since `start`
- Accepts and returns both **scalars and NumPy arrays** (any broadcastable shape)
- Compiled to a dense prefix-sum array at construction time; `elapse` is a single binary search — O(log H) per element

#### Quick example

```python
import numpy as np
from toolbox.calendar import Calendar

# Mon–Fri = 1 working day, Sat–Sun = 0
cal = Calendar([1, 1, 1, 1, 1, 0, 0])

# Add public holidays
cal.add_holiday(0)        # Monday 0 is a bank holiday
cal.add_holiday(1, 0.5)   # Tuesday 1 is a half-day

# Scalar
finish = cal.elapse(start=0.0, duration=5.0)
print(finish)   # 9.0  (loses Mon, half Tue)

# NumPy arrays
starts    = np.array([0.0, 7.0, 14.0])
durations = np.array([5.0, 3.0,  1.0])
print(cal.elapse(starts, durations))
```

---

### `toolbox.capacity`

Fixed-capacity token pool for scheduling jobs across parallel resources:

- Models a **multi-server queue** where each token represents a parallel resource (server, machine, worker)
- Jobs are assigned **greedily** to the earliest-free token — no sequencing decisions are made
- Accepts a single duration (scalar) or an **ordered vector** of durations; vector jobs are assigned in occurrence order
- Optional `Calendar` integration: finish times are computed in **working time** rather than wall-clock time

#### Quick example

```python
import numpy as np
from toolbox.capacity import Capacity
from toolbox.calendar import Calendar

# Two parallel resources on a Mon–Fri calendar
cal = Calendar([1, 1, 1, 1, 1, 0, 0])
cap = Capacity(2, calendar=cal)

# Single job
start, finish = cap.process(epoch=0.0, duration=3.0)
print(start, finish)   # 0.0  3.0

# Batch of jobs assigned in order
starts, finishes = cap.process(epoch=0.0, duration=np.array([5.0, 5.0, 2.0, 2.0]))
print(starts)    # [0. 0. 5. 5.]   both tokens occupied for first two jobs
print(finishes)  # [5. 5. 8. 8.]   next two wait for weekend, finish Monday
```

---

### `toolbox.distributions`

Small, simulation-oriented distribution utilities on top of SciPy:

- Generic, registry-backed **moment fitting** via `fit_moments.fit(...)` returning standard SciPy *frozen* distributions
- Custom distributions:
  - `rectnorm`: rectified normal distribution (`X = max(0, Z)` where `Z ~ Normal(mu, sigma)`)
  - `conditional`: zero-inflated wrapper around a base SciPy distribution

#### Quick example

```python
import numpy as np
from numpy.random import default_rng
from scipy import stats
from toolbox.distributions import fit_moments
from toolbox.distributions.conditional import conditional

rng = default_rng(42)

data = stats.gamma(a=3.0, scale=2.0).rvs(size=50_000, random_state=rng)

rv = fit_moments.fit("gamma", data, ddof=1)
print(rv.mean(), rv.std())

# Zero-inflated normal: P(X=0)=1-p, else Normal(inner_loc, inner_scale)
norm_cond = conditional(stats.norm, name="norm_cond")
x = norm_cond(0.3, 0.0, 1.0).rvs(size=10_000, random_state=rng)
print("zero fraction:", np.mean(x == 0.0))
```

---

## 🧰 Development Setup

Clone and install in editable mode:

```bash
git clone https://github.com/michielmj/disco-toolbox.git
cd disco-toolbox
pip install -e ".[dev]"
```

Run tests:

```bash
pytest -q
```

Type checking:

```bash
mypy src
```

---

## 🏗️ Building extensions locally

The toolbox includes two compiled extension modules:

- `toolbox.orderbook._core` — C++ / pybind11
- `toolbox.calendar._core` — Cython (with optional OpenMP parallelism)

Both are built automatically on install via **scikit-build-core**.

Typical local build (editable install):

```bash
pip install -e ".[dev]"
```

To enable native CPU optimisations for the Cython extension (local development only):

```bash
DISCO_NATIVE_MARCH=1 pip install -e ".[dev]"
```

If you are modifying C++ or Cython sources and want a clean rebuild:

```bash
rm -rf build
pip install -e .
```

---

## 🧾 License

Apache 2.0 License © 2026 — part of the **disco-toolbox** project.
