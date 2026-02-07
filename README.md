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
- Includes optional **C++/pybind11** extensions for hot paths (built with **scikit-build-core**).

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

## 🏗️ Building the extension locally

The orderbook implementation includes a C++ extension module (`toolbox.orderbook._core`) built with pybind11.

Typical local build (editable install):

```bash
pip install -e ".[dev]"
```

If you are modifying C++ sources and want a clean rebuild, remove the build directory and reinstall:

```bash
rm -rf build
pip install -e .
```

---

## 🧾 License

Apache 2.0 License © 2026 — part of the **disco-toolbox** project.
