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
- **Key dependencies:** `python-graphblas`, `numpy`, `scipy`
- Includes optional **C++/pybind11** extensions for hot paths (built with **scikit-build-core**).

---

## 📦 Installation

```bash
pip install disco-toolbox
```

---

## 📚 Subpackages

### `toolbox.orderbook`

A compact order book for simulation loops:

- Store outstanding orders as sparse `python-graphblas` vectors (orders are converted to sparse COO once on `append`)
- Allocate available capacity/stock against orders with a greedy strategy
- Remove fully fulfilled orders during allocation

#### Quick example

```python
from graphblas import Vector, dtypes
from toolbox.orderbook import Orderbook

n = 10
ob = Orderbook()

# Orders (metadata dicts are stored by reference)
ob.append({"order_id": "A"}, Vector.from_coo([2, 7], [5.0, 3.0], size=n, dtype=dtypes.FP64))
ob.append({"order_id": "B"}, Vector.from_coo([2, 8], [4.0, 6.0], size=n, dtype=dtypes.FP64))

values = Vector.from_coo([2, 7, 8], [6.0, 1.0, 10.0], size=n, dtype=dtypes.FP64)

allocations = ob.allocate_greedy(values)

# allocations: list[(key_dict, allocation_vector)]
for key, alloc in allocations:
    print(key["order_id"], alloc.to_coo())

# values is mutated in-place (remaining availability)
print("remaining:", values.to_coo())
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

The orderbook implementation includes a C++ extension module (`toolbox.orderbook._core`).

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
