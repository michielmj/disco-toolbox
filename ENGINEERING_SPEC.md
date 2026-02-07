# ENGINEERING_SPEC.md

## 📘 Project Overview

**Project Name:** `disco-toolbox`  
**Description:**  
Reusable Python utilities for  *disco* simulation models.  

**Owner:** Michiel Jansen  
**Repository:** https://github.com/michielmj/disco-toolbox  
**License:** Apache-2.0  
**Programming Language:** Python ≥ 3.11  
**Core Dependency Stack:** `numpy`, `python-graphblas`

---

## 🧭 Goals and Scope

### Primary Objectives


### Non-Goals
  

---

## ⚙️ Architecture Overview

## 1. Subpackages

This document describes the major subpackages of `disco-toolbox`. Each subpackage section is intended to be extended over time
with deeper implementation details, performance notes, and integration patterns.

### 1.1 `toolbox.orderbook`

#### 1.1.1 Purpose

`toolbox.orderbook` provides a small, high-performance order book that can be used by *disco* simulation models to:

- register incoming orders (each order is a sparse demand vector),
- allocate available capacity/stock against outstanding orders using a greedy strategy,
- remove orders that are fully fulfilled.

The primary use-case is frequent, sparse allocation in tight simulation loops.

#### 1.1.2 Public API

The public API is intentionally small:

- `Orderbook.append(key: dict, arr: graphblas.Vector) -> None`
- `Orderbook.remove(index: int) -> None`
- `Orderbook.get_at_index(index: int) -> tuple[dict, graphblas.Vector]`
- `Orderbook.allocate_greedy(values: graphblas.Vector) -> list[tuple[dict, graphblas.Vector]]`
- Properties:
  - `Orderbook.size: int` number of orders in the book
  - `Orderbook.length: int` vector length (dimension) for all orders; `0` when empty

**Key semantics.** `key` is a Python `dict` that is stored *by reference*. This is convenient for attaching rich metadata to an order
without copying. Mutating the dict after `append` will be observable through later `get_at_index` and `allocate_greedy` results.

**Vector semantics.** `arr` is a `python-graphblas` `Vector` representing the order quantities. The `Vector` passed to `append` is not
mutated.

**Length consistency.** The first appended order fixes `length`. All later `append` calls must use the same vector dimension.
`allocate_greedy` must be called with a `values` vector of the same dimension.

#### 1.1.3 Implementation Overview

`Orderbook` is implemented as a CPython extension module using **C++17 + pybind11**:

- Python entry point: `toolbox.orderbook.Orderbook`
- Extension module: `toolbox.orderbook._core` (built from `src/toolbox/orderbook/_core.cpp`)

Internal storage uses a linked list of entries. Each entry contains:

- `key`: `py::dict` (reference-held)
- `order`: sparse COO representation stored in C++:
  - indices: `vector<uint64_t>`
  - values: `vector<double>` (FP64)

On `append`, the input `Vector` is converted once to COO via `Vector.to_coo()` and stored in C++.
Non-positive values are dropped and indices are stored in sorted order.

On `get_at_index`, a new `Vector` is reconstructed from the stored COO using `Vector.from_coo(...)` (FP64, fixed `size`).

#### 1.1.4 Greedy Allocation Algorithm

`allocate_greedy(values)` performs a single pass over the current order list.

For each order, it allocates on the *intersection* of (order indices) and (values indices):

- For each order entry `(i, order_i)`:
  - allocate `alloc_i = min(order_i, values_i)` when `values_i > 0`
  - update `order_i -= alloc_i`
  - update `values_i -= alloc_i`

An allocation `Vector` is produced per order where at least one `alloc_i > 0`. The returned list preserves the original order book
traversal order.

Orders that become empty (no remaining positive entries) are removed from the book during the traversal.

**Mutation of `values`.** For API parity with the earlier NumPy-based implementation, `values` is mutated in place. The implementation
reconstructs an updated values vector from the internal sparse map and assigns it back to the original object using the canonical
python-graphblas idiom `values << new_values`.

#### 1.1.5 Serialization

`Orderbook` supports pickling via `__getstate__` / `__setstate__`.

The state is encoded as:

- `length` (vector dimension)
- list of entries as `(key_dict, idx_array_u64, val_array_f64)`

On restore, entries are rehydrated directly into internal COO storage (without needing to reconstruct then re-extract a `Vector`).

#### 1.1.6 Performance Notes

- The design targets sparse workloads where `allocate_greedy` is called frequently.
- Converting orders to COO once at `append` amortizes extraction costs.
- The allocation hot-path is implemented as a tight C++ loop over sparse entries to avoid Python dispatch overhead.
- Current implementation uses an `unordered_map` for `values` during allocation. If profiling shows this to be a bottleneck, consider
  replacing it with a sorted-array merge approach for improved cache locality.

#### 1.1.7 Future Extensions

This section is intentionally left open for future additions, including but not limited to:

- alternative allocation strategies (FIFO with priorities, proportional allocation, fair-share),
- supporting different numeric dtypes (e.g., INT64) or per-order dtype,
- more efficient storage containers (vector + tombstones, pooling),
- batch APIs (append_many, allocate_many),
- richer querying/iteration utilities (without exposing internal structure).

### 1.2 `toolbox.distributions`

#### 1.2.1 Purpose

`toolbox.distributions` provides small, simulation-oriented distribution utilities built on top of SciPy’s statistics API.
The goal is to support a **generic moment-based fitting** workflow while remaining compatible with SciPy usage patterns
(where possible).

Key design constraints:

- Prefer returning *standard SciPy frozen distributions* to keep downstream usage consistent (`rvs`, `cdf`, `ppf`, `mean`, `std`, …).
- Avoid relying on SciPy *internal* distribution implementations (which are not designed for public extension).
- Provide a small number of custom distributions that are not available in SciPy, but still implement the SciPy
  `rv_continuous` interface.

#### 1.2.2 Public API

Primary entry points:

- `toolbox.distributions.fit_moments`: a registry-backed factory for fitting distributions from sample moments.
  - `fit_moments.fit(dist: str | scipy.stats.rv_continuous, data: ArrayLike, *, ddof: int = 1, **kwargs) -> Any`
    returns a *frozen* SciPy distribution.
- Custom distributions and helpers:
  - `toolbox.distributions.rectnorm.rectnorm`: rectified normal distribution (`rv_continuous` instance).
  - `toolbox.distributions.conditional.conditional(base: rv_continuous, *, name: str) -> rv_continuous`:
    creates a *zero-inflated / conditional* distribution wrapper around a SciPy base distribution.

Intended usage is that simulation models can be written against a generic moment fitting API:

```python
rv = fit_moments.fit("gamma", data, ddof=1)
x = rv.rvs(size=1000, random_state=rng)
```

#### 1.2.3 Moment Fitting Strategy

`fit_moments` implements a registry of supported distributions. For each distribution key, the registry provides:

- a mapping from `(mean, std, **kwargs)` to distribution parameters (shape/loc/scale as appropriate), and
- a freeze function that returns a SciPy frozen distribution.

The moment fitting flow is:

1. Compute `(mean, std)` from data (honoring `ddof`).
2. Convert moments to parameters (closed-form where possible).
3. Return a frozen distribution object.

Closed-form examples:

- Normal: `norm(loc=mean, scale=std)`
- Uniform: `uniform(loc=mean - sqrt(3)*std, scale=2*sqrt(3)*std)`
- Gamma: `gamma(a=(mean/std)**2, loc=0, scale=mean/a)`
- Laplace: `laplace(loc=mean, scale=std/sqrt(2))`

#### 1.2.4 `rectnorm` Distribution

`rectnorm` models a **rectified normal** random variable:

- sample `Z ~ Normal(mu, sigma)`
- return `X = max(0, Z)`

This distribution has a point mass at `0` plus a continuous density on `(0, ∞)`.

Implementation notes:

- Implemented as `rv_continuous` in `src/toolbox/distributions/rectnorm.py`.
- Supports vectorized `pdf`, `cdf`, and `rvs`.
- Provides `params_from_mean_std(mean, std)` which numerically inverts the mean/std equations using `scipy.optimize.minimize`.

`fit_moments` includes a `"rectnorm"` key that uses `rectnorm.params_from_mean_std(...)` and returns `rectnorm(mu, sigma)`.

#### 1.2.5 `conditional` Distributions

The conditional distribution is a **zero-inflated wrapper** around a base SciPy continuous distribution:

- With probability `1 - p`, return `0`
- With probability `p`, draw from `base(*shape_params, loc=inner_loc, scale=inner_scale)`

The wrapper is implemented as an `rv_continuous` in `src/toolbox/distributions/conditional.py` and uses explicit `shapes=...` so that
SciPy argument parsing and freezing work correctly.

Parameter convention (positional arguments):

- `p, <base shapes...>, inner_loc, inner_scale`

Examples:

- `conditional(norm, name="norm_cond")(p, inner_loc, inner_scale)`
- `conditional(gamma, name="gamma_cond")(p, a, inner_loc, inner_scale)`

Notes:

- The distribution is mixed (an atom at `0`), so `pdf(0)` is not a true density. The implementation follows the toolbox convention
  of returning the point mass at `0` when `x == 0`.
- For simulation use-cases, `rvs` is typically the primary operation.

#### 1.2.6 Tests

Tests live under `tests/distributions/` and cover:

- registry-based moment fitting for supported distributions,
- correctness of `rectnorm.params_from_mean_std` roundtrips,
- sanity checks for `conditional(...).rvs(...)` (e.g., empirical mass at `0` close to `1 - p`).


### 1.3 Reserved Subpackage Sections
Placeholders for future subpackages:

- `toolbox.<TBD>`
- `toolbox.<TBD>`
