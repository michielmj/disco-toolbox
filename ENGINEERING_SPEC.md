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

#### Purpose

`toolbox.orderbook` provides a fast, low-overhead “orderbook” data structure used by allocation / matching logic in simulation models. An orderbook is an ordered list of `(key, demand_vector)` entries, where:

- `key` is an opaque identifier (Python `bytes`)
- `demand_vector` is a **dense** 1‑D `numpy.ndarray` of `float64` with fixed length (`length`)

The core performance goal is to support tight loops for greedy allocation while avoiding Python-level overhead.

#### Public API (Python)

- `Orderbook.append(key: bytes, arr: np.ndarray[np.float64]) -> None`  
  Appends an entry. `arr` must be 1‑D, `dtype=float64`, and its length must match the orderbook `length` (the first appended vector defines `length`).

- `Orderbook.remove(index: int) -> None`  
  Removes entry at `index` (0‑based).

- `Orderbook.get_at_index(index: int) -> tuple[bytes, np.ndarray[np.float64]]`  
  Returns a **copy** of the stored vector.

- `Orderbook.allocate_greedy(values: np.ndarray[np.float64]) -> list[tuple[bytes, np.ndarray[np.float64]]]`  
  Greedily allocates from a mutable stock vector `values` against entries in order.
  - Mutates `values` in-place by subtracting allocations.
  - Mutates the orderbook by subtracting allocations from each entry and removing entries that become fully fulfilled (sum remaining ≤ 0).
  - Returns a list of `(key, allocation_vector)` for entries that received a non‑zero allocation.

- Properties:
  - `Orderbook.size: int` – number of entries
  - `Orderbook.length: int` – vector length (0 if empty)

#### Core implementation (`_core`)

The implementation is in a CPython extension module (`toolbox.orderbook._core`) written in C++ with **pybind11**, closely mirroring the original Cython design:

- Each entry stores:
  - a heap-allocated C string copy of the Python `bytes` key (`char*`, NUL-terminated)
  - a heap-allocated contiguous `double*` array of length `length`
  - a `next` pointer (singly-linked list)

This representation minimizes per-entry overhead and makes `allocate_greedy` a straightforward pointer-walk over contiguous arrays.

**Important:** The C++ core does **not** depend on GraphBLAS and does not call into `python-graphblas`. All numerical storage is dense NumPy `float64`.

#### Pickling / state

`Orderbook` supports pickling via `__getstate__` / `__setstate__`:

- `__getstate__` returns a Python list of `(key: bytes, data: np.ndarray[np.float64])`.
- `__setstate__` clears the existing structure and replays `append` for each stored entry.
- `length` is restored from the first entry’s array length; an empty state yields an empty orderbook.

#### Invariants and error handling

- All entries in an orderbook share the same fixed `length`.
- `append` raises `TypeError` on length mismatch or invalid array shape/dtype.
- `remove` and `get_at_index` raise `IndexError` on invalid indices.
- `allocate_greedy` raises `TypeError` if `values` has the wrong length or dtype.

#### Tests

Unit tests cover:

- Append / size / length and retrieval
- Greedy allocation semantics, including removal of fulfilled entries
- Pickle roundtrip preserving keys and vector contents


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
