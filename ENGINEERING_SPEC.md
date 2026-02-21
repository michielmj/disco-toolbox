# ENGINEERING_SPEC.md

## 📘 Project Overview

**Project Name:** `disco-toolbox`  
**Description:**  
Reusable Python utilities for *disco* simulation models.

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

---

### 1.1 `toolbox.orderbook`

#### Purpose

`toolbox.orderbook` provides a fast, low-overhead "orderbook" data structure used by allocation / matching logic in simulation models. An orderbook is an ordered list of `(key, demand_vector)` entries, where:

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
- `length` is restored from the first entry's array length; an empty state yields an empty orderbook.

#### Invariants and error handling

- All entries in an orderbook share the same fixed `length`.
- `append` raises `TypeError` on length mismatch or invalid array shape/dtype.
- `remove` and `get_at_index` raise `IndexError` on invalid indices.
- `allocate_greedy` raises `TypeError` if `values` has the wrong length or dtype.

#### Tests

Unit tests live under `tests/orderbook/` and cover:

- Append / size / length and retrieval
- Greedy allocation semantics, including removal of fulfilled entries
- Pickle roundtrip preserving keys and vector contents

---

### 1.2 `toolbox.calendar`

#### Purpose

`toolbox.calendar` provides calendar-aware time arithmetic for simulation models. A `Calendar` maps wall-clock day indices to work-duration units via a cyclic weight pattern, with optional non-repeating holiday overrides. The primary operation is `elapse(start, duration) -> finish`, which answers: "if a task starts at calendar time `start` and requires `duration` units of working time, at what calendar time does it finish?"

#### Public API

- `Calendar(pattern, holidays=None, horizon=None)`  
  Constructs a calendar from a cyclic weight pattern.
  - `pattern: Sequence[float]` — cyclic daily weights. `1.0` = full working day, `0.0` = non-working day, `0.5` = half day. Must be non-empty; all weights must be non-negative.
  - `holidays: dict[int, float] | None` — optional non-repeating overrides keyed by absolute day index.
  - `horizon: int | None` — initial compiled horizon in days. Defaults to the last holiday day plus a 3-year buffer.

- `Calendar.elapse(start: float | np.ndarray, duration: float | np.ndarray) -> float | np.ndarray`  
  Returns the finish time after consuming `duration` working units from `start`. Accepts scalars or NumPy arrays of any broadcastable shape; returns the same kind.

- `Calendar.add_holiday(day: int, weight: float = 0.0) -> None`  
  Adds or updates a holiday override for a specific calendar day. Triggers an incremental prefix-sum update from `day` onward: O(H − day).

- `Calendar.remove_holiday(day: int) -> None`  
  Removes a holiday override, restoring the pattern weight for that day.

- Properties:
  - `Calendar.cycle_work: float` — total work units delivered per full pattern cycle.
  - `Calendar.horizon: int` — current compiled horizon (exclusive upper bound on day index).
  - `Calendar.holidays: dict[int, float]` — currently overridden days and their weights.

- Exceptions:
  - `CalendarError` — raised for invalid patterns, negative weights, zero-work patterns, or negative day indices.

#### Core implementation

The calendar is **compiled** at construction time into a dense weight array and a cumulative prefix-sum array covering the full horizon. This makes `elapse` a single `np.searchsorted` call plus interpolation — O(log H) per element with no per-day loops.

Key design decisions:

- **Dense weight array** — holidays are baked in at construction; no per-call holiday lookup is needed.
- **Prefix-sum array** — `target_work = prefix[start_day] + weight[start_day] * frac + duration` is computed in one step; `searchsorted` then finds the finish day.
- **Incremental prefix rebuild** — `add_holiday` recomputes only the suffix from the changed day, O(H − day).
- **Horizon auto-extension** — if a task finishes beyond the compiled horizon, the arrays are extended on demand.

#### Cython acceleration (`_core`)

The `elapse` hot path is accelerated by a Cython extension module (`toolbox.calendar._core`) when available, falling back to a pure-NumPy implementation otherwise.

- The Cython core (`elapse_core`) takes pre-validated contiguous `float64` arrays and executes a fused binary-search + interpolation loop with `prange` (OpenMP) for parallel element processing.
- Compiler flags: `-O3` for distributed wheels; `-march=native` enabled for local builds via `DISCO_NATIVE_MARCH=1`.
- OpenMP is required for parallel execution; the extension builds and runs correctly without it (single-threaded).

#### File layout

```
src/toolbox/calendar/
    __init__.py        # re-exports Calendar, CalendarError
    _exceptions.py     # CalendarError definition
    calendar.py        # Calendar implementation
    _core.pyx          # Cython hot path
    _core.pyi          # type stub for the Cython extension
    py.typed           # PEP 561 marker
```

#### Tests

Unit tests live under `tests/calendar/` and cover:

- Scalar elapse for whole days, fractional starts, weekend crossing, and large durations
- Pattern variants: alternating weights, mixed weights, all-zero patterns
- Holiday overrides: add, update, remove, construction-time holidays
- NumPy array inputs: 1-D, 2-D, broadcasting, shape preservation
- Consistency between scalar and array paths
- Horizon auto-extension
- Monotonicity and chaining invariants: `elapse(elapse(s, a), b) == elapse(s, a + b)`

---

### 1.3 `toolbox.capacity`

#### Purpose

`toolbox.capacity` provides a fixed-capacity token pool for scheduling jobs across parallel resources. From a queueing theory perspective, `Capacity` models a multi-server queue (M/G/c) where each server is represented by a token that tracks its next available time. Jobs are assigned **greedily** to the earliest-free token — no sequencing decisions are made. An optional `Calendar` makes finish times calendar-aware, so durations are measured in working time rather than wall-clock time.

#### Public API

- `Capacity(capacity: int, calendar: Calendar | None = None)`  
  Constructs a capacity pool.
  - `capacity` — number of parallel tokens (servers). Must be ≥ 1.
  - `calendar` — optional `Calendar` for working-time finish calculation.

- `Capacity.process(epoch: float, duration: float | np.ndarray) -> tuple[float, float] | tuple[np.ndarray, np.ndarray]`  
  Assigns one or more jobs to the earliest-available token(s).
  - `epoch` — the earliest moment any job may start.
  - `duration` — a single duration (scalar) or an ordered sequence of durations (array-like). Jobs in a vector are assigned **in order of occurrence**; each assignment updates the token pool before the next.
  - Returns `(start, finish)` scalars for a scalar duration, or `(starts, finishes)` arrays for an array duration.
  - When a `Calendar` is attached, `finish = calendar.elapse(start, duration)`; otherwise `finish = start + duration`.

- Properties:
  - `Capacity.capacity: int` — number of tokens in the pool.
  - `Capacity.calendar: Calendar | None` — the attached calendar, if any.

#### Core implementation

Internally, token finish times are maintained in a **min-heap** (`list[float]` with `heapq`). For each job:

1. Pop the smallest finish time from the heap (earliest free token).
2. `start = max(epoch, earliest_free)`
3. `finish = calendar.elapse(start, duration)` or `start + duration`
4. Push `finish` back onto the heap.

The vector path is a sequential Python loop over this operation; the assignments are inherently sequential because each draw mutates the heap before the next.

A C++ / Cython implementation was considered but not adopted: `heapq` is already a C extension, the loop is not parallelisable, and the Python implementation is clear and maintainable.

#### File layout

```
src/toolbox/capacity/
    __init__.py    # re-exports Capacity
    capacity.py    # Capacity implementation
    py.typed       # PEP 561 marker
```

#### Tests

Unit tests live under `tests/capacity/` and cover:

- Construction: invalid capacity, properties, repr
- Scalar process: single and multi-token pools, epoch handling, greedy token selection, zero duration
- Vector process: token assignment order, shape preservation, consistency with sequential scalar calls
- Calendar integration: weekend-crossing finish times, holiday-aware token availability, cross-path consistency
- Invariants: monotonic finish times on a single token, capacity unchanged after processing, all tokens used before reuse

---

### 1.4 `toolbox.distributions`

#### Purpose

`toolbox.distributions` provides small, simulation-oriented distribution utilities built on top of SciPy's statistics API.
The goal is to support a **generic moment-based fitting** workflow while remaining compatible with SciPy usage patterns
(where possible).

Key design constraints:

- Prefer returning *standard SciPy frozen distributions* to keep downstream usage consistent (`rvs`, `cdf`, `ppf`, `mean`, `std`, …).
- Avoid relying on SciPy *internal* distribution implementations (which are not designed for public extension).
- Provide a small number of custom distributions that are not available in SciPy, but still implement the SciPy
  `rv_continuous` interface.

#### Public API

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

#### Moment Fitting Strategy

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

#### `rectnorm` Distribution

`rectnorm` models a **rectified normal** random variable:

- sample `Z ~ Normal(mu, sigma)`
- return `X = max(0, Z)`

This distribution has a point mass at `0` plus a continuous density on `(0, ∞)`.

Implementation notes:

- Implemented as `rv_continuous` in `src/toolbox/distributions/rectnorm.py`.
- Supports vectorized `pdf`, `cdf`, and `rvs`.
- Provides `params_from_mean_std(mean, std)` which numerically inverts the mean/std equations using `scipy.optimize.minimize`.

`fit_moments` includes a `"rectnorm"` key that uses `rectnorm.params_from_mean_std(...)` and returns `rectnorm(mu, sigma)`.

#### `conditional` Distributions

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

#### Tests

Tests live under `tests/distributions/` and cover:

- Registry-based moment fitting for supported distributions
- Correctness of `rectnorm.params_from_mean_std` roundtrips
- Sanity checks for `conditional(...).rvs(...)` (e.g., empirical mass at `0` close to `1 - p`)

---

### 1.5 Reserved Subpackage Sections

Placeholders for future subpackages:

- `toolbox.<TBD>`
- `toolbox.<TBD>`
