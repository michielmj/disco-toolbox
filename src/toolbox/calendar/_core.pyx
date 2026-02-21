# cython: language_level=3
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

"""
Low-level compiled core for Calendar.elapse().

The only public symbol is elapse_core(), which takes pre-validated
contiguous double arrays and returns a newly allocated result array.
The binary search and interpolation are fused into a single prange
loop — each element is fully independent, so the parallelism is
embarrassingly perfect.
"""

import numpy as np
cimport numpy as cnp
from cython.parallel import prange
from libc.math cimport floor

cnp.import_array()


def elapse_core(
    cnp.ndarray[cnp.float64_t, ndim=1] start,
    cnp.ndarray[cnp.float64_t, ndim=1] duration,
    cnp.ndarray[cnp.float64_t, ndim=1] weights,
    cnp.ndarray[cnp.float64_t, ndim=1] prefix,
):
    """
    Vectorised, parallel elapse over flat contiguous arrays.

    Parameters
    ----------
    start, duration : float64[N]   pre-broadcast, contiguous input arrays
    weights         : float64[H]   compiled daily weights
    prefix          : float64[H+1] cumulative sum of weights (prefix[0]=0)

    Returns
    -------
    result : float64[N]
    """
    cdef:
        Py_ssize_t N = start.shape[0]
        Py_ssize_t H = weights.shape[0]

        # Typed memoryviews — raw pointer access, no Python overhead
        double[::1] s   = start
        double[::1] d   = duration
        double[::1] w   = weights
        double[::1] p   = prefix
        double[::1] out

        Py_ssize_t i, lo, hi, mid, fd
        double     s_i, d_i, day0_f, frac, base, target, fw

    result = np.empty(N, dtype=np.float64)
    out = result

    # Each iteration is fully independent → prange with no GIL.
    for i in prange(N, nogil=True, schedule="static"):
        s_i = s[i]
        d_i = d[i]

        if d_i <= 0.0:
            out[i] = s_i
            continue

        # ── base work up to `start` ──────────────────────────────────────
        day0_f = floor(s_i)
        frac   = s_i - day0_f
        lo     = <Py_ssize_t>day0_f          # integer day index

        base   = p[lo] + w[lo] * frac
        target = base + d_i

        # ── binary search in prefix[] for target ────────────────────────
        # Find smallest idx such that prefix[idx] >= target.
        # Search range: [lo+1, H] (finish cannot be before start day).
        lo = lo + 1
        hi = H           # prefix has H+1 elements (indices 0..H)

        while lo < hi:
            mid = (lo + hi) >> 1
            if p[mid] < target:
                lo = mid + 1
            else:
                hi = mid

        # lo == finish_idx;  finish_day = finish_idx - 1
        fd = lo - 1
        fw = w[fd]

        # Interpolate within the finish day.
        out[i] = <double>fd + (target - p[fd]) / fw

    return result
