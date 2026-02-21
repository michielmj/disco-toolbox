# src/toolbox/calendar/__init__.py
"""
toolbox.calendar
~~~~~~~~~~~~~~~~

Calendar-aware time arithmetic.  A Calendar maps "wall clock" day indices to
"work duration" units via a cyclic weight pattern, with optional non-repeating
holiday overrides.

Basic usage::

    from toolbox.calendar import Calendar

    cal = Calendar([1, 1, 1, 1, 1, 0, 0])          # Mon–Fri
    cal.add_holiday(0)                               # bank holiday on day 0
    finish = cal.elapse(start=0.0, duration=5.0)    # → 8.0

NumPy arrays are accepted everywhere a scalar is::

    import numpy as np
    starts    = np.array([0.0, 7.0, 14.0])
    durations = np.array([5.0, 3.0,  1.0])
    finishes  = cal.elapse(starts, durations)

Public API
----------
Calendar       The main class.
CalendarError  Base exception for all calendar-related errors.
"""

from __future__ import annotations

from toolbox.calendar._exceptions import CalendarError
from toolbox.calendar.calendar import Calendar

__all__ = [
    "Calendar",
    "CalendarError",
]
