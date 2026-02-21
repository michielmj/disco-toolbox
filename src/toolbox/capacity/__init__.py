# src/toolbox/capacity/__init__.py
"""
toolbox.capacity
~~~~~~~~~~~~~~~~

Fixed-capacity token pool for scheduling jobs across parallel resources.

Each token represents a parallel resource (server, machine, worker).  Jobs
are assigned greedily to the earliest-available token.  An optional Calendar
makes finish times calendar-aware, so durations are measured in working time.

Basic usage::

    from toolbox.capacity import Capacity

    cap = Capacity(4)
    start, finish = cap.process(epoch=0.0, duration=3.0)

With a calendar::

    from toolbox.capacity import Capacity
    from toolbox.calendar import Calendar

    cal = Calendar([1, 1, 1, 1, 1, 0, 0])   # Mon–Fri
    cap = Capacity(4, calendar=cal)
    start, finish = cap.process(epoch=0.0, duration=3.0)

Batch processing::

    import numpy as np
    starts, finishes = cap.process(epoch=0.0, duration=np.array([3.0, 1.0, 2.0]))
"""

from toolbox.capacity.capacity import Capacity

__all__ = ["Capacity"]
