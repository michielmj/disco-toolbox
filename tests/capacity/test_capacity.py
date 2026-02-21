"""
tests/capacity/test_capacity.py

Covers:
  - Basic scalar process (no calendar)
  - Vector process (no calendar)
  - Token assignment order and greedy behaviour
  - Calendar integration
  - Edge cases
  - Invariants
"""

import numpy as np
import pytest

from toolbox.capacity import Capacity
from toolbox.calendar import Calendar


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def cap1():
    """Single token."""
    return Capacity(1)

@pytest.fixture
def cap2():
    """Two tokens."""
    return Capacity(2)

@pytest.fixture
def cap4():
    """Four tokens."""
    return Capacity(4)

@pytest.fixture
def work_week():
    return Calendar([1, 1, 1, 1, 1, 0, 0])

@pytest.fixture
def cap2_cal(work_week):
    return Capacity(2, calendar=work_week)


# ── Construction ──────────────────────────────────────────────────────────────

class TestConstruction:

    def test_zero_capacity_raises(self):
        with pytest.raises(ValueError):
            Capacity(0)

    def test_negative_capacity_raises(self):
        with pytest.raises(ValueError):
            Capacity(-1)

    def test_capacity_property(self):
        assert Capacity(4).capacity == 4

    def test_no_calendar_by_default(self):
        assert Capacity(2).calendar is None

    def test_calendar_stored(self, work_week):
        cap = Capacity(2, calendar=work_week)
        assert cap.calendar is work_week

    def test_repr_no_calendar(self, cap2):
        r = repr(cap2)
        assert "Capacity(capacity=2" in r
        assert "calendar=False" in r

    def test_repr_with_calendar(self, cap2_cal):
        assert "calendar=True" in repr(cap2_cal)


# ── Scalar process, no calendar ───────────────────────────────────────────────

class TestScalarNoCalendar:

    def test_single_token_sequential(self, cap1):
        s, f = cap1.process(0.0, 3.0)
        assert s == 0.0
        assert f == 3.0
        # Next job must wait for token to free
        s2, f2 = cap1.process(0.0, 2.0)
        assert s2 == 3.0
        assert f2 == 5.0

    def test_two_tokens_parallel(self, cap2):
        s0, f0 = cap2.process(0.0, 3.0)
        s1, f1 = cap2.process(0.0, 3.0)
        assert s0 == 0.0
        assert s1 == 0.0
        assert f0 == 3.0
        assert f1 == 3.0

    def test_epoch_respected(self, cap1):
        s, f = cap1.process(5.0, 2.0)
        assert s == 5.0
        assert f == 7.0

    def test_epoch_waits_for_token(self, cap1):
        cap1.process(0.0, 10.0)
        # epoch=3 but token not free until 10
        s, f = cap1.process(3.0, 2.0)
        assert s == 10.0
        assert f == 12.0

    def test_epoch_after_token_free(self, cap1):
        cap1.process(0.0, 2.0)
        # token free at 2, epoch=5 > 2 → starts at epoch
        s, f = cap1.process(5.0, 1.0)
        assert s == 5.0
        assert f == 6.0

    def test_greedy_picks_earliest_token(self, cap2):
        cap2.process(0.0, 1.0)   # token A free at 1
        cap2.process(0.0, 5.0)   # token B free at 5
        # Next job should get token A (free at 1)
        s, f = cap2.process(0.0, 2.0)
        assert s == 1.0
        assert f == 3.0

    def test_zero_duration(self, cap1):
        s, f = cap1.process(2.0, 0.0)
        assert s == 2.0
        assert f == 2.0

    def test_returns_floats(self, cap1):
        s, f = cap1.process(0.0, 1.0)
        assert isinstance(s, float)
        assert isinstance(f, float)

    def test_finish_gte_start(self, cap4):
        rng = np.random.default_rng(42)
        for _ in range(50):
            s, f = cap4.process(
                float(rng.uniform(0, 10)),
                float(rng.uniform(0, 5))
            )
            assert f >= s


# ── Vector process, no calendar ───────────────────────────────────────────────

class TestVectorNoCalendar:

    def test_basic_vector(self, cap2):
        starts, finishes = cap2.process(0.0, np.array([3.0, 3.0, 2.0, 2.0]))
        # Jobs 0,1 fill both tokens (finish=3); jobs 2,3 get them at 3
        np.testing.assert_array_equal(starts,  [0, 0, 3, 3])
        np.testing.assert_array_equal(finishes, [3, 3, 5, 5])

    def test_vector_single_element(self, cap1):
        starts, finishes = cap1.process(0.0, np.array([4.0]))
        assert starts[0]  == 0.0
        assert finishes[0] == 4.0

    def test_vector_returns_arrays(self, cap2):
        s, f = cap2.process(0.0, np.array([1.0, 2.0]))
        assert isinstance(s, np.ndarray)
        assert isinstance(f, np.ndarray)

    def test_vector_shape(self, cap2):
        n = 10
        s, f = cap2.process(0.0, np.ones(n))
        assert s.shape == (n,)
        assert f.shape == (n,)

    def test_vector_order_matters(self, cap1):
        """Sequential assignment — order of durations affects subsequent starts."""
        cap_a = Capacity(1)
        cap_b = Capacity(1)
        s_a, f_a = cap_a.process(0.0, np.array([1.0, 4.0]))
        s_b, f_b = cap_b.process(0.0, np.array([4.0, 1.0]))
        # cap_a: job0 finishes at 1, job1 starts at 1 finishes at 5
        # cap_b: job0 finishes at 4, job1 starts at 4 finishes at 5
        np.testing.assert_array_equal(f_a, [1.0, 5.0])
        np.testing.assert_array_equal(f_b, [4.0, 5.0])

    def test_vector_consistency_with_scalar(self, cap4):
        """Vector result must match applying scalar process sequentially."""
        durations = np.array([2.0, 1.0, 3.0, 0.5, 2.5, 1.5])
        epoch = 0.0

        # Vector path
        cap_v = Capacity(4)
        sv, fv = cap_v.process(epoch, durations)

        # Scalar path
        cap_s = Capacity(4)
        ss = np.empty(len(durations))
        fs = np.empty(len(durations))
        for i, d in enumerate(durations):
            ss[i], fs[i] = cap_s.process(epoch, float(d))

        np.testing.assert_allclose(sv, ss)
        np.testing.assert_allclose(fv, fs)

    def test_vector_epoch_respected(self, cap2):
        starts, _ = cap2.process(5.0, np.array([1.0, 1.0]))
        assert np.all(starts >= 5.0)

    def test_vector_all_finish_gte_start(self, cap4):
        rng = np.random.default_rng(7)
        durations = rng.uniform(0, 5, size=30)
        starts, finishes = cap4.process(0.0, durations)
        assert np.all(finishes >= starts)


# ── Calendar integration ──────────────────────────────────────────────────────

class TestCalendarIntegration:

    def test_finish_skips_weekend(self, cap2_cal):
        # Start Friday noon (4.5), duration 1 → finishes Monday noon (7.5)
        s, f = cap2_cal.process(4.5, 1.0)
        assert s == pytest.approx(4.5)
        assert f == pytest.approx(7.5)

    def test_two_tokens_with_calendar(self, cap2_cal):
        starts, finishes = cap2_cal.process(0.0, np.array([3.0, 3.0]))
        # Both start Monday 0, both finish Thursday 3 (3 working days)
        np.testing.assert_allclose(starts,   [0.0, 0.0])
        np.testing.assert_allclose(finishes, [3.0, 3.0])

    def test_calendar_affects_token_availability(self, cap2_cal):
        # 5 working days from Monday = end of Friday = calendar day 5 (Saturday)
        cap2_cal.process(0.0, 5.0)  # token A free at day 5 (Saturday)
        cap2_cal.process(0.0, 5.0)  # token B free at day 5 (Saturday)
        # Next job: token free at 5 (Sat), calendar skips to Mon (7), finishes Tue (8)
        s, f = cap2_cal.process(0.0, 1.0)
        assert s == pytest.approx(5.0)
        assert f == pytest.approx(8.0)

    def test_calendar_with_holiday(self, work_week):
        work_week.add_holiday(0)     # Monday 0 off
        cap = Capacity(1, calendar=work_week)
        # duration 1 starting epoch 0 → Mon is off, starts Tue, finishes Wed
        s, f = cap.process(0.0, 1.0)
        assert s == pytest.approx(0.0)
        assert f == pytest.approx(2.0)

    def test_no_calendar_vs_calendar_differ_across_weekend(self, work_week):
        cap_plain = Capacity(1)
        cap_cal   = Capacity(1, calendar=work_week)
        # Start Friday noon, 1 day duration
        _, f_plain = cap_plain.process(4.5, 1.0)
        _, f_cal   = cap_cal.process(4.5, 1.0)
        assert f_plain == pytest.approx(5.5)   # Saturday noon
        assert f_cal   == pytest.approx(7.5)   # Monday noon

    def test_vector_with_calendar_consistency(self, work_week):
        """Vector and scalar paths agree when a calendar is present."""
        cap_v = Capacity(3, calendar=work_week)
        cap_s = Capacity(3, calendar=work_week)
        durations = np.array([2.0, 5.0, 1.0, 3.0, 4.0, 2.0])
        epoch = 0.0

        sv, fv = cap_v.process(epoch, durations)
        ss = np.empty(len(durations))
        fs = np.empty(len(durations))
        for i, d in enumerate(durations):
            ss[i], fs[i] = cap_s.process(epoch, float(d))

        np.testing.assert_allclose(sv, ss, rtol=1e-9)
        np.testing.assert_allclose(fv, fs, rtol=1e-9)


# ── Invariants ────────────────────────────────────────────────────────────────

class TestInvariants:

    def test_token_finish_times_monotonically_assigned(self):
        """With one token, finish times must be non-decreasing."""
        cap = Capacity(1)
        finishes = []
        for d in [1.0, 2.0, 0.5, 3.0]:
            _, f = cap.process(0.0, d)
            finishes.append(f)
        assert all(a <= b for a, b in zip(finishes, finishes[1:]))

    def test_capacity_unchanged_after_processing(self, cap4):
        cap4.process(0.0, np.ones(10))
        assert cap4.capacity == 4

    def test_all_tokens_eventually_used(self):
        """With n jobs and capacity n, all tokens should be occupied."""
        n = 4
        cap = Capacity(n)
        _, finishes = cap.process(0.0, np.ones(n))
        # All tokens are now busy; next job must wait
        _, f_next = cap.process(0.0, 1.0)
        assert f_next > 1.0
