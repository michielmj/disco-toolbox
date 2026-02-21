"""
tests/calendar/test_calendar.py

Covers:
  - Basic scalar elapse (no holidays)
  - Weekend/non-working day skipping
  - Fractional start times
  - Fractional pattern weights
  - Zero-duration tasks
  - Holiday overrides (add, remove, update)
  - Half-day holidays
  - NumPy array inputs (1-D, 2-D, broadcasting)
  - Horizon auto-extension
  - Edge cases (all-zero pattern, zero-weight days, large durations)
"""

import numpy as np
import pytest

from toolbox.calendar.calendar import Calendar, CalendarError


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def work_week():
    """Standard Mon–Fri calendar, cycle length 7."""
    return Calendar([1, 1, 1, 1, 1, 0, 0])


@pytest.fixture
def uniform():
    """Every day is a full working day."""
    return Calendar([1.0])


@pytest.fixture
def half_days():
    """Every day delivers only half a duration unit."""
    return Calendar([0.5])


# ── Helpers ───────────────────────────────────────────────────────────────────

def assert_close(a, b, rel=1e-9):
    assert abs(a - b) <= rel * max(abs(b), 1.0), f"{a} != {b}"


# ── Scalar elapse: no holidays ────────────────────────────────────────────────

class TestScalarNoHolidays:

    def test_zero_duration_returns_start(self, work_week):
        assert work_week.elapse(3.0, 0.0) == 3.0

    def test_negative_duration_returns_start(self, work_week):
        assert work_week.elapse(3.0, -1.0) == 3.0

    def test_one_full_day(self, work_week):
        # Start Mon 0, consume 1 day → finish end of Mon (= start of Tue = 1.0)
        assert_close(work_week.elapse(0.0, 1.0), 1.0)

    def test_five_days_fills_week(self, work_week):
        assert_close(work_week.elapse(0.0, 5.0), 5.0)

    def test_six_days_crosses_weekend(self, work_week):
        # 5 working days Mon–Fri, then 1 more → lands on next Mon
        assert_close(work_week.elapse(0.0, 6.0), 8.0)

    def test_ten_days_crosses_two_weekends(self, work_week):
        assert_close(work_week.elapse(0.0, 10.0), 12.0)

    def test_fractional_start_within_same_day(self, work_week):
        # Start Mon 0.5, need 0.25 → finishes at 0.75 same day
        assert_close(work_week.elapse(0.5, 0.25), 0.75)

    def test_fractional_start_spills_to_next_day(self, work_week):
        # Start Mon 0.5, need 0.75 → consumes 0.5 remaining in Mon, 0.25 into Tue
        assert_close(work_week.elapse(0.5, 0.75), 1.25)

    def test_friday_noon_to_monday_noon(self, work_week):
        # Day 4 = Friday; start at 4.5, need 1.0 → Mon noon = 7.5
        assert_close(work_week.elapse(4.5, 1.0), 7.5)

    def test_start_on_weekend_skips_to_monday(self, work_week):
        # Day 5 = Saturday (weight 0); task should start on Monday (day 7)
        assert_close(work_week.elapse(5.0, 1.0), 8.0)

    def test_start_mid_weekend(self, work_week):
        assert_close(work_week.elapse(5.5, 1.0), 8.0)

    def test_uniform_calendar_is_identity(self, uniform):
        assert_close(uniform.elapse(0.0, 3.7), 3.7)
        assert_close(uniform.elapse(2.3, 1.0), 3.3)

    def test_half_day_calendar_doubles_elapsed(self, half_days):
        # Each day = 0.5, so 1.0 duration needs 2 calendar days
        assert_close(half_days.elapse(0.0, 1.0), 2.0)
        assert_close(half_days.elapse(0.0, 0.5), 1.0)

    def test_large_duration_many_cycles(self, work_week):
        # 250 working days = 50 weeks; last day is Fri of week 50 = day 347
        # finishes at end of day 347 = 348.0
        assert_close(work_week.elapse(0.0, 250.0), 348.0)

    def test_fractional_finish_within_day(self, uniform):
        assert_close(uniform.elapse(0.0, 0.5), 0.5)
        assert_close(uniform.elapse(1.0, 0.3), 1.3)


# ── Pattern variants ──────────────────────────────────────────────────────────

class TestPatternVariants:

    def test_alternating_pattern(self):
        # [1, 0] → every other day works
        cal = Calendar([1, 0])
        assert_close(cal.elapse(0.0, 1.0), 1.0)
        assert_close(cal.elapse(0.0, 2.0), 3.0)
        assert_close(cal.elapse(0.0, 3.0), 5.0)

    def test_half_weight_pattern(self):
        cal = Calendar([0.5, 0.5])
        # 1.0 duration at weight 0.5/day = 2 days
        assert_close(cal.elapse(0.0, 1.0), 2.0)

    def test_mixed_weights(self):
        cal = Calendar([1, 0.5, 0])
        assert_close(cal.elapse(0.0, 1.0), 1.0)
        assert_close(cal.elapse(0.0, 1.5), 2.0)
        assert_close(cal.elapse(0.0, 2.0), 3.5)

    def test_single_day_pattern(self):
        cal = Calendar([1.0])
        assert_close(cal.elapse(0.0, 5.0), 5.0)

    def test_cycle_work_property(self):
        cal = Calendar([1, 1, 1, 1, 1, 0, 0])
        assert_close(cal.cycle_work, 5.0)

    def test_invalid_negative_weight_raises(self):
        with pytest.raises(CalendarError):
            Calendar([1, -0.5, 1])

    def test_all_zero_pattern_raises_on_elapse(self):
        cal = Calendar([0, 0, 0])
        with pytest.raises(CalendarError):
            cal.elapse(0.0, 1.0)

    def test_empty_pattern_raises(self):
        with pytest.raises(CalendarError):
            Calendar([])


# ── Holiday overrides ─────────────────────────────────────────────────────────

class TestHolidayOverrides:

    def test_add_full_holiday(self, work_week):
        work_week.add_holiday(0)   # Monday 0 is off
        # Need 1 working day; day 0 off → starts Tue, finishes Wed start
        assert_close(work_week.elapse(0.0, 1.0), 2.0)

    def test_add_two_holidays_same_week(self, work_week):
        work_week.add_holiday(0)
        work_week.add_holiday(1)   # Mon + Tue off
        assert_close(work_week.elapse(0.0, 1.0), 3.0)

    def test_holiday_in_second_week(self, work_week):
        work_week.add_holiday(7)   # Second Monday off
        assert_close(work_week.elapse(0.0, 6.0), 9.0)

    def test_half_day_holiday(self, work_week):
        work_week.add_holiday(0, 0.5)   # Mon = half day
        # 0.5 consumed on day 0; remaining 0.5 on day 1
        assert_close(work_week.elapse(0.0, 1.0), 1.5)

    def test_remove_holiday_restores_weight(self, work_week):
        work_week.add_holiday(0)
        work_week.remove_holiday(0)
        assert_close(work_week.elapse(0.0, 1.0), 1.0)

    def test_update_holiday_weight(self, work_week):
        work_week.add_holiday(0, 0.5)
        work_week.add_holiday(0, 0.0)   # upgrade to full day off
        assert_close(work_week.elapse(0.0, 1.0), 2.0)

    def test_remove_nonexistent_holiday_is_noop(self, work_week):
        result_before = work_week.elapse(0.0, 3.0)
        work_week.remove_holiday(99)
        assert_close(work_week.elapse(0.0, 3.0), result_before)

    def test_holidays_property(self, work_week):
        work_week.add_holiday(3, 0.5)
        work_week.add_holiday(7, 0.0)
        h = work_week.holidays
        assert h[3] == pytest.approx(0.5)
        assert h[7] == pytest.approx(0.0)

    def test_holiday_on_already_zero_weight_day(self, work_week):
        # Day 5 is already a weekend (weight 0); overriding it should not break anything
        work_week.add_holiday(5, 0.0)
        assert_close(work_week.elapse(0.0, 5.0), 5.0)

    def test_holidays_at_construction(self):
        cal = Calendar([1, 1, 1, 1, 1, 0, 0], holidays={0: 0.0, 7: 0.0})
        assert_close(cal.elapse(0.0, 5.0), 9.0)

    def test_many_holidays_large_duration(self, work_week):
        # Add a holiday every Monday for 10 weeks (days 0,7,...,63)
        for week in range(10):
            work_week.add_holiday(week * 7)
        # 10 weeks × 4 working days = 40 days by day 70
        # 10 remaining days over 2 normal weeks → finishes end of day 81
        assert_close(work_week.elapse(0.0, 50.0), 82.0)


# ── NumPy array inputs ────────────────────────────────────────────────────────

class TestNumPyInputs:

    def test_1d_array(self, work_week):
        starts    = np.array([0.0, 0.0, 0.0])
        durations = np.array([1.0, 5.0, 6.0])
        result    = work_week.elapse(starts, durations)
        expected  = np.array([1.0, 5.0, 8.0])
        np.testing.assert_allclose(result, expected)

    def test_2d_array(self, work_week):
        starts = np.array([[0.0, 4.0], [7.0, 9.0]])
        durations = np.array([[2.0, 1.0], [5.0, 3.0]])
        result = work_week.elapse(starts, durations)
        expected = np.array([[2.0, 5.0], [12.0, 12.0]])
        np.testing.assert_allclose(result, expected)

    def test_broadcast_scalar_duration(self, work_week):
        starts = np.array([0.0, 7.0, 14.0])
        result = work_week.elapse(starts, 5.0)
        np.testing.assert_allclose(result, [5.0, 12.0, 19.0])

    def test_broadcast_scalar_start(self, work_week):
        durations = np.array([1.0, 5.0, 10.0])
        result    = work_week.elapse(0.0, durations)
        np.testing.assert_allclose(result, [1.0, 5.0, 12.0])

    def test_array_zero_durations(self, work_week):
        starts    = np.array([1.5, 2.5, 3.5])
        durations = np.array([0.0, 0.0, 0.0])
        result    = work_week.elapse(starts, durations)
        np.testing.assert_allclose(result, starts)

    def test_array_mixed_zero_and_nonzero(self, work_week):
        starts    = np.array([0.0, 0.0])
        durations = np.array([0.0, 5.0])
        result    = work_week.elapse(starts, durations)
        np.testing.assert_allclose(result, [0.0, 5.0])

    def test_array_shape_preserved(self, work_week):
        starts    = np.zeros((3, 4))
        durations = np.ones((3, 4))
        result    = work_week.elapse(starts, durations)
        assert result.shape == (3, 4)

    def test_scalar_returns_float(self, work_week):
        result = work_week.elapse(0.0, 1.0)
        assert isinstance(result, float)

    def test_array_consistency_with_scalar(self, work_week):
        """Array and scalar paths must agree on every element."""
        rng   = np.random.default_rng(42)
        starts    = rng.uniform(0, 20, size=50)
        durations = rng.uniform(0, 15, size=50)
        array_result = work_week.elapse(starts, durations)
        scalar_results = np.array(
            [work_week.elapse(float(s), float(d))
             for s, d in zip(starts, durations)]
        )
        np.testing.assert_allclose(array_result, scalar_results, rtol=1e-9)

    def test_array_with_holidays_consistency(self, work_week):
        """Array and scalar paths must agree when holidays are present."""
        work_week.add_holiday(0)
        work_week.add_holiday(7, 0.5)
        rng       = np.random.default_rng(0)
        starts    = rng.uniform(0, 10, size=30)
        durations = rng.uniform(0, 8,  size=30)
        array_result  = work_week.elapse(starts, durations)
        scalar_results = np.array(
            [work_week.elapse(float(s), float(d))
             for s, d in zip(starts, durations)]
        )
        np.testing.assert_allclose(array_result, scalar_results, rtol=1e-9)


# ── Horizon extension ─────────────────────────────────────────────────────────

class TestHorizonExtension:

    def test_large_start_extends_horizon(self, work_week):
        # Start well beyond default horizon
        result = work_week.elapse(5000.0, 1.0)
        assert result > 5000.0

    def test_large_duration_extends_horizon(self, work_week):
        result = work_week.elapse(0.0, 10_000.0)
        assert result > 0.0

    def test_horizon_property_grows(self, work_week):
        initial = work_week.horizon
        work_week.elapse(0.0, 50_000.0)
        assert work_week.horizon > initial

    def test_add_holiday_beyond_horizon_extends(self, work_week):
        work_week.add_holiday(work_week.horizon + 100)
        assert work_week.horizon > work_week.horizon - 100


# ── Monotonicity and ordering invariants ──────────────────────────────────────

class TestInvariants:

    def test_finish_always_gte_start(self, work_week):
        rng = np.random.default_rng(7)
        s = rng.uniform(0, 50, 100)
        d = rng.uniform(0, 20, 100)
        assert np.all(work_week.elapse(s, d) >= s)

    def test_larger_duration_gives_later_finish(self, work_week):
        s = np.full(20, 0.0)
        d = np.linspace(1, 20, 20)
        results = work_week.elapse(s, d)
        assert np.all(np.diff(results) >= 0)

    def test_later_start_gives_later_finish(self, work_week):
        s = np.linspace(0, 28, 29)
        d = np.full(29, 3.0)
        results = work_week.elapse(s, d)
        assert np.all(np.diff(results) >= 0)

    def test_elapse_zero_then_remainder_equals_full(self, work_week):
        """elapse(elapse(s, a), b) == elapse(s, a+b) for working days."""
        rng = np.random.default_rng(13)
        for _ in range(20):
            s = float(rng.uniform(0, 10))
            a = float(rng.uniform(0.1, 5))
            b = float(rng.uniform(0.1, 5))
            mid      = work_week.elapse(s, a)
            chained  = work_week.elapse(mid, b)
            direct   = work_week.elapse(s, a + b)
            assert_close(chained, direct, rel=1e-9)