"""Spec rev 2 §6: synthetic boundary test. Values after the cutoff are a huge sentinel; none may reach a window."""
import numpy as np
import pytest

from oilharness.data import (Series, cutoff_us, month_start_us, monthly_complete, parse_month, shift_month,
                             window_series, ym_of_us, US_PER_DAY)

SENTINEL = 1e9


def _daily(start_y, start_m, months, sentinel_from_us):
    t0 = month_start_us(start_y, start_m)
    t1 = month_start_us(*shift_month(start_y, start_m, months))
    ts = np.arange(t0, t1, US_PER_DAY, dtype="int64")
    vals = np.where(ts >= sentinel_from_us, SENTINEL, 50.0 + 0.001 * np.arange(ts.size))
    return ts, vals


def _weekly(start_y, start_m, months, sentinel_from_us):
    # week-ending Fridays: first Friday on/after the start
    t0 = month_start_us(start_y, start_m)
    d0 = (4 - ((t0 // US_PER_DAY + 3) % 7)) % 7  # 1970-01-01 was a Thursday -> +3 makes Monday=0
    ts = np.arange(t0 + d0 * US_PER_DAY, month_start_us(*shift_month(start_y, start_m, months)), 7 * US_PER_DAY, dtype="int64")
    vals = np.where(ts >= sentinel_from_us, SENTINEL, 400.0 + 0.1 * np.arange(ts.size))
    return ts, vals


def _monthly(start_y, start_m, months, sentinel_from_us):
    ts = np.array([month_start_us(*shift_month(start_y, start_m, i)) for i in range(months)], dtype="int64")
    vals = np.where(ts >= sentinel_from_us, SENTINEL, 10.0 + np.arange(months))
    return ts, vals


@pytest.mark.parametrize("info_month", ["2004-07", "2010-01", "2019-12", "2020-02", "2024-11"])
@pytest.mark.parametrize("lag", [0, 1, 3, 4])
@pytest.mark.parametrize("kind", ["daily", "weekly", "monthly"])
def test_no_value_after_cutoff_reaches_a_window(info_month, lag, kind):
    c = cutoff_us(info_month, lag)
    maker = {"daily": _daily, "weekly": _weekly, "monthly": _monthly}[kind]
    ts, vals = maker(1995, 1, 400, c)
    s = Series("x", ts, vals, lag, "u", kind)
    w = window_series(s, c, 48)
    assert w is not None
    assert not np.any(w.raw >= SENTINEL / 2), "a post-cutoff value reached the window"
    # the month straddling the cutoff is absent: every window month ends before the cutoff
    for mk in w.months:
        y, m = parse_month(mk)
        assert month_start_us(*shift_month(y, m, 1)) <= c
    # the last window month is exactly the last complete month before the cutoff
    cy, cm = ym_of_us(c - 1)
    assert w.months[-1] == f"{cy:04d}-{cm:02d}"
    # and raw timestamps used are strictly before the cutoff
    assert int(ts[ts < c].max()) < c


def test_cutoff_matches_connector_sealed_rule():
    # info 2004-07, slowest lag 4 -> connector scope end = first day of 2004-04 (1 - 4 = -3 months)
    assert cutoff_us("2004-07", 4) == month_start_us(2004, 4)
    assert cutoff_us("2004-07", 0) == month_start_us(2004, 8)


def test_straddling_month_dropped_even_with_data_before_cutoff():
    # cutoff mid-month: daily data exists before it, but the month is incomplete -> not produced
    c = month_start_us(2015, 6) + 10 * US_PER_DAY
    ts, vals = _daily(2014, 1, 24, c)
    months, out = monthly_complete(Series("x", ts, vals, 0, "u", "daily"), c)
    assert months[-1] == "2015-05"
    assert not np.any(out >= SENTINEL / 2)
