"""Spec rev 2 §6 on the built dataset: every window month ends before its signal's cutoff; STEO is the only exemption; n per cell."""
import random
from collections import Counter

import pytest

from oilharness.data import (DEFAULT_FEATURES, Panel, build_example, cutoff_us, find_panel_record, month_start_us,
                             parse_month, shift_month, task_rows)

REG = "reg/sethlei/oil-fundamentals-prices/1.0.0"
BASELINE_N = {("T1", 1): 48, ("T1", 3): 48, ("T1", 6): 48, ("T1", 12): 46, ("T2", 1): 57, ("T2", 3): 55, ("T2", 6): 52, ("T2", 12): 46}


@pytest.fixture(scope="module")
def loaded():
    from timenet.reader import TimeFReader
    with TimeFReader.open(REG) as r:
        rows = task_rows(r)
        panel = Panel.from_record(find_panel_record(r), DEFAULT_FEATURES)
    return rows, panel


def test_cells_match_baselines(loaded):
    rows, _ = loaded
    n = Counter((r.family, r.horizon) for r in rows if r.split == "test")
    assert dict(n) == BASELINE_N


@pytest.mark.parametrize("information_set", ["per-signal", "sealed"])
def test_windows_end_before_cutoff(loaded, information_set):
    rows, panel = loaded
    rng = random.Random(7)
    for row in rng.sample(rows, 40):
        ex = build_example(row, panel, DEFAULT_FEATURES, 48, information_set)
        for w in ex.series:
            c = ex.cutoffs[w.name]
            for mk in w.months:
                y, m = parse_month(mk)
                assert month_start_us(*shift_month(y, m, 1)) <= c, (row.task_id, w.name, mk)
        assert ex.max_ts_used < max(ex.cutoffs.values())
        # the only forward-looking input is the STEO forecast, and only on T2
        assert (row.steo_forecast is not None) == (row.family == "T2")
        # T1 reference = the information month's own WTI mean (price lag 0), i.e. the random-walk baseline
        if row.family == "T1" and information_set == "per-signal":
            wti = next(w for w in ex.series if w.name == "wti_monthly_mean")
            assert wti.months[-1] == row.information_month


def test_no_test_example_is_dropped(loaded):
    rows, panel = loaded
    for row in (r for r in rows if r.split == "test"):
        ex = build_example(row, panel, DEFAULT_FEATURES, 48, "per-signal")
        assert ex.target > 0 and ex.reference > 0
