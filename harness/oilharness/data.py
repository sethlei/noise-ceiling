"""Data layer: tasks, the panel, the information set, cutoff-then-aggregate, windows, examples.

Spec rev 2 §1–§3. Every timestamp in this module is ABSOLUTE microseconds since the Unix epoch
(the panel record's origin is 1920-01-01, so offsets are shifted by ORIGIN_US on load).
"""
from __future__ import annotations

import calendar
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

ORIGIN_US = -1577923200000000  # 1920-01-01T00:00:00Z, the panel record's TimeOrigin
US_PER_DAY = 86_400_000_000

DEFAULT_FEATURES = [
    "wti_monthly_mean", "eia_RWTC", "eia_RBRTE", "eia_RCLC1", "eia_WCESTUS1", "eia_MCRFPUS2",
    "eia_intl_WORL", "eia_intl_OPEC", "cftc_pm_net", "frb_dollar", "gpr_gpr",
]
COMPUTED = {"cftc_pm_net": ("cftc_067651_prod_merc_long", "cftc_067651_prod_merc_short"),  # m_money columns are not in the built panel
            "frb_dollar": ("frb_DTWEXM", "frb_DTWEXBGS")}
MIN_WINDOW_VALUES = 6
HAIRCUT_DAYS = {"cftc_pm_net": 7, "frb_dollar": 7}  # pass-2 N3: as-of vs release gap inside a lag-0 month
STD_FLOOR = 1e-9


# ---------- calendar helpers ----------

def month_start_us(year: int, month: int) -> int:
    return int(datetime(year, month, 1, tzinfo=timezone.utc).timestamp()) * 1_000_000


def shift_month(year: int, month: int, by: int) -> tuple[int, int]:
    idx = year * 12 + (month - 1) + by
    return idx // 12, idx % 12 + 1


def parse_month(key: str) -> tuple[int, int]:
    y, m = key.split("-")
    return int(y), int(m)


def month_key(year: int, month: int) -> str:
    return f"{year:04d}-{month:02d}"


def cutoff_us(information_month: str, lag_months: int) -> int:
    """Exclusive cutoff for a signal with publication lag L at information month m: first day of (m - L + 1)."""
    y, m = parse_month(information_month)
    return month_start_us(*shift_month(y, m, 1 - lag_months))


def ym_of_us(ts_us: int) -> tuple[int, int]:
    d = datetime.fromtimestamp(ts_us / 1_000_000, tz=timezone.utc)
    return d.year, d.month


# ---------- the panel ----------

@dataclass
class Series:
    name: str
    ts_us: np.ndarray          # absolute, int64, ascending
    values: np.ndarray         # float64, NaN = missing
    lag_months: int
    unit_label: str
    frequency: str


@dataclass
class Panel:
    series: dict[str, Series] = field(default_factory=dict)

    @classmethod
    def from_record(cls, record, names: list[str]) -> "Panel":
        """Load the named signals (and the parts of computed features) from the panel Record."""
        needed = set()
        for n in names:
            needed.update(COMPUTED.get(n, (n,)))
        out = cls()
        for sig in record.walk_signals():
            if sig.name not in needed:
                continue
            ann = {a.key: a.value for a in sig.annotations}
            vals = _values_with_nan(sig)
            ts = _offsets_us(sig) + ORIGIN_US
            order = np.argsort(ts, kind="stable")
            out.series[sig.name] = Series(
                name=sig.name, ts_us=ts[order], values=vals[order],
                lag_months=int(ann.get("publication_lag_months", 0)),
                unit_label=_text(ann.get("unit_label", "")), frequency=_text(ann.get("frequency", "")),
            )
        missing = needed - set(out.series)
        if missing:
            raise KeyError(f"panel lacks signals: {sorted(missing)}")
        for n in names:
            if n in COMPUTED:
                out.series[n] = _computed(n, out)
        return out


def _text(value) -> str:
    """Annotation values may be text handles; read them to plain str."""
    read = getattr(value, "read", None)
    return str(read()) if callable(read) else str(value)


def _offsets_us(sig) -> np.ndarray:
    """Per-value time offsets for a regular or irregular axis (a regular axis computes them, it stores none)."""
    from timenet.dataset.axis import IrregularAxis
    axis = sig.time_axis
    if isinstance(axis, IrregularAxis):
        return np.asarray(sig.time_offsets_us(), dtype="int64")
    n = int(sig.n_values)
    if n == 0:
        return np.zeros(0, dtype="int64")
    first = int(axis.time_offset_us(0))
    step = int(axis.time_offset_us(1)) - first if n > 1 else 0
    return first + step * np.arange(n, dtype="int64")


def _values_with_nan(sig) -> np.ndarray:
    """Signal values as float64 with NaN for stored nulls (to_numpy() refuses nulls; to_arrow() keeps them)."""
    arr = sig.to_arrow()
    if hasattr(arr, "combine_chunks"):
        arr = arr.combine_chunks()
    if hasattr(arr, "to_numpy"):
        out = arr.to_numpy(zero_copy_only=False)
    else:
        out = np.asarray(arr)
    out = np.asarray(out, dtype="float64")
    if out.ndim > 1:
        out = out.reshape(out.shape[0], -1)[:, 0]
    return out


def _computed(name: str, panel: Panel) -> Series:
    a, b = (panel.series[k] for k in COMPUTED[name])
    if name == "cftc_pm_net":
        if not np.array_equal(a.ts_us, b.ts_us):
            raise ValueError("cftc long/short axes differ")
        return Series(name, a.ts_us, a.values - b.values, max(a.lag_months, b.lag_months), a.unit_label, a.frequency)
    if name == "frb_dollar":
        # splice DTWEXM (to 2019) onto DTWEXBGS (2006->) by the ratio of their 2006-01 means
        lo, hi = month_start_us(2006, 1), month_start_us(2006, 2)
        ma = np.nanmean(a.values[(a.ts_us >= lo) & (a.ts_us < hi)])
        mb = np.nanmean(b.values[(b.ts_us >= lo) & (b.ts_us < hi)])
        ratio = mb / ma
        keep_a = a.ts_us < lo
        ts = np.concatenate([a.ts_us[keep_a], b.ts_us])
        vals = np.concatenate([a.values[keep_a] * ratio, b.values])
        out = Series(name, ts, vals, max(a.lag_months, b.lag_months), b.unit_label, b.frequency)
        out.splice_us = lo  # pass-2 N2: windows whose cutoff <= splice_us use DTWEXM unscaled (see window_series)
        out.unspliced = a
        return out
    raise KeyError(name)


# ---------- cutoff then aggregate (spec §3.2) ----------

def _monthly_table(series: Series) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-month mean of the raw values, cached on the Series: (month_start_us, month_end_us, mean) ascending,
    with NaN for months that hold no non-NaN value, over the contiguous month range of the series."""
    cached = getattr(series, "_monthly", None)
    if cached is not None:
        return cached
    ts, vals = series.ts_us, series.values
    if ts.size == 0:
        out = (np.zeros(0, dtype="int64"), np.zeros(0, dtype="int64"), np.zeros(0))
        series._monthly = out
        return out
    # month index = years*12 + month-1, computed from timestamps in bulk
    days = ts // US_PER_DAY
    dates = (days.astype("datetime64[D]"))
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    months = (dates.astype("datetime64[M]").astype(int) % 12) + 1
    midx = years * 12 + (months - 1)
    lo, hi = int(midx.min()), int(midx.max())
    n = hi - lo + 1
    sums = np.zeros(n); counts = np.zeros(n)
    ok = np.isfinite(vals)
    np.add.at(sums, (midx - lo)[ok], vals[ok]); np.add.at(counts, (midx - lo)[ok], 1)
    means = np.where(counts > 0, sums / np.maximum(counts, 1), np.nan)
    starts = np.array([month_start_us((lo + i) // 12, (lo + i) % 12 + 1) for i in range(n)], dtype="int64")
    ends = np.array([month_start_us((lo + i + 1) // 12, (lo + i + 1) % 12 + 1) for i in range(n)], dtype="int64")
    out = (starts, ends, means)
    series._monthly = out
    return out


def monthly_complete(series: Series, cutoff: int) -> tuple[list[str], np.ndarray]:
    """Monthly values using only months COMPLETE before the cutoff: month_end <= cutoff.

    Every raw value of a complete month has timestamp < month_end <= cutoff, so this equals "filter raw values
    with ts < cutoff, then aggregate complete months" (spec §3.2) and never produces the straddling month.
    Months with no non-NaN raw value are NaN gaps. Daily and weekly series are averaged per month.
    """
    starts, ends, means = _monthly_table(series)
    keep = ends <= cutoff
    if not keep.any():
        return [], np.zeros(0)
    k = int(np.flatnonzero(keep)[-1]) + 1
    months = [month_key(*ym_of_us(int(t))) for t in starts[:k]]
    return months, means[:k].copy()


# ---------- windows and examples (spec §3.3–§3.6) ----------

@dataclass
class WindowedSeries:
    name: str
    months: list[str]
    raw: np.ndarray            # last W monthly values, NaN gaps kept
    scaled: np.ndarray         # standardised/3, clipped to [-1,1], NaN gaps filled with 0 AFTER scaling
    mean: float
    std: float
    last: float
    gaps: int
    unit_label: str


def window_series(series: Series, cutoff: int, width: int) -> WindowedSeries | None:
    if getattr(series, "splice_us", None) is not None and cutoff <= series.splice_us:
        series = series.unspliced
    months, vals = monthly_complete(series, cutoff)
    if len(months) == 0:
        return None
    months, vals = months[-width:], vals[-width:]
    finite = vals[np.isfinite(vals)]
    if finite.size < MIN_WINDOW_VALUES:
        return None
    mean, std = float(finite.mean()), float(finite.std())
    if std < STD_FLOOR:
        return None
    scaled = np.clip((vals - mean) / std / 3.0, -1.0, 1.0)
    scaled = np.where(np.isfinite(scaled), scaled, 0.0)
    last_idx = int(np.flatnonzero(np.isfinite(vals))[-1])
    return WindowedSeries(series.name, months, vals, scaled, mean, std, float(vals[last_idx]),
                          int(np.sum(~np.isfinite(vals))), series.unit_label)


@dataclass
class TaskRow:
    task_id: str
    family: str
    split: str
    horizon: int
    information_month: str
    target_month: str
    target: float
    steo_forecast: float | None
    sealed_end_us: int


def task_rows(reader) -> list[TaskRow]:
    from timenet.reader import LoadPlan, Loading
    rows = []
    for t in reader.iter_tasks(load=LoadPlan(text=Loading.EAGER)):
        md = t.metadata
        rows.append(TaskRow(
            task_id=str(t.id), family=str(md["family"]), split=str(t.split),
            horizon=int(md["horizon_months"]), information_month=str(md["information_month"]),
            target_month=str(md["target_month"]), target=float(t.targets[0]),
            steo_forecast=(float(md["steo_forecast"]) if "steo_forecast" in md else None),
            sealed_end_us=int(t.scope.end_us),
        ))
    return rows


def find_panel_record(reader):
    for rec in reader.iter_records():
        if rec.metadata.get("kind") == "panel":
            return rec
    raise LookupError("no panel record")


@dataclass
class Example:
    task_id: str
    family: str
    split: str
    horizon: int
    information_month: str
    target_month: str
    target: float
    reference: float
    residual: float            # ln(target / reference)
    series: list[WindowedSeries]
    omitted: list[str]
    max_ts_used: int           # the largest raw timestamp that reached any window (for the leakage tests)
    cutoffs: dict[str, int]


def build_example(row: TaskRow, panel: Panel, features: list[str], width: int, information_set: str) -> Example:
    if information_set not in ("per-signal", "sealed"):
        raise ValueError(information_set)
    series_out, omitted, cutoffs, max_ts = [], [], {}, -(1 << 62)
    for name in features:
        s = panel.series[name]
        c = row.sealed_end_us if information_set == "sealed" else cutoff_us(row.information_month, s.lag_months)
        c -= HAIRCUT_DAYS.get(name, 0) * US_PER_DAY
        cutoffs[name] = c
        w = window_series(s, c, width)
        if w is None:
            omitted.append(name)
            continue
        w.name = name  # a spliced series may have windowed its unspliced part; the feature keeps its own name
        series_out.append(w)
        used = s.ts_us[s.ts_us < c]
        if used.size:
            max_ts = max(max_ts, int(used.max()))
    if row.family == "T2":
        if row.steo_forecast is None:
            raise ValueError(f"T2 task {row.task_id} without steo_forecast")
        reference = row.steo_forecast
    else:
        wti = next((w for w in series_out if w.name == "wti_monthly_mean"), None)
        if wti is None:
            raise ValueError(f"T1 task {row.task_id}: no admissible wti_monthly_mean window")
        reference = wti.last
    if not (row.target > 0 and reference > 0):
        raise ValueError(f"task {row.task_id}: non-positive target/reference ({row.target}, {reference}); log ratio undefined")
    return Example(row.task_id, row.family, row.split, row.horizon, row.information_month, row.target_month,
                   row.target, reference, math.log(row.target / reference), series_out, omitted, max_ts, cutoffs)
