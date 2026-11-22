"""Auxiliary training examples built by the harness (spec rev 2 §9). Training split only, targets <= 2018-12."""
from __future__ import annotations

import math

from .data import (Example, Panel, Series, cutoff_us, month_key, monthly_complete, parse_month, shift_month,
                   window_series, HAIRCUT_DAYS, US_PER_DAY)

AUX_LAST_TARGET = "2018-12"
AUX_FIRST_ORIGIN = "2000-01"
HORIZONS = (1, 3, 6, 12)


def default_aux_signals(panel_names: list[str]) -> list[str]:
    out = ["eia_MCRFPUS2"]
    out += sorted(n for n in panel_names if n.startswith("eia_intl_") and n[9:].isalpha() and n[9:].isupper() and len(n) == 13)
    out += sorted(n for n in panel_names if n.startswith("jodi_") and n.endswith("_INDPROD"))
    return out


def aux_examples(panel: Panel, aux_series: dict[str, Series], features: list[str], width: int,
                 max_examples: int | None = None, seed: int = 0) -> list[Example]:
    """For each auxiliary signal, origin month and horizon: features as in §3 at information month m,
    target = the signal's own complete-month value at m + h, reference = its last admissible value."""
    import random
    out = []
    y0, m0 = parse_month(AUX_FIRST_ORIGIN)
    # enumerate (signal, origin, horizon) triples first; sample BEFORE building when capped (build cost dominates)
    triples = []
    for name in aux_series:
        y, m = y0, m0
        while month_key(y, m) <= AUX_LAST_TARGET:
            for h in HORIZONS:
                if month_key(*shift_month(y, m, h)) <= AUX_LAST_TARGET:
                    triples.append((name, y, m, h))
            y, m = shift_month(y, m, 1)
    rng = random.Random(seed)
    if max_examples is not None and len(triples) > 3 * max_examples:
        triples = rng.sample(triples, 3 * max_examples)
    by_month_cache: dict[str, dict] = {}
    for name, y, m, h in triples:
        s = aux_series[name]
        if name not in by_month_cache:
            months_all, vals_all = monthly_complete(s, 1 << 62)
            by_month_cache[name] = {mk: v for mk, v in zip(months_all, vals_all.tolist())}
        by_month = by_month_cache[name]
        info = month_key(y, m)
        if True:
            if True:
                tk = month_key(*shift_month(y, m, h))
                target = by_month.get(tk)
                if target is None or not math.isfinite(target) or target <= 0:
                    continue
                c = cutoff_us(info, s.lag_months)
                own = window_series(s, c, width)
                if own is None or not (own.last > 0):
                    continue
                series_out, omitted, cutoffs, max_ts = [own], [], {name: c}, int(s.ts_us[s.ts_us < c].max())
                for f in features:
                    if f == name:
                        continue
                    fs = panel.series[f]
                    fc = cutoff_us(info, fs.lag_months) - HAIRCUT_DAYS.get(f, 0) * US_PER_DAY
                    cutoffs[f] = fc
                    w = window_series(fs, fc, width)
                    if w is None:
                        omitted.append(f); continue
                    w.name = f
                    series_out.append(w)
                    used = fs.ts_us[fs.ts_us < fc]
                    if used.size:
                        max_ts = max(max_ts, int(used.max()))
                out.append(Example(f"aux:{name}:{info}:h{h}", f"aux:{name}", "train", h, info, tk, float(target),
                                   float(own.last), math.log(target / own.last), series_out, omitted, max_ts, cutoffs))
                if max_examples is not None and len(out) >= max_examples:
                    break
    rng.shuffle(out)
    return out[:max_examples] if max_examples is not None else out
