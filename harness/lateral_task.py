"""The hidden-point task: windows, visible sets per spacing, labels, scaling, the synthetic sources, and the twin-series filter.
No model code here, so the CPU scripts (leak_demo.py, ceiling4.py, ceiling_cv.py) import this without torch models or OpenTSLM.
36-point windows; 7 anchors (every 6th point and the last). Stage 1 shows the anchors, stage 2 the anchors and their immediate
neighbours, stage 3 everything but the asked point. Question: is the hidden point at a fixed position above or below the nearest
visible point to its left? Asked positions 16, 17, 18 (counting from 1) give (k, m) = (3, 3), (3, 1), (1, 1)."""
import numpy as np
def _torch():
    import torch; return torch
L = 36; ANCH = list(range(0, L, 6)) + [L - 1]          # 0,6,...,30,35 : 7 anchors spanning the window
QPOS = {0: 15, 1: 16, 2: 17}   # the asked position per stage: between anchors 12 and 18 (stage 1), next to a stage-2 neighbour (16), interior (17)
Q = "Some points of the sequence are hidden (shown as 0). Is the hidden point at position {pos} (counting from 1) below or above the nearest visible point to its LEFT? A: below; B: above. Answer:"
def visible_sets():
    s1 = set(ANCH); s2 = set(ANCH) | {a + d for a in ANCH for d in (-1, 1) if 0 <= a + d < L}; s3 = set(range(L)); return [s1, s2, s3]
STAGES = visible_sets()
def gen_synth(rng):
    k = rng.integers(0, 3)
    if k == 0: x = np.cumsum(rng.normal(0, 1, L)); x = np.convolve(x, np.ones(3) / 3, mode="same")
    elif k == 1: x = rng.uniform(10, 100) * np.sin(2 * np.pi * np.arange(L) / rng.uniform(6, 20) + rng.uniform(0, 6.3)) + rng.normal(0, 2, L)
    else: x = rng.uniform(-5, 5) * np.arange(L) + rng.normal(0, 1, L)
    return x + rng.uniform(1000, 9000)
SCALE_MODE = "window"   # "window" = v2/v3 as run (standardised over ALL 36 points, the hidden ones included: LEAKS, see notes/ROUND3_SPEC_2026-11-22.md); "visible" = mean and sd from the shown points only
def gen_dose(rng, dose):
    """Noise-dose source (2026-11-22): sqrt(1-dose) * sine + sqrt(dose) * random walk, each normalised to unit one-step increment variance,
    so `dose` is the share of the one-step change's variance that is walk. dose 0 = a clean sine; dose 1 = a pure random walk."""
    t = np.arange(L); s = np.sin(2 * np.pi * t / rng.uniform(6, 20) + rng.uniform(0, 6.3)); s = s / np.sqrt(np.mean(np.diff(s) ** 2))
    w = np.cumsum(rng.normal(0, 1, L)); return np.sqrt(1 - dose) * s + np.sqrt(dose) * w + rng.uniform(1000, 9000)
def make_items(seqs, stage, labels, eos, rng, per_seq=2):
    vis = STAGES[stage]; items = []; i = QPOS[stage]
    for x in seqs:
        shown = set(vis) - {i}
        left = max([j for j in shown if j < i]); y = int(x[i] > x[left])
        ref = x[sorted(shown)] if SCALE_MODE == "visible" else x
        m, s = ref.mean(), ref.std() + 1e-9; sc = np.clip((x - m) / s / 3, -1, 1); masked = np.where([j in shown for j in range(L)], sc, 0.0).astype(np.float32)
        items.append({"pre_prompt": f"A sequence of {L} points follows as a time series; some points are hidden and shown as 0. Visible positions (counting from 1): {', '.join(str(j + 1) for j in sorted(shown))}.",
                      "time_series_text": ["sequence (scaled to [-1, 1], hidden points = 0):"], "time_series": [_torch().as_tensor(masked)],
                      "post_prompt": Q.format(pos=i + 1), "answer": labels[y] + eos, "bin": y, "stage": stage})
    return items
def twin_series(panel, eval_names, thresh=0.8, min_common=36):
    """Series outside eval_names whose month-on-month changes correlate above `thresh` in absolute value with an eval series over
    their common months: {name: [eval twin, correlation]}. A twin in training makes 'held out' false."""
    from oilharness.data import monthly_complete
    d = {}
    for n, s in panel.series.items():
        months, vals = monthly_complete(s, 1 << 62); d[n] = {m: float(v) for m, v in zip(months, vals) if np.isfinite(v)}
    out = {}
    for n in sorted(d):
        if n in eval_names: continue
        for e in sorted(eval_names):
            cm = sorted(set(d[n]) & set(d[e]))
            if len(cm) < min_common: continue
            a = np.diff([d[n][m] for m in cm]); b = np.diff([d[e][m] for m in cm])
            if a.std() < 1e-12 or b.std() < 1e-12: continue
            c = float(np.corrcoef(a, b)[0, 1])
            if abs(c) > thresh and (n not in out or abs(c) > abs(out[n][1])): out[n] = [e, round(c, 3)]
    return out
