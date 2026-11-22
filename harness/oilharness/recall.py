"""Recall tasks: the target is a function of the input window only (round-2 plan §1; Seth 2026-11-19: separate recall from
prediction). Same batch dict as the prediction task; `bin` carries the recall label; K differs per task."""
from __future__ import annotations
import numpy as np
from .data import Example
from .examples import pre_prompt, series_text

def _wti(ex: Example):
    return next((w for w in ex.series if w.name == "wti_monthly_mean"), None)

def _stocks(ex: Example):
    return next((w for w in ex.series if w.name == "eia_WCESTUS1"), None)

def _logchange12(w) -> float | None:
    r = np.asarray(w.raw, dtype=float)
    if len(r) < 13 or not (np.isfinite(r[-1]) and np.isfinite(r[-13])) or r[-1] <= 0 or r[-13] <= 0: return None
    return float(np.log(r[-1] / r[-13]))

TASKS = {
    "R1": dict(k=2, question="Over the last 12 months of the WTI series shown, did the price go up or down? A: down; B: up. Answer:"),
    "R2": dict(k=3, question="In which third of the WTI series shown is its highest value? A: first third; B: middle third; C: last third. Answer:"),
    "R3": dict(k=7, question="Where does the last WTI value sit relative to the series' own mean and standard deviation? Bins from lowest to highest: A, B, C, D, E, F, G (equal-mass). Answer:"),
    "R4": dict(k=2, question="Over the last 12 months of the US crude stocks series shown, did stocks go up or down? A: down; B: up. Answer:"),
}

def recall_value(task: str, ex: Example) -> float | None:
    """The raw quantity the label is cut from; None = example unusable for this task."""
    if task == "R1":
        w = _wti(ex); return None if w is None else _logchange12(w)
    if task == "R4":
        w = _stocks(ex); return None if w is None else _logchange12(w)
    if task == "R2":
        w = _wti(ex)
        if w is None: return None
        r = np.asarray(w.raw, dtype=float)
        if not np.isfinite(r).any(): return None
        return float(np.nanargmax(r) / len(r))
    if task == "R3":
        w = _wti(ex)
        if w is None: return None
        r = np.asarray(w.raw, dtype=float); m = np.nanmean(r); s = np.nanstd(r)
        return None if not (np.isfinite(r[-1]) and s > 0) else float((r[-1] - m) / s)
    raise KeyError(task)

def label_of(task: str, v: float, edges: list[float] | None) -> int:
    if task in ("R1", "R4"): return int(v > 0)
    if task == "R2": return 0 if v < 1 / 3 else (1 if v < 2 / 3 else 2)
    if task == "R3": return int(np.searchsorted(np.asarray(edges), v, side="right"))
    raise KeyError(task)

def to_recall_item(ex: Example, task: str, label: int, labels: list[str], eos: str) -> dict:
    import torch
    return {
        "pre_prompt": pre_prompt(ex),
        "time_series_text": [series_text(w) for w in ex.series],
        "time_series": [torch.as_tensor(w.scaled, dtype=torch.float32) for w in ex.series],
        "post_prompt": TASKS[task]["question"],
        "answer": labels[label] + eos, "bin": label,
        "task_id": ex.task_id, "family": ex.family, "split": ex.split, "horizon": ex.horizon,
        "reference": ex.reference, "target": ex.target, "residual": ex.residual, "group": f"{ex.family}/h{ex.horizon}",
    }
