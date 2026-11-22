"""Prompts, bins and the OpenTSLM batch dict. Spec rev 2 §3.4–§3.6, §4."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import numpy as np

from .data import Example, WindowedSeries

K_BINS = 7
LABEL_CANDIDATES = [" A", " B", " C", " D", " E", " F", " G"]


@dataclass
class BinSpec:
    """Equal-mass bins of the log residual for one group (family+horizon, or signal+horizon for auxiliaries)."""
    group: str
    edges: list[float]          # K-1 interior edges, ascending
    mu: list[float]             # mean training residual per bin, length K
    n_train: int
    train_mean: float | None = None   # mean and sd of the training residuals (the Gaussian climatology null; added 2026-11-20); None on old files
    train_sd: float | None = None

    def bin_of(self, residual: float) -> int:
        return int(np.searchsorted(np.asarray(self.edges), residual, side="right"))

    def point_forecast(self, reference: float, probs: np.ndarray) -> float:
        return float(reference * math.exp(float(np.dot(probs, np.asarray(self.mu)))))

    def gaussian_mass(self, mean: float, sd: float) -> np.ndarray:
        """Probability of each bin under N(mean, sd) on the residual, through the interior edges (outer bins unbounded)."""
        from math import erf, sqrt
        cdf = lambda x: 0.5 * (1.0 + erf((x - mean) / (sd * sqrt(2.0))))
        c = [0.0] + [cdf(e) for e in self.edges] + [1.0]
        m = np.diff(np.asarray(c)); m = np.clip(m, 1e-9, None); return m / m.sum()

    def soft_target(self, residual: float, sigma_frac: float = 0.5) -> np.ndarray:
        """Ordinal readout target (ROUND2_PLAN §2): Gaussian centred on the realised residual with sd = sigma_frac x the local bin
        width; the outer bins borrow the adjacent interior bin's width."""
        e = self.edges; k = self.bin_of(residual); widths = [e[i + 1] - e[i] for i in range(len(e) - 1)]
        local = widths[min(max(k - 1, 0), len(widths) - 1)]
        return self.gaussian_mass(residual, max(sigma_frac * local, 1e-6))

    def to_json(self) -> dict:
        return {"group": self.group, "edges": self.edges, "mu": self.mu, "n_train": self.n_train, "train_mean": self.train_mean, "train_sd": self.train_sd}

    @classmethod
    def from_json(cls, d: dict) -> "BinSpec":
        return cls(d["group"], list(d["edges"]), list(d["mu"]), int(d["n_train"]), (None if d.get("train_mean") is None else float(d["train_mean"])), (None if d.get("train_sd") is None else float(d["train_sd"])))


def group_key(ex: Example) -> str:
    return f"{ex.family}/h{ex.horizon}"


def fit_bins(train_examples: list[Example], k: int = K_BINS) -> dict[str, BinSpec]:
    """Fit edges and per-bin means on TRAINING examples only (spec §3.6)."""
    by: dict[str, list[float]] = {}
    for ex in train_examples:
        by.setdefault(group_key(ex), []).append(ex.residual)
    out = {}
    for g, res in by.items():
        r = np.asarray(sorted(res))
        qs = np.quantile(r, [i / k for i in range(1, k)]).tolist()
        bins = np.searchsorted(np.asarray(qs), r, side="right")
        mu = []
        for b in range(k):
            sel = r[bins == b]
            mu.append(float(sel.mean()) if sel.size else float(np.mean(qs[max(b - 1, 0):b + 1] or [0.0])))
        out[g] = BinSpec(g, [float(x) for x in qs], mu, int(r.size), float(r.mean()), float(r.std(ddof=1)) if r.size > 1 else 1.0)
    return out


def save_bins(bins: dict[str, BinSpec], path) -> None:
    with open(path, "w") as f:
        json.dump({g: b.to_json() for g, b in bins.items()}, f, indent=1)


def load_bins(path) -> dict[str, BinSpec]:
    with open(path) as f:
        return {g: BinSpec.from_json(d) for g, d in json.load(f).items()}


# ---------- prompt text ----------

def _fmt(x: float) -> str:
    if x == 0 or not math.isfinite(x):
        return "0"
    mag = abs(x)
    if mag >= 1000:
        return f"{x:,.0f}"
    if mag >= 10:
        return f"{x:.1f}"
    return f"{x:.3f}"


KEY_FREE = False   # cfg key_free=true (2026-11-22): no months, no prices, no per-window statistics in the text, so no text key identifies a training window


def series_text(w: WindowedSeries) -> str:
    if KEY_FREE:
        return f"{w.name} ({w.unit_label}), monthly, {len(w.months)} values; scaled to [-1, 1]:"
    gaps = f", {w.gaps} gaps" if w.gaps else ""
    return (f"{w.name} ({w.unit_label}), monthly {w.months[0]} to {w.months[-1]}, {len(w.months)} values{gaps}; "
            f"last {_fmt(w.last)}, mean {_fmt(w.mean)}, std {_fmt(w.std)}; scaled to [-1, 1]:")


def pre_prompt(ex: Example) -> str:
    if KEY_FREE:
        ref = "EIA's STEO forecast for the target month" if ex.family == "T2" else "the last monthly mean shown"
        return "\n".join(["Task: forecast the WTI crude oil monthly mean price.", f"The target month is {ex.horizon} months after the last month shown. Reference: {ref}.",
                          "Series follow; each is scaled to [-1, 1] over its own window."])
    lines = [
        "Task: forecast the WTI crude oil monthly mean price (USD per barrel).",
        f"Information month: {ex.information_month}. Target month: {ex.target_month} ({ex.horizon} months ahead).",
    ]
    if ex.family == "T2":
        lines.append(f"EIA's STEO forecast for the target month (the reference): {ex.reference:.2f} USD per barrel.")
    else:
        lines.append(f"Reference: the information month's WTI monthly mean, {ex.reference:.2f} USD per barrel.")
    if ex.omitted:
        lines.append("Not available for this month: " + ", ".join(ex.omitted) + ".")
    lines.append("Series follow; each is scaled to [-1, 1] over its own window and its plain-number scale is stated.")
    return "\n".join(lines)


def post_prompt(ex: Example, spec: BinSpec, labels: list[str]) -> str:
    e = spec.edges
    bins = [f"{labels[0].strip()}: ratio below {math.exp(e[0]):.3f}"]
    for i in range(1, len(e)):
        bins.append(f"{labels[i].strip()}: {math.exp(e[i-1]):.3f} to {math.exp(e[i]):.3f}")
    bins.append(f"{labels[-1].strip()}: {math.exp(e[-1]):.3f} and above")
    return ("Question: realised monthly mean divided by the reference falls in which bin? "
            + "; ".join(bins) + ". Answer:")


def to_batch_item(ex: Example, spec: BinSpec, labels: list[str], eos: str, sigma_frac: float = 0.5) -> dict:
    """The dict OpenTSLMSP.pad_and_apply_batch consumes, plus the typed target, the soft ordinal target and bookkeeping."""
    import torch
    b = spec.bin_of(ex.residual)
    return {
        "soft": spec.soft_target(ex.residual, sigma_frac).astype(np.float32).tolist(),   # ordinal target (used only when cfg ordinal=true)
        "pre_prompt": pre_prompt(ex),
        "time_series_text": [series_text(w) for w in ex.series],
        "time_series": [torch.as_tensor(w.scaled, dtype=torch.float32) for w in ex.series],
        "post_prompt": post_prompt(ex, spec, labels),
        "answer": labels[b] + eos,          # text variant (spec §4, config switch)
        "bin": b,                           # typed target
        "task_id": ex.task_id, "family": ex.family, "split": ex.split, "horizon": ex.horizon,
        "reference": ex.reference, "target": ex.target, "residual": ex.residual, "group": group_key(ex),
    }
