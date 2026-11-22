"""Training, temperature scaling and evaluation. Spec rev 2 §4–§5, §7–§8."""
from __future__ import annotations

import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from .examples import BinSpec


def batches(items: list[dict], size: int, shuffle: bool, seed: int):
    idx = list(range(len(items)))
    if shuffle:
        random.Random(seed).shuffle(idx)
    for i in range(0, len(idx), size):
        yield [items[j] for j in idx[i:i + size]]


def linear_warmup(optimizer, total_steps: int, warmup_frac: float):
    warm = max(1, int(total_steps * warmup_frac))
    def f(step):
        if step < warm:
            return (step + 1) / warm
        return max(0.0, (total_steps - step) / max(1, total_steps - warm))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def run_epoch(model, items, cfg, optimizer=None, scheduler=None, seed=0, text_variant=False) -> float:
    training = optimizer is not None
    model.train() if training else model.eval()
    total, n = 0.0, 0
    for batch in batches(items, cfg["batch_size"], shuffle=training, seed=seed):
        with torch.set_grad_enabled(training):
            loss = model.text_loss(batch) if text_variant else model.loss(batch)
        if training:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            params = [p for g in optimizer.param_groups for p in g["params"]]
            torch.nn.utils.clip_grad_norm_(params, cfg["grad_clip"])
            optimizer.step(); scheduler.step()
        total += float(loss.detach()) * len(batch); n += len(batch)
    return total / max(n, 1)


@torch.no_grad()
def hard_nll(model, items, cfg) -> float:
    """Cross-entropy on the HARD bin (the control's selection criterion), regardless of the training loss (2026-11-20, review amendment 4)."""
    model.eval(); total, n = 0.0, 0
    for batch in batches(items, cfg["batch_size"], shuffle=False, seed=0):
        logits = model.typed_logits(batch); target = torch.as_tensor([b["bin"] for b in batch], device=logits.device)
        total += float(torch.nn.functional.cross_entropy(logits, target)) * len(batch); n += len(batch)
    return total / max(n, 1)


@torch.no_grad()
def predict(model, items, cfg) -> list[dict]:
    model.eval()
    out = []
    for batch in batches(items, cfg["batch_size"], shuffle=False, seed=0):
        probs = model.probabilities(batch).cpu().numpy()
        for item, p in zip(batch, probs):
            out.append({**{k: item[k] for k in ("task_id", "family", "split", "horizon", "reference", "target", "residual", "group", "bin")},
                        "probs": p.tolist()})
    return out


@torch.no_grad()
def fit_temperature(model, val_items, cfg) -> float:
    """Temperature scaling on the validation split (spec §4, §10)."""
    model.eval()
    logits, targets = [], []
    for batch in batches(val_items, cfg["batch_size"], shuffle=False, seed=0):
        logits.append(model.typed_logits(batch).cpu()); targets += [b["bin"] for b in batch]
    L = torch.cat(logits); T = torch.as_tensor(targets)
    best_t, best_nll = 1.0, float("inf")
    for t in np.exp(np.linspace(math.log(0.25), math.log(4.0), 49)):
        nll = float(torch.nn.functional.cross_entropy(L / float(t), T))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    return best_t


def score(preds: list[dict], bins: dict[str, BinSpec]) -> dict:
    """Per family and horizon: n, RMSE/MAE of the point forecast, the reference's RMSE/MAE, Brier and log score
    for the model and for the reference (its bin, the one holding residual 0, at probability 1)."""
    cells = defaultdict(list)
    for p in preds:
        cells[(p["family"], p["horizon"])].append(p)
    table = {}
    for (fam, h), rows in sorted(cells.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        spec = bins[f"{fam}/h{h}"]
        K = len(spec.mu)
        err_m, err_r, brier_m, brier_r, ll_m, ll_r = [], [], [], [], [], []
        for p in rows:
            probs = np.asarray(p["probs"]); y = p["bin"]
            point = spec.point_forecast(p["reference"], probs)
            err_m.append(point - p["target"]); err_r.append(p["reference"] - p["target"])
            onehot = np.zeros(K); onehot[y] = 1
            brier_m.append(float(np.sum((probs - onehot) ** 2)))
            ref_bin = spec.bin_of(0.0); ref_probs = np.zeros(K); ref_probs[ref_bin] = 1
            brier_r.append(float(np.sum((ref_probs - onehot) ** 2)))
            ll_m.append(-math.log(max(probs[y], 1e-12))); ll_r.append(0.0 if ref_bin == y else -math.log(1e-12))
        e_m, e_r = np.asarray(err_m), np.asarray(err_r)
        # climatology: the training bins are equal-mass, so the null forecast is uniform over K bins
        clim = np.full(K, 1.0 / K)
        brier_c = float(np.mean([np.sum((clim - np.eye(K)[p["bin"]]) ** 2) for p in rows]))
        ll_c = float(math.log(K))
        # Gaussian climatology (ROUND2_PLAN §0; ROLLING_2026-11-19 §2): N(train mean, train sd) through the bin edges, no features
        if spec.train_sd is None:
            brier_g, ll_g = float("nan"), float("nan")      # old bins.json without train moments: no Gaussian column (review note 6)
        else:
            gclim = spec.gaussian_mass(spec.train_mean, spec.train_sd)
            brier_g = float(np.mean([np.sum((gclim - np.eye(K)[p["bin"]]) ** 2) for p in rows]))
            ll_g = float(np.mean([-math.log(max(gclim[p["bin"]], 1e-12)) for p in rows]))
        table[f"{fam}/h{h}"] = {
            "family": fam, "h": h, "n": len(rows),
            "model_rmse": float(np.sqrt(np.mean(e_m ** 2))), "model_mae": float(np.mean(np.abs(e_m))),
            "reference_rmse": float(np.sqrt(np.mean(e_r ** 2))), "reference_mae": float(np.mean(np.abs(e_r))),
            "model_brier": float(np.mean(brier_m)), "climatology_brier": brier_c, "reference_brier": float(np.mean(brier_r)),
            "model_logscore": float(np.mean(ll_m)), "climatology_logscore": ll_c, "reference_logscore": float(np.mean(ll_r)),
            "gaussian_climatology_brier": brier_g, "gaussian_climatology_logscore": ll_g,
        }
    return table


def calibration(preds: list[dict], k: int) -> dict:
    """Reliability: for each bin label, mean predicted probability vs observed frequency."""
    out = {}
    for b in range(k):
        pm = np.asarray([p["probs"][b] for p in preds]); ob = np.asarray([1.0 if p["bin"] == b else 0.0 for p in preds])
        out[b] = {"mean_prob": float(pm.mean()) if pm.size else None, "observed": float(ob.mean()) if ob.size else None}
    return out


def table_md(table: dict, information_set: str, baselines_path: str) -> str:
    ref_name = {"per-signal": "random walk (T1) / STEO (T2)", "sealed": "sealed random walk (T1) / STEO (T2)"}[information_set]
    lines = [f"| family | h | n | model RMSE | model MAE | reference RMSE | reference MAE | model Brier | uniform Brier | Gaussian clim. Brier | model log score | uniform log score | Gaussian clim. log score |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for key, c in table.items():
        lines.append(f"| {c['family']} | {c['h']} | {c['n']} | {c['model_rmse']:.2f} | {c['model_mae']:.2f} | {c['reference_rmse']:.2f} | {c['reference_mae']:.2f} | {c['model_brier']:.3f} | {c['climatology_brier']:.3f} | {c.get('gaussian_climatology_brier', float('nan')):.3f} | {c['model_logscore']:.3f} | {c['climatology_logscore']:.3f} | {c.get('gaussian_climatology_logscore', float('nan')):.3f} |")
    lines.append("")
    lines.append(f"Information set: {information_set}; reference column = {ref_name}; uniform = 1/7 over the equal-mass training bins; Gaussian clim. = N(train residual mean, sd) through the bin edges, "
                 f"the ordinal null (the reference's own scores as a point mass are in table.json). USD per barrel on the WTI monthly mean; "
                 f"test split = target months 2022-01 to 2026-10; compare with `{baselines_path}`.")
    return "\n".join(lines)
