"""All real 36-month windows, cross-validated by series (5 folds; no series is in both train and test of a fold). CPU only.
Per spacing: the neighbour rule against the random walk's closed form, and whether logistic regression or boosted trees beat the rule.
Scaling is from the visible points only (the leak-free instrument). usage: PYTHONPATH=. python ceiling_cv.py configs/recall_llama1b.json OUT.json"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.model_selection import GroupKFold
from ceiling4 import feats, fit_predict, rw
from lateral_task import L
from oilharness.aux import default_aux_signals
from oilharness.data import DEFAULT_FEATURES, Panel, find_panel_record, monthly_complete
from timenet.reader import TimeFReader
cfg = json.loads(Path(sys.argv[1]).read_text())
with TimeFReader.open(cfg["registry_version_dir"]) as r:
    rec = find_panel_record(r); names = [s.name for s in rec.walk_signals()]; aux = default_aux_signals(names); panel = Panel.from_record(rec, sorted(set(DEFAULT_FEATURES) | set(aux)))
W = []; G = []
for n, s in panel.series.items():
    if n == "wti_monthly_mean": continue
    months, vals = monthly_complete(s, 1 << 62); v = np.asarray(vals, dtype=float)
    for end in range(L, len(v) + 1, 4):
        w = v[end - L:end]
        if np.isfinite(w).all() and w.std() > 1e-9: W.append(w); G.append(n)
G = np.array(G); out = {"n_windows": len(W), "n_series": int(len(set(G)))}; ser = sorted(set(G)); idx = {s: np.where(G == s)[0] for s in ser}
for st in range(3):
    X, Y, li, ri, (k, m) = feats(W, st); P = {r: np.zeros(len(Y), int) for r in ("rule", "logit", "gbm")}
    for tr, te in GroupKFold(n_splits=5).split(X, Y, G):
        p = fit_predict(X[tr], Y[tr], X[te], li, ri)
        for r in P: P[r][te] = p[r]
    row = dict(n=len(Y), k=int(k), m=int(m), random_walk_closed_form=rw(k, m), **{r: float((P[r] == Y).mean()) for r in P}); rng = np.random.default_rng(0)
    for name, d in (("rule_minus_closed_form", (P["rule"] == Y) - rw(k, m)), ("logit_minus_rule", (P["logit"] == Y).astype(float) - (P["rule"] == Y)), ("gbm_minus_rule", (P["gbm"] == Y).astype(float) - (P["rule"] == Y))):
        bs = [d[np.concatenate([idx[s] for s in rng.choice(ser, len(ser))])].mean() for _ in range(2000)]; row[name + "_ci95_by_series"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
    out[f"spacing_{k}_{m}"] = row; print(f"k={k} m={m}", json.dumps({a: (round(b, 3) if isinstance(b, float) else [round(c, 3) for c in b] if isinstance(b, list) else b) for a, b in row.items()}))
Path(sys.argv[2]).write_text(json.dumps(out, indent=1))
