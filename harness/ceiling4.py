"""Ceilings and scoring for round 3 (lateral4: leak-free scaling, held-out series, twins dropped, noise dose). CPU only. 2026-11-22, Daniel.
Rebuilds exactly the items run_lateral3.py --scale visible builds (asserted against each run's saved labels), then per source and stage:
  rule (sign(right - left), no training) · logit · gbm, fit on the run's own training sequences, scored on the run's own eval items;
  dose: gbm fit on 50,000 fresh sequences, scored on 20,000 (the ceiling with data not the limit) and the random-walk closed form.
With --runs DIR: every lateral4 run's final-phase accuracy beside those, the agree/disagree split, per-item overlap with the rule.
usage: ceiling4.py configs/lateral_llama1b.json --out OUT.json [--runs results_dir]"""
from __future__ import annotations
import argparse, json, re, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from lateral_task import L, QPOS, STAGES, gen_synth, gen_dose
from lateral_task import twin_series

def feats(seqs, stage):
    """Mirror make_items in SCALE_MODE 'visible': mean/sd from the shown points only, /3, clip; only the shown values are returned."""
    i = QPOS[stage]; shown = sorted(set(STAGES[stage]) - {i}); left = max(j for j in shown if j < i); right = min(j for j in shown if j > i)
    X = np.empty((len(seqs), len(shown))); Y = np.empty(len(seqs), int)
    for r, x in enumerate(seqs):
        v = x[shown]; X[r] = np.clip((v - v.mean()) / (v.std() + 1e-9) / 3, -1, 1); Y[r] = int(x[i] > x[left])
    return X, Y, shown.index(left), shown.index(right), (i - left, right - i)
def local(X, li, ri):
    cols = [X, (X[:, ri] - X[:, li])[:, None]]
    if li >= 1: cols.append((X[:, li] - X[:, li - 1])[:, None])
    if ri + 1 < X.shape[1]: cols.append((X[:, ri + 1] - X[:, ri])[:, None])
    return np.hstack(cols)
def fit_predict(Xtr, Ytr, Xev, li, ri, which=("logit", "gbm")):
    out = {"rule": (Xev[:, ri] > Xev[:, li]).astype(int)}
    if "logit" in which: out["logit"] = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000)).fit(local(Xtr, li, ri), Ytr).predict(local(Xev, li, ri))
    if "gbm" in which: out["gbm"] = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, max_depth=3, random_state=0).fit(local(Xtr, li, ri), Ytr).predict(local(Xev, li, ri))
    return out
def rw(k, m): return float(1 - np.arccos(np.sqrt(k / (k + m))) / np.pi)

def build(source, cfg, dose=None, n_tr=1500, n_ev=600):
    rng = np.random.default_rng(0); meta = {}
    if source == "synth": tr = [gen_synth(rng) for _ in range(n_tr)]; ev = [gen_synth(rng) for _ in range(n_ev)]
    elif source == "dose": tr = [gen_dose(rng, dose) for _ in range(n_tr)]; ev = [gen_dose(rng, dose) for _ in range(n_ev)]
    else:
        from oilharness.aux import default_aux_signals
        from oilharness.data import DEFAULT_FEATURES, Panel, find_panel_record, monthly_complete
        from timenet.reader import TimeFReader
        with TimeFReader.open(cfg["registry_version_dir"]) as r:
            rec = find_panel_record(r); names = [s.name for s in rec.walk_signals()]; aux = default_aux_signals(names); panel = Panel.from_record(rec, sorted(set(DEFAULT_FEATURES) | set(aux)))
        seqs = []
        for n, s in panel.series.items():
            months, vals = monthly_complete(s, 1 << 62); v = np.asarray(vals, dtype=float)
            for end in range(L, len(v) + 1, 4):
                w = v[end - L:end]
                if np.isfinite(w).all() and w.std() > 1e-9 and (n != "wti_monthly_mean" or months[end - 1] >= "2019-01"): seqs.append((n, months[end - 1], w))
        rng.shuffle(seqs); wti = [w for n, m, w in seqs if n == "wti_monthly_mean"]
        names_o = sorted({n for n, m, w in seqs if n != "wti_monthly_mean"}); held = set(np.random.default_rng(7).choice(names_o, 16, replace=False).tolist())
        twins = twin_series(panel, held | {"wti_monthly_mean"})
        tr = [w for n, m, w in seqs if n != "wti_monthly_mean" and n not in held and n not in twins][:n_tr]; evn = [(n, w) for n, m, w in seqs if n in held][:n_ev]
        ev = [w for _, w in evn] + wti; meta = dict(heldout_series=sorted(held), twins=twins, eval_series=[n for n, _ in evn] + ["wti_monthly_mean"] * len(wti), n_wti=len(wti))
    return tr, ev, meta

def cell(source, cfg, dose=None, n_ev=600, runs=None, tag=None):
    tr, ev, meta = build(source, cfg, dose, n_ev=n_ev); out = {"meta": {k: v for k, v in meta.items() if k != "eval_series"}, "stages": {}}
    labels = [feats(ev, st)[1] for st in range(3)]; ser = np.array(meta.get("eval_series", []))
    big = None
    if source == "dose":
        b = np.random.default_rng(12345); big_tr = [gen_dose(b, dose) for _ in range(50000)]; e = np.random.default_rng(999); big_ev = [gen_dose(e, dose) for _ in range(20000)]; big = (big_tr, big_ev)
    for st in range(3):
        Xtr, Ytr, li, ri, (k, m) = feats(tr, st); Xev, Yev, _, _, _ = feats(ev, st); P = fit_predict(Xtr, Ytr, Xev, li, ri)
        row = dict(n=int(len(Yev)), k=int(k), m=int(m), random_walk_closed_form=rw(k, m), majority=float(max(Yev.mean(), 1 - Yev.mean())), **{r: float((p == Yev).mean()) for r, p in P.items()})
        if big:
            Xb, Yb, _, _, _ = feats(big[0], st); X2, Y2, _, _, _ = feats(big[1], st); pb = fit_predict(Xb, Yb, X2, li, ri, which=("gbm",))
            row["ceiling_gbm_50k_20k"] = float((pb["gbm"] == Y2).mean()); row["rule_20k"] = float((pb["rule"] == Y2).mean())
        ag = None
        if st > 0:
            ag = labels[st] == labels[st - 1]; row["n_agree"] = int(ag.sum()); row["n_disagree"] = int((~ag).sum())
            for r, p in P.items(): row[f"{r}_agree"] = float((p[ag] == Yev[ag]).mean()); row[f"{r}_disagree"] = float((p[~ag] == Yev[~ag]).mean())
        if len(ser): row["rule_wti_only"] = float((P["rule"][ser == "wti_monthly_mean"] == Yev[ser == "wti_monthly_mean"]).mean())
        lm = {}
        for d in sorted(Path(runs).glob(f"lateral4_*_{tag}_*s[0-9]_*")) if runs else []:
            f = d / "eval_predictions.npz"; key = f"after_phase3_stage{st+1}"
            if not f.exists(): continue
            z = np.load(f)
            if key not in z: continue
            arm, ntr, seed = re.match(r"lateral4_([A-F])_[a-z0-9]+?(_n\d+)?_s(\d)_", d.name).groups()
            if ntr and source != "real": continue      # a different training size redraws the synthetic eval set; those runs are scored from their own tables
            assert len(z[key + "_y"]) == len(Yev) and (z[key + "_y"] == Yev).all(), f"label mismatch against {d.name}: these are not the run's items"
            p = (z[key] > 0.5).astype(int)
            e = dict(top1=float((p == Yev).mean()), agrees_with_rule=float((p == P["rule"]).mean()), lm_right_rule_wrong=int(((p == Yev) & (P["rule"] != Yev)).sum()), rule_right_lm_wrong=int(((p != Yev) & (P["rule"] == Yev)).sum()))
            if ag is not None: e["agree"] = float((p[ag] == Yev[ag]).mean()); e["disagree"] = float((p[~ag] == Yev[~ag]).mean())
            if len(ser): e["wti_only"] = float((p[ser == "wti_monthly_mean"] == Yev[ser == "wti_monthly_mean"]).mean())
            for ph in (1, 2):
                kk = f"after_phase{ph}_stage{st+1}"
                if kk in z: e[f"top1_after_phase{ph}"] = float(((z[kk] > 0.5).astype(int) == Yev).mean())
            lm[f"{arm}{ntr or ''}_s{seed}"] = e
        row["lm"] = lm; out["stages"][f"stage{st+1}"] = row
    return out

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--out", required=True); ap.add_argument("--runs", default=""); ap.add_argument("--skip-real", action="store_true", help="synthetic and dose cells only (no dataset build needed)"); a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()); res = {} if a.skip_real else {"real": cell("real", cfg, runs=a.runs, tag="real")}
    res["synth"] = cell("synth", cfg, n_ev=300, runs=a.runs, tag="synth")
    for d in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0): res[f"dose{int(round(100*d)):03d}"] = cell("dose", cfg, dose=d, runs=a.runs, tag=f"dose{int(round(100*d)):03d}")
    Path(a.out).write_text(json.dumps(res, indent=1))
    for src, c in res.items():
        for st, row in c["stages"].items():
            print(src, st, " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items() if k != "lm"))
            for name, e in row["lm"].items(): print("    LM", name, " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in e.items()))
if __name__ == "__main__": sys.exit(main())
