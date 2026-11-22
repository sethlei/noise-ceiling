"""Recall v2: generic reading skill, no memorisation keys. (The v1 recall prompts carried dates, the reference price and
last/mean/std for every series — unique keys per window and, for R3, the answer in text; train loss went to ~0 in 3 epochs
while validation exploded. This version strips all of that.)
Windows: 48 complete months of ANY monthly series in the panel (the 11 features + the auxiliary production signals), random end
month; the prompt names nothing (no series name, no dates, no numbers). Question: did the series go up or down over its last 12
months (sign of log change). Train: 3000 windows from non-WTI series, end month <= 2018-12. Eval: (a) 400 held-out windows from
OTHER non-WTI series, (b) WTI windows ending at every month 2019-01..2026-09 (the transfer that matters).
usage: run_recall2.py configs/recall_llama1b.json [--device cuda] [--n-train 3000] [--limit N]"""
from __future__ import annotations
import argparse, json, os, platform, sys, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
HF_TOKEN_FILE = Path(os.environ.get("HF_TOKEN_FILE", "hf_token.txt"))
if "HF_TOKEN" not in os.environ and HF_TOKEN_FILE.exists(): os.environ["HF_TOKEN"] = HF_TOKEN_FILE.read_text().strip()
import numpy as np, torch
from oilharness.aux import default_aux_signals
from oilharness.data import DEFAULT_FEATURES, Panel, find_panel_record, monthly_complete
from oilharness.model import TypedOpenTSLM, pick_device, single_token_labels
from oilharness.train import batches, linear_warmup, run_epoch
from run import Receipt, git_sha
W = 48
QS = {"updown": "Over the last 12 months of the series shown, did it go up or down? A: down; B: up. Answer:",
      "amplitude": "Is the series shown more volatile (larger spread) over its last 24 months than over its first 24 months? A: no; B: yes. Answer:",
      "absvol": "Is the series shown more volatile than is usual for a series of this kind? A: no; B: yes. Answer:"}
def windows_of(name, months, vals, ends, task="updown", scale="window", gstats=None):
    out = []
    for end in ends:
        if end < W: continue
        v = np.asarray(vals[end - W:end], dtype=float)
        if not (np.isfinite(v[-1]) and np.isfinite(v[-13]) and v[-1] > 0 and v[-13] > 0): continue
        f = v[np.isfinite(v)]
        if f.size < 36 or f.std() < 1e-9: continue
        if scale == "window": mu, sd = f.mean(), f.std()
        elif scale == "global": mu, sd = gstats[name]   # global: the series' TRAINING-era mean and sd (level and amplitude survive)
        elif scale == "centerglobal": mu, sd = f.mean(), gstats[name][1]   # centerglobal: level sd over the training years (too coarse: squashes windows)
        else: mu, sd = f.mean(), gstats[name][3]   # centerwin: the series' TYPICAL WINDOW spread (median window sd over training windows) — units of a typical remainder (Seth): start in the middle, keep the units — the remainder in a fixed scale
        scaled = np.clip((v - mu) / sd / 3.0, -1, 1); scaled = np.where(np.isfinite(scaled), scaled, 0.0).astype(np.float32)
        if task == "updown": y = int(np.log(v[-1] / v[-13]) > 0)
        elif task == "absvol":
            lv = np.log(v[v > 0]) if (v > 0).sum() >= 36 else None
            if lv is None or len(gstats[name]) < 3 or not np.isfinite(gstats[name][2]): continue
            y = int(np.std(np.diff(lv)) > gstats[name][2])
        else:
            a, b = v[:24], v[24:]; a, b = a[np.isfinite(a)], b[np.isfinite(b)]
            if a.size < 12 or b.size < 12: continue
            y = int(np.std(np.diff(np.log(b[b > 0]))) > np.std(np.diff(np.log(a[a > 0])))) if (a > 0).all() and (b > 0).all() else int(b.std() > a.std())
        out.append(dict(name=name, end_month=months[end - 1], scaled=scaled, y=y))
    return out
def to_item(w, labels, eos):
    return {"pre_prompt": "A single monthly time series follows, scaled to [-1, 1]. No values are given in text.", "time_series_text": ["series:"],
            "time_series": [torch.as_tensor(w["scaled"])], "post_prompt": QS[TASK], "answer": labels[w["y"]] + eos, "bin": w["y"], "name": w["name"], "end_month": w["end_month"]}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--device", default="auto"); ap.add_argument("--n-train", type=int, default=3000); ap.add_argument("--limit", type=int, default=0); ap.add_argument("--task", default="updown", choices=list(QS)); ap.add_argument("--scale", default="window", choices=["window", "global", "centerglobal", "centerwin"]); a = ap.parse_args()
    global TASK; TASK = a.task
    cfg = json.loads(Path(a.config).read_text()); K = 2
    run_id = f"recall2_{a.task}_{a.scale}_{Path(a.config).stem}_{time.strftime('%Y%m%d_%H%M%S')}"; out = HERE / "results" / run_id; out.mkdir(parents=True, exist_ok=True)
    rc = Receipt(out / "RECEIPT.json", {**cfg, "recall2": a.task, "scale": a.scale, "K": K, "n_train": a.n_train})
    try:
        device = pick_device(a.device); torch.manual_seed(0); rng = np.random.default_rng(0)
        rc.set(status="RUNNING", device=device, platform=platform.platform(), torch=torch.__version__, sha={"box": git_sha(str(HERE.parent))})
        from timenet.reader import TimeFReader
        with TimeFReader.open(cfg["registry_version_dir"]) as r:
            rec = find_panel_record(r); names = [s.name for s in rec.walk_signals()]; aux = default_aux_signals(names)
            panel = Panel.from_record(rec, sorted(set(DEFAULT_FEATURES) | set(aux)))
        tables = {n: monthly_complete(panel.series[n], 1 << 62) for n in panel.series}
        gstats = {}
        for n, (mo, va_) in tables.items():
            tv = np.asarray([x for m_, x in zip(mo, va_) if m_ <= "2018-12" and np.isfinite(x)], dtype=float)
            if tv.size >= 36 and tv.std() > 1e-9:
                pos = tv[tv > 0]; lv = np.log(pos) if pos.size >= 60 else None
                typ = float(np.median([np.std(np.diff(lv[i:i + W])) for i in range(0, len(lv) - W + 1, 6)])) if lv is not None and len(lv) >= W + 6 else float("nan")
                wsd = float(np.median([np.std(tv[i:i + W]) for i in range(0, len(tv) - W + 1, 6)])) if len(tv) >= W + 6 else float("nan")
                gstats[n] = (float(tv.mean()), float(tv.std()), typ, wsd)
            else: gstats[n] = (float("nan"), float("nan"), float("nan"), float("nan"))
        other = [n for n in tables if n != "wti_monthly_mean" and len(tables[n][0]) >= W + 13 and np.isfinite(gstats[n][1])]
        rng.shuffle(other); held = set(other[: max(5, len(other) // 8)]); train_names = [n for n in other if n not in held]
        def sample(names_, n, max_end_month=None, seed=0):
            r_ = np.random.default_rng(seed); pool = []
            for nm in names_:
                months, vals = tables[nm]; ends = [i for i in range(W, len(months) + 1) if (max_end_month is None or months[i - 1] <= max_end_month)]
                pool += windows_of(nm, months, vals, ends, a.task, a.scale, gstats)
            r_.shuffle(pool); return pool[:n]
        tr = sample(train_names, a.n_train, "2018-12", 1); ev_other = sample(sorted(held), 400, None, 2)
        wm, wv = tables["wti_monthly_mean"]; ev_wti = windows_of("wti_monthly_mean", wm, wv, [i for i in range(W, len(wm) + 1) if wm[i - 1] >= "2019-01"], a.task, a.scale, gstats)
        va = sample(train_names, 300, "2018-12", 3)  # validation = more train-era windows from train series (different sample)
        if a.limit: tr, va, ev_other, ev_wti = tr[: a.limit], va[: a.limit], ev_other[: a.limit], ev_wti[: a.limit]
        model = TypedOpenTSLM(cfg["llm_id"], device, lora=cfg.get("lora", True), lora_r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32))
        model.labels, model.label_ids = single_token_labels(model.model.tokenizer, K); model.label_ids_t = torch.as_tensor(model.label_ids, device=device)
        if cfg.get("gradient_checkpointing", True): model.enable_gradient_checkpointing()
        if cfg.get("checkpoint_repo"):
            from huggingface_hub import hf_hub_download
            ck = torch.load(hf_hub_download(cfg["checkpoint_repo"], "model_checkpoint.pt"), map_location="cpu", weights_only=False)
            model.model.encoder.load_state_dict(ck["encoder_state"]); model.model.projector.load_state_dict(ck["projector_state"])
        eos = model.model.get_eos_token(); I = lambda ws: [to_item(w, model.labels, eos) for w in ws]
        tr_i, va_i, eo_i, ew_i = I(tr), I(va), I(ev_other), I(ev_wti)
        base = np.bincount([it["bin"] for it in tr_i], minlength=K) / max(1, len(tr_i))
        rc.set(n_examples=dict(train=len(tr_i), validation=len(va_i), eval_other=len(eo_i), eval_wti=len(ew_i)), n_series=dict(train=len(train_names), held=len(held)), label_dist_train=base.tolist(), labels=model.labels)
        opt = torch.optim.AdamW(model.param_groups(cfg["lr_encoder"], cfg["lr_projector"], cfg["lr_lora"], cfg["weight_decay"]))
        steps = cfg["epochs"] * max(1, (len(tr_i) + cfg["batch_size"] - 1) // cfg["batch_size"]); sched = linear_warmup(opt, steps, cfg.get("warmup_frac", 0.1))
        best, best_epoch, bad = float("inf"), -1, 0; ck_best = out / "best.pt"; t0 = time.time()
        for epoch in range(cfg["epochs"]):
            te = time.time(); tl = run_epoch(model, tr_i, cfg, opt, sched, seed=epoch); vl = run_epoch(model, va_i, cfg)
            rc.event(epoch=epoch, train_loss=round(tl, 4), val_loss=round(vl, 4), seconds=round(time.time() - te, 1))
            if vl < best: best, best_epoch, bad = vl, epoch, 0; model.save(str(ck_best))
            else:
                bad += 1
                if bad >= cfg["patience"]: rc.event(early_stop=True, epoch=epoch); break
        if ck_best.exists(): model.load(str(ck_best))
        table = {}
        with torch.no_grad():
            for nm, items in (("eval_other_series", eo_i), ("eval_wti_2019_2026", ew_i)):
                if not items: continue
                P = []; Y = []
                for batch in batches(items, cfg["batch_size"], shuffle=False, seed=0): P.append(model.probabilities(batch).cpu().numpy()); Y += [b["bin"] for b in batch]
                P = np.concatenate(P); Y = np.array(Y); maj = int(np.argmax(base))
                table[nm] = dict(n=int(len(Y)), top1=float((P.argmax(1) == Y).mean()), majority_baseline=float((Y == maj).mean()), logscore=float(-np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, 1)).mean()), uniform_logscore=float(np.log(K)), output_variation=float(np.abs(P - P.mean(0)).mean()), argmax_counts=np.bincount(P.argmax(1), minlength=K).tolist())
        rc.set(best_epoch=best_epoch, best_val_loss=best, train_seconds=round(time.time() - t0, 1))
        (out / "table.json").write_text(json.dumps(table, indent=1)); md = [f"# recall2 {a.task} / scale={a.scale}: generic windows, no keys in text", "| eval set | n | top-1 | majority | log score | uniform | output variation | argmax counts |", "|---|---|---|---|---|---|---|---|"]
        for nm, c in table.items(): md.append(f"| {nm} | {c['n']} | {c['top1']:.3f} | {c['majority_baseline']:.3f} | {c['logscore']:.3f} | {c['uniform_logscore']:.3f} | {c['output_variation']:.4f} | {c['argmax_counts']} |")
        (out / "table.md").write_text("\n".join(md)); rc.set(status="DONE", ended=time.strftime("%Y-%m-%dT%H:%M:%S%z")); print("\n".join(md)); return 0
    except Exception:
        rc.set(status="FAILED", traceback=traceback.format_exc()); raise
if __name__ == "__main__": sys.exit(main())
