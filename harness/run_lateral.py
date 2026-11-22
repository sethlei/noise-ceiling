"""Lateral curriculum (Seth, 2026-11-19 evening): multi-scale masked interpolation as recall training.
36-point windows. Stage 1: show the 6 anchors (every 6th point); ask where a hidden point between two anchors sits relative to
their average. Stage 2: show 18 points (anchors + their immediate neighbours); ask about the remaining hidden points from the two
visible neighbours. Stage 3: show all but the asked point. Same LoRA, stages in order. Question (K=2): A: below the average of
the two visible neighbours; B: above. Nothing is a forecast target; everything is inside the window.
Data: --source synth (random walks + sines + ramps, fresh) or --source real (generic 36-month windows from any panel series,
no keys, centered per window, scale = typical window spread). Eval per stage on held-out items; receipts.
usage: run_lateral.py configs/recall_llama1b.json --source synth|real [--n-train 3000] [--device cuda] [--limit N]"""
from __future__ import annotations
import argparse, json, os, platform, sys, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
HF_TOKEN_FILE = Path(os.environ.get("HF_TOKEN_FILE", "hf_token.txt"))
if "HF_TOKEN" not in os.environ and HF_TOKEN_FILE.exists(): os.environ["HF_TOKEN"] = HF_TOKEN_FILE.read_text().strip()
import numpy as np, torch
from oilharness.model import TypedOpenTSLM, pick_device, single_token_labels
from oilharness.train import batches, linear_warmup, run_epoch
from run import Receipt, git_sha
import lateral_task
from lateral_task import L, ANCH, QPOS, Q, STAGES, visible_sets, gen_synth, gen_dose, make_items
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--source", choices=["synth", "real"], default="synth"); ap.add_argument("--device", default="auto")
    ap.add_argument("--n-train", type=int, default=1500); ap.add_argument("--n-eval", type=int, default=300); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()); K = 2
    run_id = f"lateral2_{a.source}_{Path(a.config).stem}_{time.strftime('%Y%m%d_%H%M%S')}"; out = HERE / "results" / run_id; out.mkdir(parents=True, exist_ok=True)
    rc = Receipt(out / "RECEIPT.json", {**cfg, "lateral_source": a.source, "K": K, "n_train": a.n_train, "anchors": ANCH})
    try:
        device = pick_device(a.device); torch.manual_seed(0); rng = np.random.default_rng(0)
        rc.set(status="RUNNING", device=device, platform=platform.platform(), torch=torch.__version__, sha={"box": git_sha(str(HERE.parent))})
        if a.source == "synth":
            n_tr, n_ev = (a.limit, a.limit) if a.limit else (a.n_train, a.n_eval); tr_seqs = [gen_synth(rng) for _ in range(n_tr)]; ev_seqs = [gen_synth(rng) for _ in range(n_ev)]
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
            rng.shuffle(seqs); wti = [w for n, m, w in seqs if n == "wti_monthly_mean"]; others = [w for n, m, w in seqs if n != "wti_monthly_mean"]
            n_tr, n_ev = (a.limit, a.limit) if a.limit else (a.n_train, a.n_eval); tr_seqs = others[:n_tr]; ev_seqs = others[n_tr:n_tr + n_ev] + wti
            rc.set(n_wti_eval=len(wti))
        model = TypedOpenTSLM(cfg["llm_id"], device, lora=cfg.get("lora", True), lora_r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32))
        model.labels, model.label_ids = single_token_labels(model.model.tokenizer, K); model.label_ids_t = torch.as_tensor(model.label_ids, device=device)
        if cfg.get("gradient_checkpointing", True): model.enable_gradient_checkpointing()
        if cfg.get("checkpoint_repo"):
            from huggingface_hub import hf_hub_download
            ck = torch.load(hf_hub_download(cfg["checkpoint_repo"], "model_checkpoint.pt"), map_location="cpu", weights_only=False)
            model.model.encoder.load_state_dict(ck["encoder_state"]); model.model.projector.load_state_dict(ck["projector_state"])
        eos = model.model.get_eos_token()
        # ZERO-SHOT-ish baseline: evaluate each stage before any training
        def evaluate(items):
            P = []; Y = []
            with torch.no_grad():
                for batch in batches(items, cfg["batch_size"], shuffle=False, seed=0): P.append(model.probabilities(batch).cpu().numpy()); Y += [b["bin"] for b in batch]
            P = np.concatenate(P); Y = np.array(Y); return dict(n=int(len(Y)), top1=float((P.argmax(1) == Y).mean()), majority=float(max(Y.mean(), 1 - Y.mean())), logscore=float(-np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, 1)).mean()), output_variation=float(np.abs(P - P.mean(0)).mean()))
        ev_items = [make_items(ev_seqs, st, model.labels, eos, np.random.default_rng(100 + st)) for st in range(3)]
        rc.set(before_training={f"stage{st+1}": evaluate(ev_items[st]) for st in range(3)})
        opt = torch.optim.AdamW(model.param_groups(cfg["lr_encoder"], cfg["lr_projector"], cfg["lr_lora"], cfg["weight_decay"]))
        epochs_per_stage = max(1, cfg["epochs"] // 3); table = {}; t0 = time.time()
        for st in range(3):
            tr_items = make_items(tr_seqs, st, model.labels, eos, np.random.default_rng(st)); va_items = make_items(ev_seqs[: max(20, len(ev_seqs) // 4)], st, model.labels, eos, np.random.default_rng(50 + st))
            steps = epochs_per_stage * max(1, (len(tr_items) + cfg["batch_size"] - 1) // cfg["batch_size"]); sched = linear_warmup(opt, steps, cfg.get("warmup_frac", 0.1))
            for ep in range(epochs_per_stage):
                te = time.time(); tl = run_epoch(model, tr_items, cfg, opt, sched, seed=10 * st + ep); vl = run_epoch(model, va_items, cfg)
                rc.event(stage=st + 1, epoch=ep, train_loss=round(tl, 4), val_loss=round(vl, 4), seconds=round(time.time() - te, 1))
            table[f"after_stage{st+1}"] = {f"stage{s2+1}": evaluate(ev_items[s2]) for s2 in range(3)}; rc.set(table=table)
        model.save(str(out / "best.pt")); rc.set(status="DONE", train_seconds=round(time.time() - t0, 1), ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        md = [f"# lateral curriculum v2 (fixed asked position, label = above/below the left visible neighbour) / {a.source}: top-1 on held-out items per eval stage (majority ≈ 0.5); rows = training progress", "| trained through | eval stage 1 (anchors) | eval stage 2 (18 visible) | eval stage 3 (all but one) |", "|---|---|---|---|"]
        bt = rc.data["before_training"]; md.append(f"| nothing | {bt['stage1']['top1']:.3f} | {bt['stage2']['top1']:.3f} | {bt['stage3']['top1']:.3f} |")
        for k, v in table.items(): md.append(f"| {k} | {v['stage1']['top1']:.3f} | {v['stage2']['top1']:.3f} | {v['stage3']['top1']:.3f} |")
        (out / "table.md").write_text("\n".join(md)); print("\n".join(md)); return 0
    except Exception:
        rc.set(status="FAILED", traceback=traceback.format_exc()); raise
if __name__ == "__main__": sys.exit(main())
