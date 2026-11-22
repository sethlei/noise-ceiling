"""Curriculum-structure arms over the lateral task (spec: an earlier internal spec, not published). Task, data, model and recipe are
imported from lateral_task.py and run_lateral.py unchanged; only the composition of each 3-epoch phase differs by arm:
  A staged coarse->fine (S1/S2/S3; seed 0 reproduces v2)   B staged + replay (S1 / 2:1 S2:S1 / 2:1:1 S3:S2:S1)
  C mixed, no curriculum (each phase: every sequence at a uniformly random stage)   D staged fine->coarse (S3/S2/S1)   E fine only (S3/S3/S3)
Data rng is fixed at 0 (same sequences and split in every arm); --seed varies torch init, batch order and the arms' stage draws.
usage: run_lateral3.py configs/lateral_llama1b.json --arm A|B|C|D|E --source synth|real [--seed 0] [--device cuda] [--limit N]"""
from __future__ import annotations
import argparse, json, platform, sys, time, traceback
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import numpy as np, torch
import lateral_task
from lateral_task import L, ANCH, make_items, gen_synth, gen_dose, twin_series
from oilharness.model import TypedOpenTSLM, pick_device, single_token_labels
from oilharness.train import batches, linear_warmup, run_epoch
from run import Receipt, git_sha
ARMS = {"A": [[(0, 1.0)], [(1, 1.0)], [(2, 1.0)]],
        "B": [[(0, 1.0)], [(1, 2 / 3), (0, 1 / 3)], [(2, 0.5), (1, 0.25), (0, 0.25)]],
        "C": [None, None, None],
        "D": [[(2, 1.0)], [(1, 1.0)], [(0, 1.0)]],
        "E": [[(2, 1.0)], [(2, 1.0)], [(2, 1.0)]],
        "F": [[(1, 1.0)], [(1, 1.0)], [(1, 1.0)]]}     # middle scale only (2026-11-22, the S2-gap follow-up)
def phase_items(arm, phase, tr_seqs, labels, eos, rng):
    """Items for one phase: exactly len(tr_seqs) items; every sequence appears exactly once, at one stage (a partition).
    rng = the training-seed rng for this phase."""
    N = len(tr_seqs); comp = ARMS[arm][phase]
    if comp is None:                                   # C: every sequence once, at a uniformly random stage
        st = rng.integers(0, 3, N); groups = {s: [tr_seqs[i] for i in range(N) if st[i] == s] for s in range(3)}
    else:
        sizes = [int(round(f * N)) for _, f in comp]; sizes[0] += N - sum(sizes)   # the first (main) stage absorbs rounding
        perm = rng.permutation(N) if len(comp) > 1 else np.arange(N); groups = {}; at = 0
        for (s, _), k in zip(comp, sizes): groups[s] = [tr_seqs[i] for i in perm[at:at + k]]; at += k
    items = []
    for s, seqs in groups.items():
        if seqs: items += make_items(seqs, s, labels, eos, np.random.default_rng(s))
    assert len(items) == N, (len(items), N)
    return items, {f"stage{s+1}": len(seqs) for s, seqs in groups.items()}
def main():
    ap = argparse.ArgumentParser(); ap.add_argument("config"); ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--source", choices=["synth", "real", "dose"], default="synth"); ap.add_argument("--dose", type=float, default=None); ap.add_argument("--scale", choices=["window", "visible"], default="window"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--device", default="auto")
    ap.add_argument("--n-train", type=int, default=1500); ap.add_argument("--n-eval", type=int, default=300); ap.add_argument("--limit", type=int, default=0); a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text()); K = 2
    lateral_task.SCALE_MODE = a.scale; assert (a.source == "dose") == (a.dose is not None), "--dose goes with --source dose, and only with it"
    src_tag = a.source if a.source != "dose" else f"dose{int(round(100 * a.dose)):03d}"; ver = "lateral3" if a.scale == "window" else "lateral4"
    if a.n_train != 1500: src_tag += f"_n{a.n_train}"
    run_id = f"{ver}_{a.arm}_{src_tag}_s{a.seed}_{time.strftime('%Y%m%d_%H%M%S')}"; out = HERE / "results" / run_id; out.mkdir(parents=True, exist_ok=True)
    rc = Receipt(out / "RECEIPT.json", {**cfg, "arm": a.arm, "arm_phases": ARMS[a.arm], "lateral_source": a.source, "dose": a.dose, "scale_mode": a.scale, "train_seed": a.seed, "data_seed": 0, "K": K, "n_train": a.n_train, "anchors": ANCH, "spec": "an earlier internal spec, not published"})
    try:
        device = pick_device(a.device); torch.manual_seed(a.seed); rng = np.random.default_rng(0)      # DATA rng: fixed, as v2
        rc.set(status="RUNNING", device=device, platform=platform.platform(), torch=torch.__version__, sha={"box": git_sha(str(HERE.parent))})
        if a.source == "synth":
            n_tr, n_ev = (a.limit, a.limit) if a.limit else (a.n_train, a.n_eval); tr_seqs = [gen_synth(rng) for _ in range(n_tr)]; ev_seqs = [gen_synth(rng) for _ in range(n_ev)]
        elif a.source == "dose":
            n_tr, n_ev = (a.limit, a.limit) if a.limit else (a.n_train, a.n_eval); tr_seqs = [gen_dose(rng, a.dose) for _ in range(n_tr)]; ev_seqs = [gen_dose(rng, a.dose) for _ in range(n_ev)]
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
            n_tr, n_ev = (a.limit, a.limit) if a.limit else (a.n_train, a.n_eval)
            if a.scale == "window":       # v3 split as run: windows shuffled, the eval windows' series are also in training
                others = [w for n, m, w in seqs if n != "wti_monthly_mean"]; tr_seqs = others[:n_tr]; ev_seqs = others[n_tr:n_tr + n_ev] + wti
            else:                         # lateral4: eval series are HELD OUT of training (16 series, fixed draw, rng 7), as recall v2 did,
                                          # and so is every TWIN of an eval series or of WTI (cold review 2026-11-22: eia_RWTC is WTI value for value)
                names_o = sorted({n for n, m, w in seqs if n != "wti_monthly_mean"}); held = set(np.random.default_rng(7).choice(names_o, 16, replace=False).tolist())
                twins = twin_series(panel, held | {"wti_monthly_mean"})
                tr_seqs = [w for n, m, w in seqs if n != "wti_monthly_mean" and n not in held and n not in twins][:n_tr]; ev_seqs = [w for n, m, w in seqs if n in held][:n_ev] + wti
                rc.set(heldout_series=sorted(held), twins_excluded_from_training=twins, n_eval_heldout=len(ev_seqs) - len(wti))
            rc.set(n_wti_eval=len(wti))
        model = TypedOpenTSLM(cfg["llm_id"], device, lora=cfg.get("lora", True), lora_r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32))
        model.labels, model.label_ids = single_token_labels(model.model.tokenizer, K); model.label_ids_t = torch.as_tensor(model.label_ids, device=device)
        if cfg.get("gradient_checkpointing", True): model.enable_gradient_checkpointing()
        if cfg.get("checkpoint_repo"):
            from huggingface_hub import hf_hub_download
            ck = torch.load(hf_hub_download(cfg["checkpoint_repo"], "model_checkpoint.pt"), map_location="cpu", weights_only=False)
            model.model.encoder.load_state_dict(ck["encoder_state"]); model.model.projector.load_state_dict(ck["projector_state"])
        eos = model.model.get_eos_token()
        preds = {}
        def evaluate(items, tag=None):
            P = []; Y = []
            with torch.no_grad():
                for batch in batches(items, cfg["batch_size"], shuffle=False, seed=0): P.append(model.probabilities(batch).cpu().numpy()); Y += [b["bin"] for b in batch]
            P = np.concatenate(P); Y = np.array(Y)
            if tag: preds[tag] = P[:, 1].astype(np.float32); preds[tag + "_y"] = Y.astype(np.int8); np.savez_compressed(out / "eval_predictions.npz", **preds)
            return dict(n=int(len(Y)), top1=float((P.argmax(1) == Y).mean()), majority=float(max(Y.mean(), 1 - Y.mean())), logscore=float(-np.log(np.clip(P[np.arange(len(Y)), Y], 1e-12, 1)).mean()), output_variation=float(np.abs(P - P.mean(0)).mean()))
        ev_items = [make_items(ev_seqs, st, model.labels, eos, np.random.default_rng(100 + st)) for st in range(3)]      # identical across arms
        rc.set(before_training={f"stage{st+1}": evaluate(ev_items[st], f"before_stage{st+1}") for st in range(3)})
        opt = torch.optim.AdamW(model.param_groups(cfg["lr_encoder"], cfg["lr_projector"], cfg["lr_lora"], cfg["weight_decay"]))
        assert cfg["epochs"] == 9, f"spec pins 9 epochs = 3 per phase (v2 receipt); config has {cfg['epochs']}"
        epochs_per_phase = cfg["epochs"] // 3; rc.set(epochs_per_phase=epochs_per_phase); table = {}; comps = {}; t0 = time.time()
        for ph in range(3):
            tr_items, comp = phase_items(a.arm, ph, tr_seqs, model.labels, eos, np.random.default_rng(1000 + 7 * ph + a.seed)); comps[f"phase{ph+1}"] = comp
            va_items = [it for st in range(3) for it in make_items(ev_seqs[: max(20, len(ev_seqs) // 12)], st, model.labels, eos, np.random.default_rng(50 + st))]   # small all-scale val set, monitoring only
            steps = epochs_per_phase * max(1, (len(tr_items) + cfg["batch_size"] - 1) // cfg["batch_size"]); sched = linear_warmup(opt, steps, cfg.get("warmup_frac", 0.1))
            for ep in range(epochs_per_phase):
                te = time.time(); tl = run_epoch(model, tr_items, cfg, opt, sched, seed=100 * a.seed + 10 * ph + ep); vl = run_epoch(model, va_items, cfg)
                rc.event(phase=ph + 1, epoch=ep, train_loss=round(tl, 4), val_loss=round(vl, 4), seconds=round(time.time() - te, 1))
            table[f"after_phase{ph+1}"] = {f"stage{s2+1}": evaluate(ev_items[s2], f"after_phase{ph+1}_stage{s2+1}") for s2 in range(3)}
            passes = {f"stage{s2+1}": epochs_per_phase * sum(c.get(f"stage{s2+1}", 0) for c in comps.values()) for s2 in range(3)}
            rc.set(table=table, phase_composition=comps, item_passes_per_scale_so_far=passes)
        rc.set(status="DONE", train_seconds=round(time.time() - t0, 1), ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        md = [f"# curriculum arm {a.arm} / {a.source} / train seed {a.seed}: held-out top-1 per eval scale (majority ≈ 0.5); rows = training progress; phases = {ARMS[a.arm]}",
              "| trained through | eval stage 1 (anchors) | eval stage 2 (18 visible) | eval stage 3 (all but one) |", "|---|---|---|---|"]
        bt = rc.data["before_training"]; md.append(f"| nothing | {bt['stage1']['top1']:.3f} | {bt['stage2']['top1']:.3f} | {bt['stage3']['top1']:.3f} |")
        for k, v in table.items(): md.append(f"| {k} | {v['stage1']['top1']:.3f} | {v['stage2']['top1']:.3f} | {v['stage3']['top1']:.3f} |")
        (out / "table.md").write_text("\n".join(md)); print("\n".join(md)); return 0
    except Exception:
        rc.set(status="FAILED", traceback=traceback.format_exc()); raise
if __name__ == "__main__": sys.exit(main())
