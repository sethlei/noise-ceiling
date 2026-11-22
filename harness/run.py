"""One command, one receipt. Spec rev 2 §7–§8.   usage: run.py configs/<name>.json [--device auto]"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

# Hub credential for gated bases: read from a token file inside the process, never from the command line.
HF_TOKEN_FILE = Path(os.environ.get("HF_TOKEN_FILE", "hf_token.txt"))
if "HF_TOKEN" not in os.environ and HF_TOKEN_FILE.exists():
    os.environ["HF_TOKEN"] = HF_TOKEN_FILE.read_text().strip()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from oilharness import aux as auxmod  # noqa: E402
from oilharness.data import DEFAULT_FEATURES, Panel, build_example, find_panel_record, task_rows  # noqa: E402
from oilharness.examples import K_BINS, fit_bins, group_key, save_bins, to_batch_item  # noqa: E402
from oilharness.model import TypedOpenTSLM, check_tokenizers, pick_device  # noqa: E402
from oilharness.train import (hard_nll, calibration, fit_temperature, linear_warmup, predict, run_epoch, score,  # noqa: E402
                              table_md)

BASELINE_N = {("T1", 1): 48, ("T1", 3): 48, ("T1", 6): 48, ("T1", 12): 46, ("T2", 1): 57, ("T2", 3): 55, ("T2", 6): 52, ("T2", 12): 46}


def git_sha(path: str) -> str:
    try:
        return subprocess.check_output(["git", "-C", path, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


class Receipt:
    def __init__(self, path: Path, cfg: dict):
        self.path, self.data = path, {"config": cfg, "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "events": []}
        self.write()

    def write(self):
        self.path.write_text(json.dumps(self.data, indent=1, default=str))

    def set(self, **kv):
        self.data.update(kv); self.write()

    def event(self, **kv):
        self.data["events"].append({"t": time.strftime("%H:%M:%S"), **kv}); self.write()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("config"); ap.add_argument("--device", default="auto")
    a = ap.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    run_id = f"{Path(a.config).stem}_{time.strftime('%Y%m%d_%H%M%S')}"
    out = HERE / "results" / run_id; out.mkdir(parents=True, exist_ok=True)
    rc = Receipt(out / "RECEIPT.json", cfg)
    try:
        return _run(cfg, a, out, rc)
    except Exception:
        rc.set(status="FAILED", traceback=traceback.format_exc(), ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
        raise


def _run(cfg: dict, a, out: Path, rc: Receipt) -> int:
    device = pick_device(a.device)
    torch.manual_seed(cfg.get("seed", 0)); np.random.seed(cfg.get("seed", 0))
    rc.set(status="RUNNING", device=device, platform=platform.platform(), torch=torch.__version__,
           hub_token="from file" if HF_TOKEN_FILE.exists() else "none",
           sha={"box": git_sha(str(HERE.parent)), "timenet": git_sha(os.environ.get("TIMENET_SRC", "../../TimeNet")),
                "opentslm": git_sha(os.environ.get("OPENTSLM_REPO", "../../OpenTSLM"))})
    # §4 tokenizer check for every base named
    labels_by_base = check_tokenizers(sorted(set([cfg["llm_id"]] + cfg.get("bases_to_check", []))))
    rc.set(labels_by_base=labels_by_base)

    # --- data ---
    from timenet.reader import TimeFReader
    features = cfg.get("features", DEFAULT_FEATURES); width = cfg.get("window", 48); info_set = cfg.get("information_set", "per-signal")
    t0 = time.time()
    with TimeFReader.open(cfg["registry_version_dir"]) as r:
        rows = task_rows(r); rec = find_panel_record(r)
        names = [s.name for s in rec.walk_signals()]
        aux_names = cfg.get("aux_signals") or auxmod.default_aux_signals(names) if cfg.get("aux", False) else []
        panel = Panel.from_record(rec, sorted(set(features) | set(aux_names)))
        manifest = (Path(cfg["registry_version_dir"]) / "manifest.json")
        rc.set(registry_manifest_sha256=__import__("hashlib").sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else None)
    examples = [build_example(rw, panel, features, width, info_set) for rw in rows]
    by_split = {s: [e for e in examples if e.split == s] for s in ("train", "validation", "test")}
    n_test = {}
    for e in by_split["test"]:
        n_test[(e.family, e.horizon)] = n_test.get((e.family, e.horizon), 0) + 1
    if n_test != BASELINE_N:
        raise RuntimeError(f"test cells {n_test} != baselines {BASELINE_N}")
    aux_examples = []
    if cfg.get("aux", False):
        aux_examples = auxmod.aux_examples(panel, {n: panel.series[n] for n in aux_names}, features, width,
                                           cfg.get("aux_max"), cfg.get("seed", 0))
    train_wti = by_split["train"]
    if cfg.get("max_train"):
        import random
        random.Random(cfg.get("seed", 0)).shuffle(train_wti); train_wti = train_wti[:cfg["max_train"]]
    rc.set(n_examples={"train_wti": len(train_wti), "aux": len(aux_examples), "validation": len(by_split["validation"]),
                       "test": len(by_split["test"])}, data_seconds=round(time.time() - t0, 1),
           features=features, information_set=info_set, window=width, aux_signals=aux_names)

    # --- bins on training examples only (§3.6, §9) ---
    bins = fit_bins(train_wti + aux_examples, K_BINS)
    # groups that appear only in val/test (none expected) would be missing; guard:
    for e in by_split["validation"] + by_split["test"]:
        if group_key(e) not in bins:
            raise RuntimeError(f"no training bins for group {group_key(e)}")
    save_bins(bins, out / "bins.json")

    # --- model ---
    model = TypedOpenTSLM(cfg["llm_id"], device, lora=cfg.get("lora", True), lora_r=cfg.get("lora_r", 16), lora_alpha=cfg.get("lora_alpha", 32))
    rc.set(labels=model.labels, label_ids=model.label_ids)
    if cfg.get("gradient_checkpointing", True):
        model.enable_gradient_checkpointing()
    if cfg.get("checkpoint_repo"):
        from huggingface_hub import hf_hub_download
        ck_path = hf_hub_download(cfg["checkpoint_repo"], "model_checkpoint.pt")
        ck = torch.load(ck_path, map_location="cpu", weights_only=False)
        model.model.encoder.load_state_dict(ck["encoder_state"])
        loaded = ["encoder"]
        try:
            model.model.projector.load_state_dict(ck["projector_state"]); loaded.append("projector")
        except Exception as e:  # hidden-size mismatch with a different base: keep the fresh projector
            rc.event(note=f"projector not loaded from {cfg['checkpoint_repo']}: {str(e)[:120]}")
        rc.set(checkpoint={"repo": cfg["checkpoint_repo"], "loaded": loaded})
    eos = model.model.get_eos_token()
    import oilharness.examples as _exmod; _exmod.KEY_FREE = bool(cfg.get("key_free", False)); rc.set(key_free=_exmod.KEY_FREE)
    model.ordinal = bool(cfg.get("ordinal", False)); sigma_frac = float(cfg.get("ordinal_sigma_frac", 0.5))
    def items_of(exs):
        return [to_batch_item(e, bins[group_key(e)], model.labels, eos, sigma_frac) for e in exs]
    tr_items = items_of(train_wti + aux_examples); va_items = items_of(by_split["validation"]); te_items = items_of(by_split["test"])
    ent = lambda its: float(np.mean([-np.sum(np.asarray(it["soft"]) * np.log(np.asarray(it["soft"]))) for it in its]))
    wti_tr_items = [it for it in tr_items if not str(it["family"]).startswith("aux")]
    rc.set(ordinal=model.ordinal, ordinal_sigma_frac=sigma_frac, soft_target_entropy_train=round(ent(tr_items), 4), soft_target_entropy_train_wti=round(ent(wti_tr_items), 4),
           n_train_wti_items=len(wti_tr_items),
           gaussian_climatology={g: {"mean": b.train_mean, "sd": b.train_sd} for g, b in bins.items() if not g.startswith("aux")})

    # --- train ---
    text_variant = cfg.get("text_variant", False)
    opt = torch.optim.AdamW(model.param_groups(cfg["lr_encoder"], cfg["lr_projector"], cfg["lr_lora"], cfg["weight_decay"]))
    steps = cfg["epochs"] * max(1, (len(tr_items) + cfg["batch_size"] - 1) // cfg["batch_size"])
    sched = linear_warmup(opt, steps, cfg.get("warmup_frac", 0.1))
    best, best_epoch, bad = float("inf"), -1, 0
    ck_best = out / "best.pt"
    t_train = time.time()
    for epoch in range(cfg["epochs"]):
        te = time.time()
        tr_loss = run_epoch(model, tr_items, cfg, opt, sched, seed=cfg.get("seed", 0) + epoch, text_variant=text_variant)
        va_loss = run_epoch(model, va_items, cfg, text_variant=text_variant)
        va_hard = hard_nll(model, va_items, cfg) if model.ordinal else va_loss          # selection stays on the HARD bin (review amendment 4)
        wti_loss = run_epoch(model, wti_tr_items, cfg, text_variant=text_variant) if model.ordinal else None   # WTI-only train loss, no grad (amendment 3)
        peak = torch.mps.driver_allocated_memory() / 1e9 if device == "mps" else (torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else None)
        rc.event(epoch=epoch, train_loss=round(tr_loss, 4), val_loss=round(va_loss, 4), val_hard_nll=round(va_hard, 4),
                 train_loss_wti=(None if wti_loss is None else round(wti_loss, 4)), seconds=round(time.time() - te, 1), peak_gb=peak)
        if va_hard < best:
            best, best_epoch, bad = va_hard, epoch, 0
            model.save(str(ck_best))
        else:
            bad += 1
            if bad >= cfg["patience"]:
                rc.event(early_stop=True, epoch=epoch); break
    if ck_best.exists():
        model.load(str(ck_best))
    rc.set(best_epoch=best_epoch, best_val_loss=best, train_seconds=round(time.time() - t_train, 1))

    # --- temperature, predictions, scores (§4, §5) ---
    model.temperature = fit_temperature(model, va_items, cfg) if not text_variant else 1.0
    model.save(str(ck_best))
    rc.set(temperature=model.temperature)
    preds_val = predict(model, va_items, cfg); preds_test = predict(model, te_items, cfg)
    table = score(preds_test, bins); table_val = score(preds_val, bins)
    (out / "table.json").write_text(json.dumps({"test": table, "validation": table_val}, indent=1))
    (out / "table.md").write_text(table_md(table, info_set, "baselines_2026-11-11.md"))
    (out / "calibration.json").write_text(json.dumps({"test": calibration(preds_test, K_BINS), "validation": calibration(preds_val, K_BINS)}, indent=1))
    import pandas as pd
    pd.DataFrame(preds_test + preds_val).to_parquet(out / "predictions.parquet", index=False)
    rc.set(status="DONE", table_path=str(out / "table.md"), ended=time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    print((out / "table.md").read_text())
    return 0


if __name__ == "__main__":
    sys.exit(main())
