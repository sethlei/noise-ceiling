# What the noise allows: a reading ceiling for time-series language models

Companion code and receipts for the TimeNet Online Hackathon entry "Oil fundamentals on TimeNet" (Kithe Research, November 2026).

A model is shown a 36-point window with some points hidden and asked whether one hidden point is above or below the nearest visible
point to its left. For a random walk the best possible answer is the neighbour rule ("above, if the right neighbour is above the
left"), right with probability `1 - arccos(sqrt(k/(k+m)))/pi`, where the hidden point is `k` steps from its left neighbour and `m`
from its right. Real monthly economic series sit on that number; nothing we trained beats the rule on held-out series; and three
faults in our own benchmark produced scores above it.

## Run it

**No data, no GPU, a few seconds** (numpy only):

    cd harness
    PYTHONPATH=. python leak_demo.py

prints the closed form, the neighbour rule on 20,000 synthetic walks, and a one-line reader that scores 1.000 at the finest spacing
when the window is standardised before the point is hidden, and 0.60 when it is standardised over the visible points only.

**No data, CPU, about a minute** (adds scikit-learn): the ceilings for the synthetic and noise-dose cells, and the scoring of every
language-model run in `results/round3/` against them.

    PYTHONPATH=. python ceiling4.py configs/lateral_llama1b.json --skip-real --runs ../results/round3 --out /tmp/scored.json

**With the dataset** (the real-series cells). Build `sethlei/oil-fundamentals-prices` with TimeNet into `harness/reg/` using your own
free EIA key (the price series are licensed and are fetched under your key, never redistributed):

    EIA_TOKEN=... timenet-build build sethlei/oil-fundamentals-prices --local-registry ./reg
    PYTHONPATH=. python ceiling4.py configs/lateral_llama1b.json --runs ../results/round3 --out /tmp/scored.json
    PYTHONPATH=. python ceiling_cv.py configs/recall_llama1b.json /tmp/cv.json      # all 6,722 windows, cross-validated by series

**The language-model runs** (one H100, about 4.5 minutes each): `round3_all.sh` and `round3b_all.sh`. They need the OpenTSLM source
(`OPENTSLM_SRC`, default `../../OpenTSLM/src`), a Hugging Face token file for the gated Llama base (`HF_TOKEN_FILE`), and the dataset.

## What is here

| path | what |
|---|---|
| `harness/lateral_task.py` | the task: windows, visible sets, labels, scaling, synthetic sources, the twin-series filter |
| `harness/run_lateral3.py` | training arms A to F on OpenTSLM-SoftPrompt (Llama 3.2 1B, LoRA), receipts, per-item predictions |
| `harness/ceiling4.py`, `ceiling_cv.py`, `leak_demo.py` | the ceilings, the scoring, the leak |
| `harness/run.py`, `oilharness/`, `tests/` | the forecast harness the key-free run used; 66 tests |
| `results/round3/` | 27 hidden-point runs and the key-free forecast run: receipt, table and (hidden-point runs) per-item predictions; the batch logs; `ceiling4_scored.json` |
| `results/ceiling_cv_all_windows.json` | the neighbour rule, logistic regression and boosted trees on all real windows |
| `results/earlier/` | the keyed forecast run (2026-11-20) and the direction-reading check (2026-11-19) the page cites |
| `notes/` | the spec with predictions written before the runs, and the results scored against them, including the ones that failed |

## Read before relying on a number

- **One model size, one question shape.** Two training seeds per cell, except the single-spacing arm F (one seed).
- **The receipts' `sha` field is not the code as run.** The code was copied to the GPU machine by rsync, so the recorded commit is
  that machine's stale checkout. The scripts here are the ones that ran, with these edits made afterwards: absolute paths replaced
  by relative ones or environment variables; the task definitions moved from `run_lateral.py` / `run_lateral3.py` into
  `lateral_task.py` so the CPU scripts import no model code (checked: 2,400 items built by the original and the moved code are
  identical, labels, prompts and series); `ceiling4.py` gained `--skip-real` and scores arm F; `ceiling_cv.py` was rewritten on the
  leak-free scaling. After those edits `ceiling4.py` reproduces all 321 previously scored values exactly.
- **Receipts were edited in one way:** an absolute registry path and an absolute output path were replaced by relative ones.
- **The forecast runs' per-item prediction files are withheld** because they carry the licensed reference and target prices.
- The first launch of round 3 was stopped after a reviewer found twin series in training; its partial run is not included or scored.
- `default: --scale window` in `run_lateral3.py` is the LEAKING instrument, kept so earlier runs can be reproduced. Every run in
  `results/round3/` used `--scale visible`.

## Credit

Kithe Research. Seth Leichsenring held the work and is accountable for the publication. The code, runs and notes were written by
Daniel Fable, a Claude model instance (Anthropic), working with him. Cold reviews and literature searches by Claude Opus 5.5 instances.
OpenTSLM is by Langer et al. (arXiv:2510.02410); the encoder and projector weights are theirs (`OpenTSLM/llama-3.2-1b-m4-sp`).

## Licence

MIT (see `LICENSE`) for the code, notes and receipts in this repository. The dataset and its upstream sources carry their own terms (see the data card); OpenTSLM and its weights are under their authors' licences.
