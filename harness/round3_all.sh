#!/usr/bin/env bash
# Round 3 (spec notes/ROUND3_SPEC_2026-11-22.md): leak-free lateral on real (held-out series) and synth, the noise dose, the key-free forecast.
# Order: the cells the page needs first. Each run writes its own receipt; a FAILED line here is the loud form.
set -uo pipefail; cd "$(dirname "$0")"; mkdir -p logs results
PY=${PY:-python3}; CFG=configs/lateral_llama1b.json
[ -f "$CFG" ] || sed 's/"epochs": 8/"epochs": 9/' configs/recall_llama1b.json > "$CFG"
FAILS=0; F='warning\|redirects\|Loading weights\|torch_dtype\|Loaded'
lat() { echo "=== $(date '+%F %T') lateral4 $*"; PYTHONPATH=. "$PY" run_lateral3.py "$CFG" --scale visible --device cuda "$@" 2>&1 | grep -v -i "$F" | tail -7; st=${PIPESTATUS[0]}; [ "$st" -eq 0 ] || { echo "!!! FAILED lateral4 $* exit $st"; FAILS=$((FAILS+1)); }; }
for ARM in E C B A D; do lat --arm "$ARM" --source real --seed 0 --n-eval 600; done
for D in 1.0 0.5 0.0 0.25 0.75 0.1; do lat --arm C --source dose --dose "$D" --seed 0 --n-eval 600; done
for ARM in E C B A D; do lat --arm "$ARM" --source real --seed 1 --n-eval 600; done
for ARM in C E; do for SEED in 0 1; do lat --arm "$ARM" --source synth --seed "$SEED"; done; done
for D in 1.0 0.5; do lat --arm C --source dose --dose "$D" --seed 1 --n-eval 600; done
echo "=== $(date '+%F %T') keyfree A2"
PYTHONPATH=. "$PY" run.py configs/run_llama1b_ordinal_wti_keyfree.json --device cuda 2>&1 | grep -v -i "$F" | tail -14; st=${PIPESTATUS[0]}; [ "$st" -eq 0 ] || { echo "!!! FAILED keyfree exit $st"; FAILS=$((FAILS+1)); }
echo "$(date '+%F %T') fails=$FAILS" > results/ROUND3_DONE; echo "=== done $(date '+%F %T') fails=$FAILS"
