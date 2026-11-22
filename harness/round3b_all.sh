#!/usr/bin/env bash
# Round 3b: is the middle-scale gap a data limit or the reader's? Arm F (S2 only), 1,500 vs 3x-4x training sequences, pure walk and real.
set -uo pipefail; cd "$(dirname "$0")"; mkdir -p logs results
PY=${PY:-python3}; CFG=configs/lateral_llama1b.json; FAILS=0; F='warning\|redirects\|Loading weights\|torch_dtype\|Loaded'
lat() { echo "=== $(date '+%F %T') lateral4 $*"; PYTHONPATH=. "$PY" run_lateral3.py "$CFG" --scale visible --device cuda "$@" 2>&1 | grep -v -i "$F" | tail -7; st=${PIPESTATUS[0]}; [ "$st" -eq 0 ] || { echo "!!! FAILED lateral4 $* exit $st"; FAILS=$((FAILS+1)); }; }
lat --arm F --source dose --dose 1.0 --seed 0 --n-eval 600
lat --arm F --source real --seed 0 --n-eval 600
lat --arm F --source dose --dose 1.0 --seed 0 --n-eval 600 --n-train 6000
lat --arm F --source real --seed 0 --n-eval 600 --n-train 4500
echo "$(date '+%F %T') fails=$FAILS" > results/ROUND3B_DONE; echo "=== done $(date '+%F %T') fails=$FAILS"
