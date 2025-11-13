#!/usr/bin/env bash
set -euo pipefail
OUT=datasets/OpenTSE-Logs
mkdir -p "$OUT"
FAULTS=(clean io_slow grad_explode lr_spike cpu_steal ckpt_fail loss_nan gpu_throttle shuffle_off oom_vram oom_ram)
for f in "${FAULTS[@]}"; do
  for r in 1 2 3; do
    python3 -u src/train.py --fault "$f" --epochs 1 --out-dir "$OUT"
  done
done
