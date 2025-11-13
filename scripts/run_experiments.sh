#!/usr/bin/env bash
set -euo pipefail
OUT=${OUT:-datasets/OpenTSE-Logs}
mkdir -p "$OUT"

# Mode & defaults
MODE=${MODE:-safe}             # safe|standard
EPOCHS=${EPOCHS:-1}
REPEATS=${REPEATS:-3}
if [[ "$MODE" == "safe" ]]; then
  NUM_WORKERS=${NUM_WORKERS:-0}
  BATCH_SIZE=${BATCH_SIZE:-64}
  OOM_BATCH_SIZE=${OOM_BATCH_SIZE:-32}
else
  NUM_WORKERS=${NUM_WORKERS:-2}
  BATCH_SIZE=${BATCH_SIZE:-128}
  OOM_BATCH_SIZE=${OOM_BATCH_SIZE:-64}
fi
SYNTHETIC=${SYNTHETIC:-0}

NON_OOM=(clean io_slow grad_explode lr_spike cpu_steal ckpt_fail loss_nan gpu_throttle shuffle_off)
OOM=(oom_vram oom_ram)

echo "Mode=$MODE workers=$NUM_WORKERS batch=$BATCH_SIZE repeats=$REPEATS synthetic=$SYNTHETIC"

for f in "${NON_OOM[@]}"; do
  for r in $(seq 1 "$REPEATS"); do
    if [[ "$SYNTHETIC" == "1" ]]; then SYN_FLAG="--synthetic"; else SYN_FLAG=""; fi
    python3 -u src/train.py --fault "$f" --epochs "$EPOCHS" --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" $SYN_FLAG --out-dir "$OUT"
  done
done

# OOM faults run separately with smaller batch
for f in "${OOM[@]}"; do
  for r in $(seq 1 "$REPEATS"); do
    if [[ "$SYNTHETIC" == "1" ]]; then SYN_FLAG="--synthetic"; else SYN_FLAG=""; fi
    python3 -u src/train.py --fault "$f" --epochs "$EPOCHS" --batch-size "$OOM_BATCH_SIZE" --num-workers "$NUM_WORKERS" $SYN_FLAG --out-dir "$OUT"
  done
done
