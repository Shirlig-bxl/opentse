#!/usr/bin/env bash
set -e
CU=$(nvidia-smi 2>/dev/null | awk '/CUDA Version/ {print $NF}') || true
EXTRA=""
if [[ -n "$CU" ]]; then
  if [[ "$CU" =~ ^12 ]]; then EXTRA="https://download.pytorch.org/whl/cu121"; fi
  if [[ "$CU" =~ ^11 ]]; then EXTRA="https://download.pytorch.org/whl/cu118"; fi
fi
if [[ -z "$SKIP_INSTALL" ]]; then
  python3 -m pip install --upgrade pip
  if [[ -n "$EXTRA" ]]; then
    python3 -m pip install --extra-index-url "$EXTRA" torch torchvision torchaudio
  else
    python3 -m pip install torch torchvision torchaudio
  fi
  python3 -m pip install numpy pandas matplotlib psutil pynvml scikit-learn scipy
fi
if [[ -z "$RUNS_GLOB" ]]; then RUNS_GLOB="datasets/OpenTSE-Logs/run_*"; fi
if [[ -z "$AE_EPOCHS" ]]; then AE_EPOCHS=20; fi
if [[ -z "$SKIP_TRAIN" ]]; then
  bash scripts/run_experiments.sh
fi
python3 scripts/rebuild_matrix.py --runs-glob "$RUNS_GLOB" --bucket-seconds 10
shopt -s nullglob
RUNS=( $RUNS_GLOB )
for d in "${RUNS[@]}"; do
  if [[ -f "$d/tse_matrix.npy" ]]; then
    python3 scripts/analyze_tse.py --runs "$d" --bucket-seconds 10 --window 12 --min-seg-buckets 6 --fusion stage-aware-v1 --ae-epochs "$AE_EPOCHS"
  else
    echo "Skipping $d (missing tse_matrix.npy)"
  fi
done
python3 - << 'PY'
import os, json, glob, csv
rows=[]
for rp in glob.glob('outputs/*/report.json'):
    r=json.load(open(rp))
    rows.append({'run': r['run'], 'AUPRC': r['AUPRC'], 'LaAP': r['LaAP'], 'mean_delay': r['mean_delay'], 'median_delay': r['median_delay'], 'threshold': r['threshold']})
if not rows:
    print('No reports found under outputs/*/report.json'); raise SystemExit(1)
rows.sort(key=lambda x: x['run'])
os.makedirs('outputs', exist_ok=True)
path='outputs/summary_all_runs.csv'
with open(path,'w',newline='') as f:
    w=csv.DictWriter(f, fieldnames=rows[0].keys())
    w.writeheader(); w.writerows(rows)
print(path)
PY
