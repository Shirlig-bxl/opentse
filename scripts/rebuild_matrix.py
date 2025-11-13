import argparse
import os
import sys
import pathlib

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from src.tse.build_matrix import build_tse_matrix

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs-glob', type=str, default='datasets/OpenTSE-Logs/run_*')
    p.add_argument('--bucket-seconds', type=int, default=10)
    p.add_argument('--force', action='store_true')
    args = p.parse_args()
    import glob
    runs = sorted(glob.glob(args.runs_glob))
    for d in runs:
        out_npy = os.path.join(d, 'tse_matrix.npy')
        if os.path.exists(out_npy) and not args.force:
            print(f'Skip {d} (exists)')
            continue
        try:
            build_tse_matrix(d, args.bucket_seconds)
            print(f'Rebuilt {d}')
        except Exception as e:
            print(f'Failed {d}: {e}')

if __name__ == '__main__':
    main()

