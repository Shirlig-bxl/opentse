import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def _ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _load_series(run_dir, metric):
    sp = os.path.join('outputs', os.path.basename(run_dir.rstrip('/')), 'scores.csv')
    if os.path.exists(sp):
        df = pd.read_csv(sp)
        if metric in ('Sfinal','S1','S2','S3'):
            return df['ts'].values, df[metric].values
        else:
            return df['ts'].values, df['Sfinal'].values
    mp = os.path.join(run_dir, 'metrics.csv')
    sp2 = os.path.join(run_dir, 'system.csv')
    if metric in ('loss','lr','grad_norm','throughput') and os.path.exists(mp):
        df = pd.read_csv(mp)
        return df['ts'].values, df[metric].values
    if metric in ('gpu_util','gpu_mem','cpu_util','ram_used') and os.path.exists(sp2):
        df = pd.read_csv(sp2)
        return df['ts'].values, df[metric].values
    return None, None

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--metric', type=str, default='Sfinal')
    args = p.parse_args()
    name = 'compare_' + '_vs_'.join([os.path.basename(r.rstrip('/')) for r in args.runs])
    out_dir = os.path.join('outputs', name)
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    rows = []
    for r in args.runs:
        ts, ys = _load_series(r, args.metric)
        if ts is None:
            continue
        lab = os.path.basename(r.rstrip('/'))
        ax.plot(ts, ys, label=lab)
        rows.append({'run': lab, 'min': float(np.min(ys)), 'max': float(np.max(ys)), 'mean': float(np.mean(ys))})
    ax.set_title(args.metric)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{name}_{args.metric}.png'))
    plt.close(fig)
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, f'{name}_summary.csv'), index=False)
    print(os.path.join(out_dir, f'{name}_{args.metric}.png'))

if __name__ == '__main__':
    main()

