import argparse
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def load_events(path):
    ev = []
    if not os.path.exists(path):
        return ev
    with open(path,'r') as f:
        for line in f:
            try:
                ev.append(json.loads(line))
            except Exception:
                pass
    return ev

def load_labels(path):
    if not os.path.exists(path):
        return pd.DataFrame(columns=['ts_start','ts_end','label'])
    return pd.read_csv(path)

def plot_run(run_dir):
    metrics = pd.read_csv(os.path.join(run_dir,'metrics.csv'))
    system = pd.read_csv(os.path.join(run_dir,'system.csv'))
    labels = load_labels(os.path.join(run_dir,'anomaly_labels.csv'))
    fig, axes = plt.subplots(3,1, figsize=(10,8), sharex=True)
    axes[0].plot(metrics['ts'], metrics['loss'], label='loss')
    axes[0].set_ylabel('loss')
    axes[1].plot(metrics['ts'], metrics['throughput'], label='throughput', color='tab:orange')
    axes[1].set_ylabel('samples/sec')
    axes[2].plot(system['ts'], system['gpu_util'], label='GPU util', color='tab:green')
    axes[2].set_ylabel('gpu util %')
    for _, r in labels.iterrows():
        for ax in axes:
            ax.axvspan(r['ts_start'], r['ts_end'], color='red', alpha=0.15)
    axes[2].set_xlabel('ts')
    fig.suptitle(os.path.basename(run_dir))
    for ax in axes:
        ax.legend()
    plt.tight_layout()
    plt.show()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run-dir', required=True)
    args = ap.parse_args()
    plot_run(args.run_dir)

if __name__ == '__main__':
    main()
