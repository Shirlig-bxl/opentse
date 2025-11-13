import argparse
import os
import json
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

def _ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _load_run(run_dir):
    mat = np.load(os.path.join(run_dir, 'tse_matrix.npy'))
    fmap = json.load(open(os.path.join(run_dir, 'feature_map.json'))) if os.path.exists(os.path.join(run_dir, 'feature_map.json')) else {}
    labels = pd.read_csv(os.path.join(run_dir, 'anomaly_labels.csv')) if os.path.exists(os.path.join(run_dir, 'anomaly_labels.csv')) else pd.DataFrame(columns=['ts_start','ts_end','label'])
    stages = pd.read_csv(os.path.join(run_dir, 'stage.csv')) if os.path.exists(os.path.join(run_dir, 'stage.csv')) else pd.DataFrame(columns=['ts','stage'])
    inv = [None]*len(fmap) if fmap else []
    if fmap:
        for k,v in fmap.items():
            if v < len(inv):
                inv[v] = k
    cols = inv if inv else [str(i) for i in range(mat.shape[1])]
    df = pd.DataFrame(mat, columns=cols)
    if 'ts_x' in df.columns and 'ts_y' in df.columns:
        df['ts'] = df[['ts_x','ts_y']].min(axis=1)
    elif 'ts_x' in df.columns:
        df['ts'] = df['ts_x']
    elif 'ts_y' in df.columns:
        df['ts'] = df['ts_y']
    else:
        base = pd.read_csv(os.path.join(run_dir, 'metrics.csv')) if os.path.exists(os.path.join(run_dir, 'metrics.csv')) else None
        if base is not None and 'ts' in base.columns:
            bt = base['ts'].values
            df['ts'] = pd.Series(bt[:len(df)])
        else:
            now = time.time()
            df['ts'] = pd.Series(np.arange(len(df))) + now
    return df, labels, stages

def _bucket_index(ts, bucket_seconds):
    return (ts // bucket_seconds).astype(int)

def _assign_stage(df_ts, stages, bucket_seconds):
    if stages is None or len(stages) == 0:
        return pd.Series(['train']*len(df_ts))
    s = stages.sort_values('ts')
    arr = []
    j = 0
    for t in df_ts:
        while j+1 < len(s) and s.iloc[j+1]['ts'] <= t:
            j += 1
        arr.append(str(s.iloc[j]['stage']))
    return pd.Series(arr)

def _zscore_stage(df_feat, stages):
    vals = df_feat.values.astype(float)
    st = stages.values
    uniq = np.unique(st)
    out = np.zeros_like(vals, dtype=float)
    for u in uniq:
        idx = np.where(st == u)[0]
        if len(idx) == 0:
            continue
        sub = vals[idx]
        mu = np.nanmean(sub, axis=0)
        sd = np.nanstd(sub, axis=0)
        sd[sd == 0] = 1.0
        out[idx] = (sub - mu) / sd
    s1 = np.nanmean(np.abs(out), axis=1)
    return s1

def _try_iforest(X):
    try:
        from sklearn.ensemble import IsolationForest
        model = IsolationForest(n_estimators=200, max_features=1.0, random_state=42)
        s2 = -model.fit_predict(X)
        s2 = (s2 - s2.min()) / (s2.max() - s2.min() + 1e-12)
        return s2
    except Exception:
        med = np.nanmedian(X, axis=0)
        mad = np.nanmedian(np.abs(X - med), axis=0)
        mad[mad == 0] = 1.0
        rz = np.nanmean(np.abs((X - med) / mad), axis=1)
        rz = (rz - np.nanmin(rz)) / (np.nanmax(rz) - np.nanmin(rz) + 1e-12)
        return rz

def _ae_scores(X, window, epochs):
    import torch
    import torch.nn as nn
    import torch.optim as optim
    F = X.shape[1]
    class AE(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(F, int(F*1.5))
            self.l2 = nn.Linear(int(F*1.5), F)
            self.l3 = nn.Linear(F, int(F*1.5))
            self.l4 = nn.Linear(int(F*1.5), F)
            self.d = nn.Dropout(0.1)
            self.a = nn.ReLU()
        def forward(self, x):
            x = self.a(self.l1(x))
            x = self.d(self.l2(x))
            x = self.a(self.l3(x))
            x = self.l4(x)
            return x
    seqs = []
    for i in range(0, len(X)-window+1):
        seqs.append(X[i:i+window])
    if len(seqs) == 0:
        return np.zeros(len(X))
    data = torch.tensor(np.stack(seqs), dtype=torch.float32)
    model = AE()
    opt = optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    bs = 64
    best = None
    noimp = 0
    last = 1e9
    for ep in range(epochs):
        perm = torch.randperm(data.size(0))
        tot = 0.0
        for i in range(0, data.size(0), bs):
            idx = perm[i:i+bs]
            x = data[idx].reshape(-1, F)
            opt.zero_grad()
            y = model(x)
            l = loss_fn(y, x)
            l.backward()
            opt.step()
            tot += float(l.detach().cpu())
        avg = tot / max(1, math.ceil(data.size(0)/bs))
        if avg < last - 1e-6:
            last = avg
            best = {k: v.clone() for k, v in model.state_dict().items()}
            noimp = 0
        else:
            noimp += 1
        if noimp >= 5:
            break
    if best is not None:
        model.load_state_dict(best)
    with torch.no_grad():
        recon = model(data.reshape(-1, F)).reshape(-1, window, F)
        err = torch.mean((recon - data)**2, dim=(1,2)).cpu().numpy()
    s3 = np.zeros(len(X))
    for i in range(len(err)):
        j = i + window//2
        if j < len(s3):
            s3[j] = err[i]
    s3 = (s3 - np.nanmin(s3)) / (np.nanmax(s3) - np.nanmin(s3) + 1e-12)
    return s3

def _fusion(s1, s2, s3, stages):
    out = np.zeros_like(s1)
    st = stages.values
    for i in range(len(out)):
        si = st[i]
        if si in ('forward_start','forward_end','backward_start','backward_end','step_start','step_end'):
            w1, w2, w3 = 0.2, 0.3, 0.5
        else:
            w1, w2, w3 = 0.5, 0.3, 0.2
        out[i] = w1*s1[i] + w2*s2[i] + w3*s3[i]
    return out

def _gamma_threshold(scores):
    try:
        import scipy.stats as st
        s = np.array(scores)
        s = s[np.isfinite(s)]
        if len(s) < 10:
            thr = np.quantile(scores, 0.99)
            return thr, None
        mu = np.mean(s)
        v = np.var(s)
        k = mu*mu/(v+1e-12)
        theta = v/(mu+1e-12)
        thr = st.gamma.ppf(0.99, a=k, scale=theta)
        pv = 0.01
        return float(thr), pv
    except Exception:
        thr = np.quantile(scores, 0.99)
        return float(thr), None

def _label_vector(ts, labels):
    y = np.zeros(len(ts), dtype=int)
    for _, r in labels.iterrows():
        s = float(r['ts_start'])
        e = float(r['ts_end'])
        idx = np.where((ts >= s) & (ts <= e))[0]
        y[idx] = 1
    return y

def _metrics(y_true, y_score, ts, delta_buckets, bucket_seconds):
    try:
        from sklearn.metrics import average_precision_score
        auprc = float(average_precision_score(y_true, y_score))
    except Exception:
        thr = np.linspace(np.min(y_score), np.max(y_score), num=50)
        ps = []
        rs = []
        for t in thr:
            yp = (y_score >= t).astype(int)
            tp = int(np.sum((yp == 1) & (y_true == 1)))
            fp = int(np.sum((yp == 1) & (y_true == 0)))
            fn = int(np.sum((yp == 0) & (y_true == 1)))
            p = tp / (tp + fp + 1e-12)
            r = tp / (tp + fn + 1e-12)
            ps.append(p)
            rs.append(r)
        auprc = float(np.trapz(ps, rs))
    det_idx = np.where(y_score >= np.quantile(y_score, 0.99))[0]
    det_ts = ts[det_idx] if len(det_idx) > 0 else np.array([])
    delays = []
    for t in det_ts:
        cand = []
        for i in range(len(ts)):
            if y_true[i] == 1:
                if abs(ts[i] - t) <= (delta_buckets*bucket_seconds):
                    cand.append(abs(ts[i] - t))
        if len(cand) > 0:
            delays.append(min(cand))
    if len(delays) == 0:
        mdel = None
        meddel = None
    else:
        mdel = float(np.mean(delays))
        meddel = float(np.median(delays))
    laap = float(auprc) if mdel is None else float(max(0.0, auprc - (mdel/(delta_buckets*bucket_seconds+1e-12))*0.1))
    return auprc, laap, mdel, meddel

def _plot(run_name, out_dir, ts, s1, s2, s3, sf, labels, stages):
    _ensure_dir(out_dir)
    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(ts, sf, label='Sfinal')
    ax.plot(ts, s1, label='S1', alpha=0.5)
    ax.plot(ts, s2, label='S2', alpha=0.5)
    ax.plot(ts, s3, label='S3', alpha=0.5)
    for _, r in labels.iterrows():
        ax.axvspan(float(r['ts_start']), float(r['ts_end']), color='red', alpha=0.15)
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, f'{run_name}_scores.png'))
    plt.close(fig)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--runs', nargs='+', required=True)
    p.add_argument('--bucket-seconds', type=int, default=10)
    p.add_argument('--window', type=int, default=12)
    p.add_argument('--min-seg-buckets', type=int, default=6)
    p.add_argument('--fusion', type=str, default='stage-aware-v1')
    p.add_argument('--ae-epochs', type=int, default=50)
    args = p.parse_args()
    for run_dir in args.runs:
        name = os.path.basename(run_dir.rstrip('/'))
        out_dir = os.path.join('outputs', name)
        _ensure_dir(out_dir)
        df, labels, stages = _load_run(run_dir)
        drop_cols = [c for c in df.columns if c.startswith('ts') or c in ('stage',)]
        feats = df.drop(columns=drop_cols, errors='ignore')
        ts = df['ts'].values
        stage_series = _assign_stage(ts, stages, args.bucket_seconds)
        mu = np.nanmean(feats.values, axis=0)
        sd = np.nanstd(feats.values, axis=0)
        sd[sd == 0] = 1.0
        X = (feats.values - mu) / sd
        s1 = _zscore_stage(feats, stage_series)
        s2 = _try_iforest(X)
        s3 = _ae_scores(X, args.window, args.ae_epochs)
        sf = _fusion(s1, s2, s3, stage_series)
        thr, pv = _gamma_threshold(sf)
        y = _label_vector(ts, labels)
        auprc, laap, mdel, meddel = _metrics(y, sf, ts, 2, args.bucket_seconds)
        sc = pd.DataFrame({'ts': ts, 'S1': s1, 'S2': s2, 'S3': s3, 'Sfinal': sf, 'label': y})
        sc.to_csv(os.path.join(out_dir, 'scores.csv'), index=False)
        rep = {
            'run': name,
            'bucket_seconds': args.bucket_seconds,
            'window': args.window,
            'min_seg_buckets': args.min_seg_buckets,
            'fusion': args.fusion,
            'threshold': thr,
            'pvalue': pv,
            'AUPRC': auprc,
            'LaAP': laap,
            'mean_delay': mdel,
            'median_delay': meddel
        }
        json.dump(rep, open(os.path.join(out_dir, 'report.json'), 'w'))
        _plot(name, out_dir, ts, s1, s2, s3, sf, labels, stage_series)
        print(f"{name}: AUPRC={auprc:.3f}, LaAP={laap:.3f}, mean_delay={(mdel if mdel is not None else 'NA')}, median_delay={(meddel if meddel is not None else 'NA')}, threshold={thr:.3f}")

if __name__ == '__main__':
    main()

