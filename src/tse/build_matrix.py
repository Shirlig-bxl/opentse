import os
import json
import numpy as np
import pandas as pd
from src.tse.align import bucketize

def build_tse_matrix(run_dir, bucket_seconds):
    metrics = pd.read_csv(os.path.join(run_dir, 'metrics.csv'))
    system = pd.read_csv(os.path.join(run_dir, 'system.csv'))
    stages = pd.read_csv(os.path.join(run_dir, 'stage.csv'))
    events = []
    p = os.path.join(run_dir, 'events.json')
    with open(p, 'r') as f:
        for line in f:
            try:
                events.append(json.loads(line))
            except Exception:
                pass
    events_df = pd.DataFrame(events)
    metrics_b = bucketize(metrics, bucket_seconds, {'loss':'mean','lr':'mean','grad_norm':'mean','throughput':'mean','ts':'min'})
    system_b = bucketize(system, bucket_seconds, {'cpu_util':'mean','ram_used':'mean','gpu_util':'mean','gpu_mem':'mean','io_read':'max','io_write':'max','ts':'min'})
    if not events_df.empty:
        events_df['bucket'] = (events_df['ts'] // bucket_seconds).astype(int)
        counts = events_df.groupby(['bucket','type']).size().unstack(fill_value=0)
    else:
        counts = pd.DataFrame()
    stage_b = bucketize(stages, bucket_seconds, {'ts':'min'})
    if not stage_b.empty:
        stage_b['stage'] = 'train'
    X = metrics_b.merge(system_b, on='bucket', how='outer')
    if not counts.empty:
        X = X.merge(counts, on='bucket', how='left').fillna(0)
    X = X.sort_values('bucket')
    cols = [c for c in X.columns if c not in ['bucket','ts']]
    arr = X[cols].to_numpy(dtype=np.float32)
    np.save(os.path.join(run_dir, 'tse_matrix.npy'), arr)
    fmap = {c:i for i,c in enumerate(cols)}
    with open(os.path.join(run_dir, 'feature_map.json'), 'w') as f:
        f.write(json.dumps(fmap))
