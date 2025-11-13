import pandas as pd

def bucketize(df, bucket_seconds, agg):
    if df.empty:
        return df
    df = df.copy()
    df['bucket'] = (df['ts'] // bucket_seconds).astype(int)
    return df.groupby('bucket').agg(agg).reset_index()
