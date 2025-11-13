import csv
import os
import threading

class MetricsLogger:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.buffer = []
        self.lock = threading.Lock()
        with open(self.path, 'w') as f:
            w = csv.writer(f)
            w.writerow(['ts','epoch','step','loss','lr','grad_norm','throughput'])

    def write(self, row):
        with self.lock:
            self.buffer.append(row)

    def flush(self):
        with self.lock:
            with open(self.path, 'a') as f:
                w = csv.writer(f)
                for r in self.buffer:
                    w.writerow([r['ts'], r['epoch'], r['step'], r['loss'], r['lr'], r['grad_norm'], r['throughput']])
            self.buffer = []
