import csv
import os
import threading
import time
import subprocess

class SystemMonitor:
    def __init__(self, path, hz=1):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.hz = hz
        self.thread = None
        self.stop_flag = threading.Event()
        self.buffer = []
        self.lock = threading.Lock()
        with open(self.path, 'w') as f:
            w = csv.writer(f)
            w.writerow(['ts','cpu_util','ram_used','gpu_util','gpu_mem','io_read','io_write'])

    def _get_cpu_ram(self):
        try:
            import psutil
            cpu = psutil.cpu_percent(interval=None)
            ram = psutil.virtual_memory().used
            io = psutil.disk_io_counters()
            read_b = getattr(io, 'read_bytes', 0)
            write_b = getattr(io, 'write_bytes', 0)
            return cpu, ram, read_b, write_b
        except Exception:
            return 0.0, 0, 0, 0

    def _get_gpu(self):
        util = 0.0
        mem = 0
        try:
            import pynvml
            pynvml.nvmlInit()
            h = pynvml.nvmlDeviceGetHandleByIndex(0)
            u = pynvml.nvmlDeviceGetUtilizationRates(h)
            util = float(u.gpu)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(h)
            mem = int(meminfo.used)
            pynvml.nvmlShutdown()
        except Exception:
            try:
                out = subprocess.check_output(['nvidia-smi','--query-gpu=utilization.gpu,memory.used','--format=csv,noheader,nounits'], timeout=1).decode().strip()
                parts = out.split(',')
                if len(parts) >= 2:
                    util = float(parts[0])
                    mem = int(parts[1])
            except Exception:
                util = 0.0
                mem = 0
        return util, mem

    def _run(self):
        while not self.stop_flag.is_set():
            ts = time.time()
            cpu, ram, io_r, io_w = self._get_cpu_ram()
            gpu_u, gpu_m = self._get_gpu()
            with self.lock:
                self.buffer.append({'ts': ts, 'cpu': cpu, 'ram': ram, 'gpu_u': gpu_u, 'gpu_m': gpu_m, 'io_r': io_r, 'io_w': io_w})
            time.sleep(1.0/max(self.hz,1))

    def start(self):
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def stop(self):
        self.stop_flag.set()
        if self.thread is not None:
            self.thread.join(timeout=2)

    def flush(self):
        with self.lock:
            with open(self.path, 'a') as f:
                w = csv.writer(f)
                for r in self.buffer:
                    w.writerow([r['ts'], r['cpu'], r['ram'], r['gpu_u'], r['gpu_m'], r['io_r'], r['io_w']])
            self.buffer = []
