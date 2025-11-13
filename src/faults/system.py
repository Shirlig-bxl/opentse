import time
import threading
import subprocess


class GpuThrottler:
    def __init__(self, event_cb=None):
        self.event_cb = event_cb
        self.nvml_ok = False
        self.handle = None
        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml = pynvml
            self.handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.nvml_ok = True
        except Exception:
            self.nvml_ok = False

    def _event(self, payload):
        if self.event_cb:
            evt = {}
            evt['ts'] = time.time()
            evt['stage'] = 'fault'
            evt['type'] = 'system'
            evt['source'] = 'fault_injector'
            evt['name'] = 'gpu_throttle'
            evt['payload'] = payload
            self.event_cb(evt)

    def get_power_limit(self):
        if not self.nvml_ok:
            return None
        try:
            return int(self.nvml.nvmlDeviceGetPowerManagementLimit(self.handle))
        except Exception:
            return None

    def set_power_limit(self, watts):
        if not self.nvml_ok:
            return False
        try:
            self.nvml.nvmlDeviceSetPowerManagementLimit(self.handle, int(watts) * 1000)
            return True
        except Exception as e:
            self._event({'action': 'nvml_set', 'status': 'error', 'reason': str(e)})
            return False

    def throttle(self, seconds, watts):
        start = time.time()
        self._event({'action': 'start', 'status': 'begin'})
        ok = False
        before = self.get_power_limit()
        if self.nvml_ok and before is not None:
            if self.set_power_limit(watts):
                ok = True
                self._event({'action': 'nvml_set', 'status': 'ok', 'before_pl': before, 'after_pl': watts * 1000})
        if not ok:
            try:
                rc = subprocess.call(['nvidia-smi', '-pl', str(int(watts))])
                if rc == 0:
                    ok = True
                    self._event({'action': 'nvidia-smi', 'status': 'ok'})
                else:
                    self._event({'action': 'nvidia-smi', 'status': 'denied', 'rc': rc})
            except Exception as e:
                self._event({'action': 'nvidia-smi', 'status': 'error', 'reason': str(e)})
        if not ok:
            time.sleep(float(seconds))
            self._event({'action': 'sleep', 'status': 'simulated', 'seconds': seconds})
        else:
            time.sleep(float(seconds))
            if self.nvml_ok and before is not None:
                try:
                    self.nvml.nvmlDeviceSetPowerManagementLimit(self.handle, before)
                except Exception:
                    pass
        end = time.time()
        self._event({'action': 'end', 'status': 'done', 'duration': end - start})


class SystemFaults:
    def __init__(self, fault, event_cb=None, throttle_watts=120, throttle_seconds=5, oom_mode='vram'):
        self.fault = fault
        self.event_cb = event_cb
        self.throttle_watts = throttle_watts
        self.throttle_seconds = throttle_seconds
        self.oom_mode = oom_mode
        self.label_windows = []
        self.bg_thread = None
        self.stop_flag = threading.Event()
        self.ckpt_fail_once = False
        self.throttle_flag = False
        self._throttler = GpuThrottler(event_cb=self.event_cb)

    def start_background(self):
        if self.fault == 'cpu_steal':
            self.bg_thread = threading.Thread(target=self._cpu_burn, daemon=True)
            self.bg_thread.start()

    def stop_background(self):
        self.stop_flag.set()
        if self.bg_thread is not None:
            self.bg_thread.join(timeout=2)

    def _cpu_burn(self):
        start = time.time()
        self.label_windows.append({'start': start, 'end': start + 20, 'label': 'cpu_steal'})
        while not self.stop_flag.is_set():
            x = 0
            for i in range(1000000):
                x += i
            time.sleep(0.01)

    def should_ckpt_fail(self, step):
        if self.fault == 'ckpt_fail' and not self.ckpt_fail_once and step >= 50:
            self.ckpt_fail_once = True
            now = time.time()
            self.label_windows.append({'start': now, 'end': now + 2, 'label': 'ckpt_fail'})
            return True
        return False

    def maybe_throttle_gpu(self, step):
        if self.fault == 'gpu_throttle' and step % 40 == 15:
            self.throttle_flag = True
            now = time.time()
            end = now + float(self.throttle_seconds)
            self.label_windows.append({'start': now, 'end': end, 'label': 'gpu_throttle'})
            self._throttler.throttle(self.throttle_seconds, self.throttle_watts)
        else:
            self.throttle_flag = False

    def should_oom(self, step):
        if self.fault in ('oom', 'oom_vram', 'oom_ram') and step % 80 == 25:
            return True
        return False

    def apply_oom(self, device):
        now = time.time()
        mode = self.oom_mode
        if self.fault == 'oom_vram':
            mode = 'vram'
        if self.fault == 'oom_ram':
            mode = 'ram'
        if mode == 'vram':
            import torch
            try:
                blocks = []
                for _ in range(8):
                    if device.type == 'cuda':
                        blocks.append(torch.empty((8192, 8192), device=device))
                    else:
                        blocks.append(torch.empty((8192, 8192)))
                blocks.clear()
                if device.type == 'cuda':
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
            except Exception as e:
                end = now + 5
                self.label_windows.append({'start': now, 'end': end, 'label': 'oom_vram'})
                if self.event_cb:
                    payload = {'caught': True, 'error': str(e)}
                    evt = {}
                    evt['ts'] = time.time()
                    evt['stage'] = 'fault'
                    evt['type'] = 'system'
                    evt['source'] = 'fault_injector'
                    evt['name'] = 'oom_vram'
                    evt['payload'] = payload
                    self.event_cb(evt)
        else:
            try:
                import psutil
                try:
                    avail = psutil.virtual_memory().available
                except Exception:
                    avail = 512 * 1024 * 1024
                reserved = []
                target = max(int(avail * 0.5), 100 * 1024 * 1024)
                chunk = 100 * 1024 * 1024
                acc = 0
                while acc < target:
                    reserved.append(bytearray(chunk))
                    acc += chunk
                time.sleep(2)
                reserved.clear()
                end = now + 2
                self.label_windows.append({'start': now, 'end': end, 'label': 'oom_ram'})
                if self.event_cb:
                    payload = {'reserved_bytes': acc}
                    evt = {}
                    evt['ts'] = time.time()
                    evt['stage'] = 'fault'
                    evt['type'] = 'system'
                    evt['source'] = 'fault_injector'
                    evt['name'] = 'oom_ram'
                    evt['payload'] = payload
                    self.event_cb(evt)
