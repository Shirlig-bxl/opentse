import time
import torch
import torch.nn.functional as F

class DataFaults:
    def __init__(self, fault):
        self.fault = fault
        self.label_windows = []

    def should_slow_io(self, step):
        return self.fault == 'io_slow' and step % 30 == 5

    def io_sleep_seconds(self):
        now = time.time()
        self.label_windows.append({'start': now, 'end': now+3, 'label': 'io_slow'})
        return 0.3

    def should_corrupt(self, step):
        return self.fault == 'data_corrupt' and step % 40 == 7

    def apply_corrupt(self, inputs):
        noise = torch.randn_like(inputs) * 0.25
        now = time.time()
        self.label_windows.append({'start': now, 'end': now+5, 'label': 'data_corrupt'})
        return torch.clamp(inputs + noise, 0.0, 1.0)

    def register_shuffle_off_window(self):
        if self.fault == 'shuffle_off':
            now = time.time()
            self.label_windows.append({'start': now, 'end': now+3600, 'label': 'shuffle_off'})
