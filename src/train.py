import argparse
import sys
import pathlib
import os
import time
import json
import threading
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))
from src.monitor.metrics_logger import MetricsLogger
from src.monitor.system_monitor import SystemMonitor
from src.events.event_sink import EventSink
from src.faults.model import ModelFaults
from src.faults.data import DataFaults
try:
    from src.faults.system import SystemFaults
except Exception:
    class SystemFaults:
        def __init__(self, fault, event_cb=None, throttle_watts=120, throttle_seconds=5, oom_mode='vram'):
            import threading, time
            self.fault = fault
            self.event_cb = event_cb
            self.throttle_watts = throttle_watts
            self.throttle_seconds = throttle_seconds
            self.oom_mode = oom_mode
            self.label_windows = []
            self.bg_thread = None
            self.stop_flag = threading.Event()
        def start_background(self):
            pass
        def stop_background(self):
            pass
        def should_ckpt_fail(self, step):
            return False
        def maybe_throttle_gpu(self, step):
            import time
            if self.fault == 'gpu_throttle' and step % 40 == 15:
                now = time.time()
                self.label_windows.append({'start': now, 'end': now+self.throttle_seconds, 'label': 'gpu_throttle'})
                time.sleep(self.throttle_seconds)
        def should_oom(self, step):
            return self.fault in ('oom','oom_vram','oom_ram') and step % 80 == 25
        def apply_oom(self, device):
            import time, torch, psutil
            now = time.time()
            if self.fault == 'oom_vram' and device.type == 'cuda':
                try:
                    _ = torch.empty((8192,8192), device=device)
                    del _
                except Exception:
                    self.label_windows.append({'start': now, 'end': now+5, 'label': 'oom_vram'})
            else:
                try:
                    avail = psutil.virtual_memory().available
                except Exception:
                    avail = 512*1024*1024
                reserved = []
                target = max(int(avail*0.5), 100*1024*1024)
                chunk = 100*1024*1024
                acc = 0
                while acc < target:
                    reserved.append(bytearray(chunk))
                    acc += chunk
                time.sleep(2)
                reserved.clear()
                self.label_windows.append({'start': now, 'end': now+2, 'label': 'oom_ram'})

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, n=5000, num_classes=10):
        self.n = n
        self.num_classes = num_classes
    def __len__(self):
        return self.n
    def __getitem__(self, idx):
        x = torch.rand(3,32,32)
        y = torch.randint(0, self.num_classes, (1,)).item()
        return x, y

def build_dataloaders(batch_size, num_workers, shuffle, synthetic=False):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    if synthetic:
        trainset = SyntheticDataset()
        testset = SyntheticDataset(n=1000)
    else:
        trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
        testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return trainloader, testloader

def build_model(device):
    model = torchvision.models.resnet18(num_classes=10)
    model.to(device)
    return model

def compute_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += p.grad.data.norm(2).item() ** 2
    return np.sqrt(total)

def save_checkpoint(path, model, optimizer, epoch, step):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({'model': model.state_dict(), 'optim': optimizer.state_dict(), 'epoch': epoch, 'step': step}, path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fault', type=str, default='clean')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--out-dir', type=str, default='datasets/OpenTSE-Logs')
    parser.add_argument('--bucket-seconds', type=int, default=10)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--throttle-watts', type=int, default=120)
    parser.add_argument('--throttle-seconds', type=int, default=5)
    parser.add_argument('--oom-mode', type=str, default='vram')
    args = parser.parse_args()

    run_name = f"run_{int(time.time())}_{args.fault}"
    run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trainloader, testloader = build_dataloaders(args.batch_size, args.num_workers, shuffle=(args.fault != 'shuffle_off'), synthetic=args.synthetic)
    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

    metrics_logger = MetricsLogger(os.path.join(run_dir, 'metrics.csv'))
    system_monitor = SystemMonitor(os.path.join(run_dir, 'system.csv'))
    event_sink = EventSink(os.path.join(run_dir, 'events.json'))

    model_faults = ModelFaults(args.fault)
    data_faults = DataFaults(args.fault)
    system_faults = SystemFaults(
        fault=args.fault,
        event_cb=event_sink.write,
        throttle_watts=args.throttle_watts,
        throttle_seconds=args.throttle_seconds,
        oom_mode=args.oom_mode,
    )

    system_monitor.start()
    system_faults.start_background()
    data_faults.register_shuffle_off_window()

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            ts0 = time.time()
            event_sink.write({'ts': ts0, 'stage': 'forward_start', 'type': 'stage', 'source': 'train'})
            if data_faults.should_slow_io(global_step):
                time.sleep(data_faults.io_sleep_seconds())
            if data_faults.should_corrupt(global_step):
                inputs = data_faults.apply_corrupt(inputs)
            inputs, targets = inputs.to(device), targets.to(device)
            system_faults.maybe_throttle_gpu(global_step)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            if model_faults.should_loss_nan(global_step):
                loss = model_faults.apply_loss_nan(loss, device)
            event_sink.write({'ts': time.time(), 'stage': 'forward_end', 'type': 'stage', 'source': 'train'})

            event_sink.write({'ts': time.time(), 'stage': 'backward_start', 'type': 'stage', 'source': 'train'})
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if model_faults.should_explode(global_step):
                model_faults.apply_grad_explode(model)
            event_sink.write({'ts': time.time(), 'stage': 'backward_end', 'type': 'stage', 'source': 'train'})

            event_sink.write({'ts': time.time(), 'stage': 'step_start', 'type': 'stage', 'source': 'train'})
            if model_faults.should_lr_spike(global_step):
                model_faults.apply_lr_spike(optimizer)
            optimizer.step()
            if model_faults.should_lr_restore(global_step):
                model_faults.restore_lr(optimizer)
            event_sink.write({'ts': time.time(), 'stage': 'step_end', 'type': 'stage', 'source': 'train'})

            event_sink.write({'ts': time.time(), 'stage': 'scheduler_step', 'type': 'stage', 'source': 'train'})
            scheduler.step()

            if system_faults.should_oom(global_step):
                event_sink.write({'ts': time.time(), 'stage': 'oom_attempt', 'type': 'system', 'source': 'train'})
                system_faults.apply_oom(device)

            grad_norm = compute_grad_norm(model)
            lr = optimizer.param_groups[0]['lr']
            step_time = max(time.time() - ts0, 1e-6)
            throughput = inputs.size(0) / step_time
            metrics_logger.write({'ts': time.time(), 'epoch': epoch, 'step': global_step, 'loss': loss.item(), 'lr': lr, 'grad_norm': grad_norm, 'throughput': throughput})

            if batch_idx % 200 == 0:
                ckpt_path = os.path.join(run_dir, f'ckpt_epoch{epoch}_step{global_step}.pt')
                try:
                    event_sink.write({'ts': time.time(), 'stage': 'checkpoint_start', 'type': 'platform', 'source': 'train'})
                    if system_faults.should_ckpt_fail(global_step):
                        raise RuntimeError('checkpoint_write_failed')
                    save_checkpoint(ckpt_path, model, optimizer, epoch, global_step)
                    event_sink.write({'ts': time.time(), 'stage': 'checkpoint_end', 'type': 'platform', 'source': 'train'})
                except Exception as e:
                    event_sink.write({'ts': time.time(), 'stage': 'checkpoint_error', 'type': 'platform', 'source': 'train', 'error': str(e)})

            global_step += 1

    system_monitor.stop()
    system_faults.stop_background()

    event_sink.flush()
    metrics_logger.flush()
    system_monitor.flush()

    labels_path = os.path.join(run_dir, 'anomaly_labels.csv')
    with open(labels_path, 'w') as f:
        f.write('ts_start,ts_end,label\n')
        for w in system_faults.label_windows:
            f.write(f"{w['start']},{w['end']},{w['label']}\n")
        for w in data_faults.label_windows:
            f.write(f"{w['start']},{w['end']},{w['label']}\n")
        for w in model_faults.label_windows:
            f.write(f"{w['start']},{w['end']},{w['label']}\n")

    stage_path = os.path.join(run_dir, 'stage.csv')
    with open(stage_path, 'w') as f:
        f.write('ts,stage\n')
        for e in event_sink.buffer:
            if e.get('type') == 'stage':
                f.write(f"{e['ts']},{e['stage']}\n")

    from src.tse.build_matrix import build_tse_matrix
    build_tse_matrix(run_dir, args.bucket_seconds)

if __name__ == '__main__':
    main()
