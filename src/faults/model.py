import time
import torch

class ModelFaults:
    def __init__(self, fault):
        self.fault = fault
        self.lr_spike_applied = False
        self.lr_saved = None
        self.label_windows = []

    def should_explode(self, step):
        return self.fault == 'grad_explode' and step % 50 == 10

    def apply_grad_explode(self, model):
        for p in model.parameters():
            if p.grad is not None:
                p.grad.mul_(1000.0)
        now = time.time()
        self.label_windows.append({'start': now, 'end': now+5, 'label': 'grad_explode'})

    def should_lr_spike(self, step):
        return self.fault == 'lr_spike' and not self.lr_spike_applied and step >= 100

    def apply_lr_spike(self, optimizer):
        if self.lr_saved is None:
            self.lr_saved = optimizer.param_groups[0]['lr']
        optimizer.param_groups[0]['lr'] = self.lr_saved * 10.0
        self.lr_spike_applied = True
        now = time.time()
        self.label_windows.append({'start': now, 'end': now+10, 'label': 'lr_spike'})

    def should_lr_restore(self, step):
        return self.fault == 'lr_spike' and self.lr_spike_applied and step >= 110

    def restore_lr(self, optimizer):
        if self.lr_saved is not None:
            optimizer.param_groups[0]['lr'] = self.lr_saved

    def should_loss_nan(self, step):
        return self.fault == 'loss_nan' and step % 60 == 20

    def apply_loss_nan(self, loss, device):
        import torch
        loss = loss * torch.tensor(float('nan'), device=device)
        now = time.time()
        self.label_windows.append({'start': now, 'end': now+5, 'label': 'loss_nan'})
        return loss
