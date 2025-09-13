from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.amp import GradScaler, autocast
from tqdm import tqdm
from utils.meter import AverageMeter

def make_optimizer(model, lr: float, weight_decay: float):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def make_scheduler(optimizer, step_size: int, gamma: float):
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def train_one_epoch(model, loader, device, criterion, optimizer, amp: bool=True):
    model.train()
    loss_meter, acc_meter = AverageMeter("loss"), AverageMeter("acc")
    
    # Print device info for debugging
    print(f"Training on device: {device} ({'MPS' if device.type == 'mps' else 'CUDA' if device.type == 'cuda' else 'CPU'})")
    
    # Determine AMP configuration based on device
    if device.type == 'cuda' and torch.cuda.is_available():
        # CUDA: Use native CUDA AMP
        device_type = 'cuda'
        use_amp = amp
    elif device.type == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS: Use CPU scaler for AMP (MPS doesn't have native AMP support)
        device_type = 'cpu'
        use_amp = amp
    else:
        # CPU: Use CPU AMP or disable
        device_type = 'cpu'
        use_amp = amp
    
    # Initialize scaler only if AMP is enabled
    scaler = GradScaler(device_type, enabled=use_amp) if use_amp else None
    for images, targets in tqdm(loader, desc="train", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp and scaler is not None:
            # Use AMP
            with autocast(device_type, enabled=True):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without AMP
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
        # metrics
        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == targets).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
    return loss_meter.avg, acc_meter.avg

@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_meter, acc_meter = AverageMeter("loss"), AverageMeter("acc")
    for images, targets in tqdm(loader, desc="eval", leave=False):
        images, targets = images.to(device), targets.to(device)
        logits = model(images)
        loss = criterion(logits, targets)
        preds = logits.argmax(dim=1)
        acc = (preds == targets).float().mean().item()
        loss_meter.update(loss.item(), images.size(0))
        acc_meter.update(acc, images.size(0))
    return loss_meter.avg, acc_meter.avg
