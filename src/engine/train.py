from typing import Optional
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from ..utils.meter import AverageMeter

def make_optimizer(model, lr: float, weight_decay: float):
    return optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

def make_scheduler(optimizer, step_size: int, gamma: float):
    return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

def train_one_epoch(model, loader, device, criterion, optimizer, amp: bool=True):
    model.train()
    loss_meter, acc_meter = AverageMeter("loss"), AverageMeter("acc")
    scaler = GradScaler(enabled=amp)
    for images, targets in tqdm(loader, desc="train", leave=False):
        images, targets = images.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(enabled=amp):
            logits = model(images)
            loss = criterion(logits, targets)
        scaler.scale(loss).step(optimizer)
        scaler.update()
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
