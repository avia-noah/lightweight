import argparse, os, torch, torch.nn as nn
from src.config import TrainConfig
from src.utils.seed import set_seed
from src.utils.device import pick_device, device_name
from src.data.cifar10 import get_loaders
from src.models.resnet import resnet18_cifar10
from src.engine.train import make_optimizer, make_scheduler, train_one_epoch, evaluate

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default=TrainConfig.data_root)
    p.add_argument("--epochs", type=int, default=TrainConfig.epochs)
    p.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    p.add_argument("--lr", type=float, default=TrainConfig.lr)
    p.add_argument("--weight-decay", type=float, default=TrainConfig.weight_decay)
    p.add_argument("--step-size", type=int, default=TrainConfig.step_size)
    p.add_argument("--gamma", type=float, default=TrainConfig.gamma)
    p.add_argument("--num-workers", type=int, default=TrainConfig.num_workers)
    p.add_argument("--amp", type=int, default=int(TrainConfig.amp))
    p.add_argument("--checkpoint", type=str, default=TrainConfig.checkpoint)
    p.add_argument("--subset", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
    set_seed(42)
    device = pick_device()
    print(f"Using device: {device} ({device_name(device)})")

    # data
    train_loader, test_loader = get_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        subset=args.subset
    )

    # model
    model = resnet18_cifar10(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)
    scheduler = make_scheduler(optimizer, args.step_size, args.gamma)

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, device, criterion, optimizer, amp=bool(args.amp))
        va_loss, va_acc = evaluate(model, test_loader, device, criterion)
        scheduler.step()
        print(f"Epoch {epoch:03d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | val loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            torch.save({"model": model.state_dict(),
                        "epoch": epoch,
                        "val_acc": best_acc}, args.checkpoint)
            print(f"✓ Saved checkpoint → {args.checkpoint} (acc={best_acc:.4f})")

if __name__ == "__main__":
    main()
