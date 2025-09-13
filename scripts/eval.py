import argparse, torch, torch.nn as nn
from src.utils.device import pick_device, device_name
from src.data.cifar10 import get_loaders
from src.models.resnet import resnet18_cifar10
from src.engine.train import evaluate

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--checkpoint", type=str, default="checkpoints/resnet18_cifar10.pth")
    return p.parse_args()

def main():
    args = parse_args()
    device = pick_device()
    print(f"Using device: {device} ({device_name(device)})")
    _, test_loader = get_loaders(args.data_root, args.batch_size, args.num_workers)

    model = resnet18_cifar10().to(device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(ckpt["model"])
    print(f"Loaded checkpoint from epoch {ckpt.get('epoch','?')} with val_acc={ckpt.get('val_acc','?')}")
    criterion = nn.CrossEntropyLoss()
    val_loss, val_acc = evaluate(model, test_loader, device, criterion)
    print(f"Eval: loss={val_loss:.4f} acc={val_acc:.4f}")

if __name__ == "__main__":
    main()
