import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# ---- 0) Pick best device: CUDA (WSL) → MPS (macOS) → CPU
def pick_device(prefer_cuda: bool = True) -> torch.device:
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")

device = pick_device()
is_cuda = device.type == "cuda"
is_mps  = device.type == "mps"
print("Using device:", device)

# Optional CUDA perf tweak for convnets with stable shapes
if is_cuda:
    torch.backends.cudnn.benchmark = True

# ---- 1) Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),

])

data_root = "./data"
trainset = torchvision.datasets.CIFAR10(root=data_root, train=True,  download=True, transform=transform)
testset  = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)

# Use pin_memory only for CUDA (gives faster H2D copies). Non-blocking copies only on CUDA.
num_workers = min(4, os.cpu_count() or 1)   # you can raise this if disks/CPU are fast
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True,
    num_workers=num_workers,
    pin_memory=is_cuda,
    persistent_workers=is_cuda and num_workers > 0,
)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False,
    num_workers=num_workers,
    pin_memory=is_cuda,
    persistent_workers=is_cuda and num_workers > 0,
)

# ---- 2) Model
model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)

# ---- 3) Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Mixed precision: enable only on CUDA (MPS autocast is still limited/experimental across versions)
scaler = torch.cuda.amp.GradScaler(enabled=is_cuda)

# ---- 4) Training loop
epochs = 5
for epoch in range(epochs):
    model.train()
    running = 0.0

    for images, labels in trainloader:
        # Send data to the selected device
        if is_cuda:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
        else:
            images = images.to(device)
            labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        if is_cuda:
            # Faster & memory-efficient on NVIDIA
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # MPS/CPU path
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        running += loss.item()

    print(f"Epoch {epoch+1}/{epochs} - Loss: {running/len(trainloader):.4f}")

# ---- 5) Evaluation


