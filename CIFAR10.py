import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


import os

print("CWD:", os.getcwd())


# 0. Pick device
device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu")
print("Using device:", device)

# 1. Data
MEAN = (0.4914, 0.4822, 0.4465)
STD = (0.2470, 0.2435, 0.2616)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)



#%% For My Understanding - Ploting
images, labels = next(iter(trainloader))

print("Images shape:", images.shape)
print("Labels shape:", labels.shape)

# Get one batch
# Number of images you want to show
num_images = 16  # show 4x4 grid
images = images[:num_images]
labels = labels[:num_images]

# Plot
fig, axes = plt.subplots(4, 4, figsize=(6, 6))
for i, ax in enumerate(axes.flat):
    # Convert to HWC on CPU
    img = images[i].permute(1, 2, 0).cpu()

    # Undo normalization for better display (CIFAR-10 stats)
    img = (img * torch.tensor(STD)) + torch.tensor(MEAN)
    img = img.numpy()

    ax.imshow(img.squeeze(), cmap="gray" if img.shape[2] == 1 else None)
    ax.set_title(f"Label: {labels[i].item()}", fontsize=8)
    ax.axis("off")

plt.tight_layout()
plt.show()


#%%
# 2. Model
model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)

# 3. Loss & Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 4. Training loop
num_epochs = 3
train_losses, val_losses = [], []
train_accuracies, val_accuracies = [], []
for epoch in range(num_epochs):
    running_loss = 0.0
    correct_train, total_train = 0, 0
    for images, labels in tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]", leave=False):
        # ⬇️ Send data to device
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()

    epoch_train_loss = running_loss / max(1, len(trainloader))
    epoch_train_acc = 100.0 * correct_train / max(1, total_train)
    train_losses.append(epoch_train_loss)
    train_accuracies.append(epoch_train_acc)

    # Validation
    model.eval()
    val_running_loss = 0.0
    correct_val, total_val = 0, 0
    with torch.no_grad():
        for images, labels in tqdm(testloader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]", leave=False):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()
    model.train()

    epoch_val_loss = val_running_loss / max(1, len(testloader))
    epoch_val_acc = 100.0 * correct_val / max(1, total_val)
    val_losses.append(epoch_val_loss)
    val_accuracies.append(epoch_val_acc)

    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print(
        f"Epoch {epoch+1}/{num_epochs} | Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.2f}% | "
        f"Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.2f}% | LR: {current_lr:.6f}"
    )

# 5. Evaluation & Plots
# Final test accuracy and confusion matrix
correct, total = 0, 0
all_preds, all_labels = [], []
with torch.no_grad():
    for images, labels in tqdm(testloader, desc="Final Eval", leave=False):
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        all_preds.append(predicted.cpu())
        all_labels.append(labels.cpu())

test_accuracy = 100.0 * correct / max(1, total)
print(f"Final Test Accuracy: {test_accuracy:.2f}%")

# Plot training curves
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].plot(range(1, num_epochs+1), train_losses, label='Train')
axes[0].plot(range(1, num_epochs+1), val_losses, label='Val')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(range(1, num_epochs+1), train_accuracies, label='Train')
axes[1].plot(range(1, num_epochs+1), val_accuracies, label='Val')
axes[1].set_title('Accuracy (%)')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy (%)')
axes[1].legend()
plt.tight_layout()
plt.show()

# Confusion matrix
all_preds = torch.cat(all_preds).numpy()
all_labels = torch.cat(all_labels).numpy()
cm = confusion_matrix(all_labels, all_preds, labels=list(range(10)))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=trainset.classes)
fig, ax = plt.subplots(figsize=(6, 6))
disp.plot(ax=ax, cmap='Blues', colorbar=False)
plt.title('Confusion Matrix - CIFAR-10')
plt.tight_layout()
plt.show()
#%%
