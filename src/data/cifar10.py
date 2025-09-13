from typing import Tuple, Optional
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torch, os, numpy as np

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def _build_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

def get_loaders(
    data_root: str, batch_size: int, num_workers: int = 4, subset: Optional[int]=None
) -> Tuple[DataLoader, DataLoader]:
    os.makedirs(data_root, exist_ok=True)
    train_set = datasets.CIFAR10(data_root, train=True, download=True, transform=_build_transforms(True))
    test_set  = datasets.CIFAR10(data_root, train=False, download=True, transform=_build_transforms(False))

    if subset is not None:
        idx = np.random.RandomState(0).choice(len(train_set), size=subset, replace=False)
        train_set = Subset(train_set, idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)
    return train_loader, test_loader
