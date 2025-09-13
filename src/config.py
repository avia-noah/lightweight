from dataclasses import dataclass
import os

@dataclass
class TrainConfig:
    data_root: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    epochs: int = 20
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-2
    step_size: int = 30
    gamma: float = 0.1
    num_workers: int = 4
    amp: bool = True      # mixed precision
    checkpoint: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "checkpoints", "resnet18_cifar10.pth")
    subset: int | None = None  # for quick dev runs
