import torch

def pick_device() -> torch.device:
    # Prefer MPS on Apple Silicon when available
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    # Prefer CUDA on WSL2/Windows/Linux when available
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def device_name(dev: torch.device) -> str:
    if dev.type == "cuda":
        try:
            return torch.cuda.get_device_name(0)
        except Exception:
            return "NVIDIA CUDA"
    if dev.type == "mps":
        return "Apple Metal (MPS)"
    return "CPU"
