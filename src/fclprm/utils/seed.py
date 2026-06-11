"""Random seed management for reproducibility."""

import random

import numpy as np
import torch


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set random seeds for Python, NumPy, and PyTorch.

    Args:
        seed: Integer seed value.
        deterministic: If True, also set torch.backends.cudnn.deterministic
            and disable cudnn.benchmark for fully deterministic results.
            Note: this may reduce GPU performance.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
