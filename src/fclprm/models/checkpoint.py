"""Model checkpoint save/load utilities."""

import os
from pathlib import Path

import torch
import torch.nn as nn


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    round_num: int,
    client_id: int,
    milestone: str,
    save_dir: str,
) -> str:
    """Save training checkpoint.

    Naming: {model_name}_m{milestone}_r{round}_c{client_id}.pt

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        round_num: Federated round number.
        client_id: Client identifier (-1 for global model).
        milestone: Milestone tag (e.g., "M4").
        save_dir: Directory to save checkpoint.

    Returns:
        Path to saved checkpoint file.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_name = getattr(model, "name", "model")
    filename = f"{model_name}_m{milestone}_r{round_num}_c{client_id}.pt"
    filepath = save_path / filename

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "round_num": round_num,
        "client_id": client_id,
        "milestone": milestone,
    }

    torch.save(checkpoint, filepath)
    return str(filepath)


def load_checkpoint(
    path: str, model: nn.Module, optimizer: torch.optim.Optimizer | None = None
) -> dict:
    """Load checkpoint into model.

    Args:
        path: Path to checkpoint file.
        model: Model to load state into.
        optimizer: Optional optimizer to restore state.

    Returns:
        Dict of metadata (round_num, client_id, milestone).
    """
    checkpoint = torch.load(path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    return {
        "round_num": checkpoint.get("round_num", 0),
        "client_id": checkpoint.get("client_id", -1),
        "milestone": checkpoint.get("milestone", ""),
    }
