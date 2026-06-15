"""Model checkpoint save/load utilities."""

import gc
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
    device: str = "cuda",
) -> str:
    """Save training checkpoint.

    All tensors are moved to CPU before serialization, then freed, to avoid
    holding GPU memory during disk I/O.  GPU cache is flushed afterward.

    Naming: {model_name}_m{milestone}_r{round}_c{client_id}.pt

    Args:
        model: Model to save.
        optimizer: Optimizer state.
        round_num: Federated round number.
        client_id: Client identifier (-1 for global model).
        milestone: Milestone tag (e.g., "M4").
        save_dir: Directory to save checkpoint.
        device: Device hint for cache flush ("cuda" or "cpu").

    Returns:
        Path to saved checkpoint file.
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    model_name = getattr(model, "name", "model")
    filename = f"{model_name}_m{milestone}_r{round_num}_c{client_id}.pt"
    filepath = save_path / filename

    # Move state dict to CPU before saving to avoid GPU memory spike during I/O
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # optimizer.state_dict() returns nested structure:
    #   {"state": {param_id: {tensor_name: tensor, ...}}, "param_groups": [{...}]}
    # Only the leaf tensors inside "state" need .cpu().
    raw_optim = optimizer.state_dict()
    cpu_optim: dict = {"state": {}, "param_groups": raw_optim["param_groups"]}
    for param_id, state_vals in raw_optim["state"].items():
        cpu_optim["state"][param_id] = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in state_vals.items()
        }

    checkpoint = {
        "model_state_dict": cpu_state,
        "optimizer_state_dict": cpu_optim,
        "round_num": round_num,
        "client_id": client_id,
        "milestone": milestone,
    }

    torch.save(checkpoint, filepath)

    # Free CPU-side copies and flush GPU cache
    del cpu_state, cpu_optim, checkpoint
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

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
