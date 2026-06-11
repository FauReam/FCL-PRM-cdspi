"""Shared data utilities: tokenization, collate, step splitting, file loading."""

from pathlib import Path
from typing import Callable

import torch
from torch.nn.utils.rnn import pad_sequence


def _normalize_labels(labels: list) -> list[float]:
    """Normalize raw step labels to float rewards in [0, 1].

    Handles common label formats:
        - 1 / 0  -> 1.0 / 0.0
        - "+" / "-"  -> 1.0 / 0.0
        - "correct" / "incorrect"  -> 1.0 / 0.0

    Args:
        labels: Raw label list from dataset JSON.

    Returns:
        Normalized float labels.
    """
    normalized = []
    for lbl in labels:
        if isinstance(lbl, (int, float)):
            normalized.append(1.0 if lbl > 0 else 0.0)
        elif isinstance(lbl, str):
            normalized.append(
                1.0 if lbl.lower() in ("1", "+", "correct", "true") else 0.0
            )
        else:
            normalized.append(0.0)
    return normalized


def _load_jsonl_or_json(data_dir: Path, stem: str) -> list[dict]:
    """Load a dataset from JSONL or JSON file.

    Tries ``{stem}.jsonl`` first, then ``{stem}.json``.

    Args:
        data_dir: Directory containing the data file.
        stem: File stem (e.g., "train", "versa_prm", "med_prm").

    Returns:
        List of raw sample dicts.

    Raises:
        FileNotFoundError: If neither JSONL nor JSON file exists.
    """
    data_file = data_dir / f"{stem}.jsonl"
    if not data_file.exists():
        data_file = data_dir / f"{stem}.json"

    if not data_file.exists():
        raise FileNotFoundError(
            f"No data file found for stem '{stem}' in {data_dir} "
            f"(tried .jsonl and .json)"
        )

    samples = []
    if data_file.suffix == ".jsonl":
        # Count lines for progress bar
        with open(data_file, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

        with open(data_file, "r", encoding="utf-8") as f:
            try:
                from tqdm import tqdm

                pbar = tqdm(
                    f, total=total_lines, desc=f"Loading {data_file.name}", leave=False
                )
            except ImportError:
                pbar = f
            for line in pbar:
                if line.strip():
                    import json

                    samples.append(json.loads(line))
            if hasattr(pbar, "close"):
                pbar.close()
    else:
        import json

        with open(data_file, "r", encoding="utf-8") as f:
            samples = json.load(f)

    return samples


def _normalize_dataset(samples: list[dict]) -> list[dict]:
    """In-place normalize labels for a list of samples.

    Args:
        samples: List of sample dicts, each optionally containing a "labels" key.

    Returns:
        The same list with normalized labels.
    """
    for sample in samples:
        if "labels" in sample:
            sample["labels"] = _normalize_labels(sample["labels"])
    return samples


def split_cot_into_steps(cot_text: str, delimiter: str = "\n\n") -> list[str]:
    """Split a chain-of-thought into individual steps.

    Args:
        cot_text: Full chain-of-thought text.
        delimiter: Step delimiter. Default is double newline.

    Returns:
        List of step strings with empty lines stripped.
    """
    steps = [s.strip() for s in cot_text.split(delimiter) if s.strip()]
    return steps


def collate_step_batch(batch: list[dict], pad_token_id: int = 0) -> dict:
    """Collate a batch of step samples for DataLoader.

    Expects each sample dict to have:
        - input_ids: torch.Tensor of shape (L,)
        - attention_mask: torch.Tensor of shape (L,)
        - label: float

    Pads to max length in batch using torch.nn.utils.rnn.pad_sequence.

    Args:
        batch: List of sample dicts.
        pad_token_id: Token ID used for padding input_ids. Default is 0.

    Returns:
        Batched dict with padded tensors and labels.
    """
    input_ids = pad_sequence(
        [sample["input_ids"] for sample in batch],
        batch_first=True,
        padding_value=pad_token_id,
    )
    attention_mask = pad_sequence(
        [sample["attention_mask"] for sample in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = torch.tensor([sample["label"] for sample in batch], dtype=torch.float32)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }
