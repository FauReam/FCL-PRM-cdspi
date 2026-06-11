"""MedPRMBench medical reasoning dataset loader.

Reference:
    MedPRMBench (arXiv:2604.17282, 2026)

Data format (JSONL):
    Each line is a dict with keys:
        - question: str, clinical scenario
        - steps: List[str], clinical reasoning steps
        - labels: List[int], step correctness labels
"""

from pathlib import Path
from typing import Optional

from fclprm.data.utils import _load_jsonl_or_json, _normalize_dataset


class MedPRMBenchLoader:
    """Load medical PRM data with clinical CoT and step labels."""

    def __init__(self, data_dir: str) -> None:
        """Initialize loader.

        Args:
            data_dir: Path to MedPRMBench data.
        """
        self.data_dir = Path(data_dir)
        self._data: Optional[list[dict]] = None

    def load(self) -> list[dict]:
        """Load medical reasoning samples.

        Returns:
            List of dicts with clinical CoT steps and labels.

        Raises:
            FileNotFoundError: If data file is not found.
        """
        if self._data is not None:
            return self._data

        samples = _load_jsonl_or_json(self.data_dir, "med_prm")
        samples = _normalize_dataset(samples)

        self._data = samples
        return samples
