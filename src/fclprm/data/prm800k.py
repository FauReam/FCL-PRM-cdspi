"""PRM800K dataset loader with step-level reward labels.

Reference:
    Lightman et al. "Let's Verify Step by Step" (2023)
    https://arxiv.org/abs/2305.20050

Data format (JSONL):
    Each line is a dict with keys:
        - question: str, the problem statement
        - steps: List[str], chain-of-thought steps
        - labels: List[int], per-step labels (+1=correct, -1=incorrect)
"""

from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from fclprm.data.utils import (
    _load_jsonl_or_json,
    _normalize_dataset,
    split_cot_into_steps,
)


class PRM800KLoader:
    """Load and preprocess PRM800K data.

    Each sample contains:
        - question: str, the problem statement
        - steps: List[str], chain-of-thought steps
        - labels: List[float], per-step reward labels
    """

    def __init__(self, data_dir: str, split: str = "train") -> None:
        """Initialize loader.

        Args:
            data_dir: Path to PRM800K data directory.
            split: Data split, one of "train", "val", "test".
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self._data: Optional[list[dict]] = None

    @staticmethod
    def _parse_sample(raw: dict) -> dict | None:
        """Normalize a single PRM800K record to {question, steps, labels}.

        Handles both the flat format (used in tests and synthetic data) and
        the nested OpenAI format (question.problem, question.pre_generated_steps,
        label.steps[].completions[].rating).

        Returns:
            Normalized sample dict, or None if the record cannot be parsed.
        """
        # Already flat — nothing to do
        if "steps" in raw and "labels" in raw:
            return raw

        # Nested OpenAI format
        question_block = raw.get("question", {})
        label_block = raw.get("label", {})

        if not question_block or not label_block:
            return None

        question = question_block.get("problem", "")
        steps = question_block.get("pre_generated_steps", [])
        step_labels = label_block.get("steps", [])

        if not steps or not step_labels or len(steps) != len(step_labels):
            return None

        labels = []
        for step_label in step_labels:
            completions = step_label.get("completions", [])
            chosen = step_label.get("chosen_completion")

            if not completions:
                labels.append(0.0)
                continue

            if chosen is not None and 0 <= chosen < len(completions):
                rating = completions[chosen].get("rating", 0)
            else:
                # Fall back to first completion if chosen is absent
                rating = completions[0].get("rating", 0)

            # Map -1/0/+1 to 0.0/0.5/1.0
            if rating == 1:
                labels.append(1.0)
            elif rating == -1:
                labels.append(0.0)
            else:
                labels.append(0.5)

        return {"question": question, "steps": steps, "labels": labels}

    def load(self) -> list[dict]:
        """Load raw samples from disk.

        Returns:
            List of dicts with keys: question, steps, labels.

        Raises:
            FileNotFoundError: If data file is not found.
        """
        if self._data is not None:
            return self._data

        raw_samples = _load_jsonl_or_json(self.data_dir, self.split)
        samples = []
        for raw in raw_samples:
            parsed = self._parse_sample(raw)
            if parsed is not None:
                samples.append(parsed)

        samples = _normalize_dataset(samples)

        self._data = samples
        return samples

    def tokenize_steps(
        self,
        steps: list[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ) -> list[dict]:
        """Tokenize each step into model input format.

        Args:
            steps: List of step strings.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum token length per step.

        Returns:
            List of tokenized step dicts with input_ids and attention_mask.
        """
        encoded = tokenizer(
            steps,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_tensors=None,
        )
        return [
            {
                "input_ids": encoded["input_ids"][i],
                "attention_mask": encoded["attention_mask"][i],
            }
            for i in range(len(steps))
        ]

    def build_step_dataset(
        self,
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
    ) -> list[dict]:
        """Build flat step-level dataset from CoT samples.

        Each sample in the output corresponds to one step (not one full CoT).
        This is the format needed for PRM training.

        Args:
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum token length per step.

        Returns:
            List of dicts with keys: input_ids, attention_mask, label.
        """
        samples = self.load()
        step_samples = []

        for sample in tqdm(samples, desc="[PRM800K] Tokenizing steps"):
            question = sample.get("question", "")
            steps = sample.get("steps", [])
            labels = sample.get("labels", [])

            if len(steps) != len(labels):
                continue

            # Prepend question context to each step
            for step_text, label in zip(steps, labels):
                text = f"{question}\n{step_text}"
                encoded = tokenizer(
                    text,
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None,
                )
                step_samples.append(
                    {
                        "input_ids": torch.tensor(
                            encoded["input_ids"], dtype=torch.long
                        ),
                        "attention_mask": torch.tensor(
                            encoded["attention_mask"], dtype=torch.long
                        ),
                        "label": float(label),
                    }
                )

        return step_samples
