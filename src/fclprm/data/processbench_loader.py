"""ProcessBench evaluation dataset loader.

ProcessBench (Qwen Team, arXiv:2412.06559) measures a model's ability to
identify erroneous steps in mathematical reasoning.

Reference:
    https://arxiv.org/abs/2412.06559
    https://github.com/QwenLM/ProcessBench

Data format:
    Each sample has:
        - problem: str, the math problem
        - steps: list[str], step-by-step solution
        - label: int, index of first erroneous step (-1 if all correct)
"""

from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F


class ProcessBenchAccuracy:
    """Evaluate PRM on ProcessBench benchmark.

    Measures step-level error detection accuracy: the model must assign
    the lowest reward score to the first erroneous step.

    This is a harder evaluation than simple MSE-based metrics because it
    tests whether the model can *discriminate* correct from incorrect steps
    within the same solution, rather than just regressing label values.
    """

    def __init__(self, data_dir: str) -> None:
        """Initialize evaluator.

        Args:
            data_dir: Path to ProcessBench data directory.
        """
        self.data_dir = Path(data_dir)
        self.samples = self._load_data()

    def _load_data(self) -> list[dict]:
        """Load ProcessBench data from JSONL."""
        import json

        data_file = self.data_dir / "processbench.jsonl"
        if not data_file.exists():
            # Fall back to steps format
            data_file = self.data_dir / "processbench_steps.jsonl"
        if not data_file.exists():
            raise FileNotFoundError(
                f"ProcessBench data not found in {self.data_dir}"
            )

        samples = []
        with open(data_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        return samples

    @property
    def num_samples(self) -> int:
        return len(self.samples)

    def evaluate(
        self,
        model: torch.nn.Module,
        tokenizer,
        device: str = "cuda",
        max_length: int = 512,
        batch_size: int = 64,
    ) -> dict:
        """Evaluate model on ProcessBench.

        For each problem, scores every step and checks whether the
        lowest-scored step matches the ground-truth error step.

        Args:
            model: StepRewardModel instance.
            tokenizer: HuggingFace tokenizer.
            device: Device for inference.
            max_length: Max token length per step.
            batch_size: Batch size for scoring steps.

        Returns:
            Dict with:
                - accuracy: Fraction of problems where the lowest-scored
                  step matches the error step.
                - error_detection_accuracy: Same as accuracy, but excluding
                  "all correct" (-1) samples.
                - total: Number of problems evaluated.
        """
        model.to(device)
        model.eval()

        correct = 0
        total = 0
        error_detected = 0
        error_total = 0

        from tqdm import tqdm

        for sample in tqdm(self.samples, desc="[ProcessBench] Evaluating"):
            problem = sample["problem"]
            steps = sample.get("steps", [])
            error_step = sample.get("label", sample.get("error_step", -1))

            if not steps:
                continue

            # Score each step: question + step_text
            all_inputs = []
            for step_text in steps:
                text = f"{problem}\n{step_text}"
                all_inputs.append(text)

            # Tokenize all steps at once
            encoded = tokenizer(
                all_inputs,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded["attention_mask"].to(device)

            # Score in batches to avoid OOM on long sequences
            all_scores = []
            n_steps = len(steps)
            for i in range(0, n_steps, batch_size):
                batch_ids = input_ids[i : i + batch_size]
                batch_mask = attention_mask[i : i + batch_size]
                with torch.no_grad(), torch.autocast(
                    device_type="cuda", dtype=torch.bfloat16
                ):
                    scores = model(batch_ids, batch_mask)
                all_scores.append(scores.detach().cpu())

            scores = torch.cat(all_scores)  # (n_steps,)

            # Lowest score = predicted error step
            pred_error = int(scores.argmin())

            # If all steps are correct (label == -1), the model should
            # still predict some step as "most likely wrong" — but in
            # ProcessBench evaluation, the standard metric only counts
            # accuracy on samples with an actual error.
            if error_step == -1:
                # All-correct sample: we check if avg score > 0.5
                # (model thinks all steps are correct)
                avg_score = float(scores.mean())
                if avg_score > 0.5:
                    correct += 1
            else:
                error_total += 1
                if pred_error == error_step:
                    correct += 1
                    error_detected += 1

            total += 1

        accuracy = correct / max(total, 1)
        error_acc = error_detected / max(error_total, 1)

        return {
            "accuracy": accuracy,
            "error_detection_accuracy": error_acc,
            "total": total,
            "with_errors": error_total,
        }
