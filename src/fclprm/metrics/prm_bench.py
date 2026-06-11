"""ProcessBench / PRMBench evaluation interfaces."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F


class ProcessBenchEvaluator:
    """Evaluate PRM on ProcessBench benchmark."""

    def __init__(self, bench_dir: str) -> None:
        """Initialize evaluator.

        Args:
            bench_dir: Path to ProcessBench data.
        """
        self.bench_dir = Path(bench_dir)
        self.samples = self._load_data()

    def _load_data(self) -> list[dict]:
        """Load ProcessBench data from JSONL."""
        data_file = self.bench_dir / "test.jsonl"
        if not data_file.exists():
            data_file = self.bench_dir / "test.json"

        samples = []
        if data_file.suffix == ".jsonl":
            with open(data_file, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        samples.append(json.loads(line))
        else:
            with open(data_file, "r", encoding="utf-8") as f:
                samples = json.load(f)

        return samples

    def evaluate(
        self, model, tokenizer, device: str = "cuda", max_length: int = 512
    ) -> dict:
        """Run full ProcessBench evaluation.

        Evaluates PRM at step-level granularity: each reasoning step is
        scored independently, and metrics are computed over all steps
        across all CoT samples.

        Args:
            model: PRM model to evaluate.
            tokenizer: HuggingFace tokenizer.
            device: Device for inference.
            max_length: Max token length.

        Returns:
            Dict of step-level metrics (accuracy, f1, precision, recall, auc).
        """
        model.to(device)
        model.eval()

        all_step_preds = []
        all_step_labels = []
        all_step_scores = []

        with torch.no_grad():
            for sample in self.samples:
                steps = sample.get("steps", [])
                labels = sample.get("labels", [])

                if not steps or not labels or len(steps) != len(labels):
                    continue

                for step_text, label in zip(steps, labels):
                    encoded = tokenizer(
                        step_text,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=max_length,
                    )
                    input_ids = encoded["input_ids"].to(device)
                    attention_mask = encoded["attention_mask"].to(device)

                    score = model(input_ids, attention_mask).item()
                    pred = 1 if score > 0.5 else 0

                    all_step_preds.append(pred)
                    all_step_labels.append(int(label))
                    all_step_scores.append(score)

        n_steps = len(all_step_labels)
        if n_steps == 0:
            return {"num_steps": 0, "accuracy": 0.0}

        correct = sum(1 for p, l in zip(all_step_preds, all_step_labels) if p == l)
        metrics = {
            "num_steps": n_steps,
            "accuracy": correct / n_steps,
        }

        try:
            from sklearn.metrics import (
                f1_score,
                roc_auc_score,
                precision_score,
                recall_score,
            )

            if len(set(all_step_labels)) > 1:
                metrics["f1"] = f1_score(
                    all_step_labels, all_step_preds, zero_division=0
                )
                metrics["precision"] = precision_score(
                    all_step_labels, all_step_preds, zero_division=0
                )
                metrics["recall"] = recall_score(
                    all_step_labels, all_step_preds, zero_division=0
                )
                metrics["auc"] = roc_auc_score(all_step_labels, all_step_scores)
            else:
                metrics["f1"] = 0.0
                metrics["precision"] = 0.0
                metrics["recall"] = 0.0
                metrics["auc"] = 0.0
        except ImportError:
            pass

        return metrics
