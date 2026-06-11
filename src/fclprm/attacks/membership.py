"""Membership inference attacks on chain-of-thought data."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MembershipInferenceAttack:
    """Determine if a specific CoT was in the training data.

    Reference: "When Reasoning Leaks Membership" (arXiv:2601.13607, 2026)
    """

    def __init__(self, shadow_model: nn.Module, device: str = "cuda") -> None:
        """Initialize attack with shadow model.

        Args:
            shadow_model: Shadow model trained on similar data.
            device: Device for computation.
        """
        self.shadow_model = shadow_model.to(device)
        self.device = device

    def infer(
        self,
        target_model: nn.Module,
        sample: dict,
        method: str = "loss",
    ) -> float:
        """Compute membership score for a sample.

        Args:
            target_model: Target model to attack.
            sample: CoT sample to test with keys:
                - input_ids: token IDs
                - attention_mask: attention mask
                - label: ground truth label
            method: "loss" or "confidence"

        Returns:
            Membership probability (higher = more likely member).
        """
        target_model.to(self.device)
        target_model.eval()

        input_ids = sample["input_ids"].unsqueeze(0).to(self.device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(self.device)
        label = torch.tensor([sample["label"]], device=self.device)

        with torch.no_grad():
            pred = target_model(input_ids, attention_mask)

            if method == "loss":
                loss = F.mse_loss(pred, label).item()
                # Lower loss -> higher membership likelihood
                return max(0.0, 1.0 - loss)
            elif method == "confidence":
                # For binary-like rewards, confidence = proximity to 0 or 1
                confidence = max(pred.item(), 1.0 - pred.item())
                return confidence
            else:
                raise ValueError(f"Unknown inference method: {method}")

    def infer_batch(
        self,
        target_model: nn.Module,
        samples: list[dict],
        method: str = "loss",
    ) -> list[float]:
        """Compute membership scores for a batch of samples.

        Args:
            target_model: Target model to attack.
            samples: List of CoT samples.
            method: "loss" or "confidence"

        Returns:
            List of membership scores.
        """
        return [self.infer(target_model, sample, method) for sample in samples]
