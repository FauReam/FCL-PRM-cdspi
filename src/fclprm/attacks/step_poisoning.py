"""Step-level label poisoning attacks on federated PRM training."""

import random
from typing import Literal


class StepPoisoningAttack:
    """Poison step-level labels to degrade global PRM performance.

    Attack strategies:
        - flip: Reverse step labels (correct -> incorrect, etc.)
        - scale: Multiply rewards by a scaling factor
        - targeted: Poison specific step types (e.g., logical connectors)
    """

    def __init__(
        self,
        attack_type: Literal["flip", "scale", "targeted"],
        poison_rate: float = 0.1,
        scale_factor: float = -1.0,
        target_keywords: list[str] | None = None,
    ) -> None:
        """Initialize attack.

        Args:
            attack_type: Type of poisoning attack.
            poison_rate: Fraction of steps to poison.
            scale_factor: Scaling factor for 'scale' attack (default -1 flips sign).
            target_keywords: Keywords for 'targeted' attack.
        """
        self.attack_type = attack_type
        self.poison_rate = poison_rate
        self.scale_factor = scale_factor
        self.target_keywords = target_keywords or ["therefore", "because", "so", "thus"]

    def poison(self, steps: list[str], labels: list[float]) -> list[float]:
        """Apply poisoning to step labels.

        Args:
            steps: List of step texts.
            labels: Original step labels in [0, 1].

        Returns:
            Poisoned labels.
        """
        poisoned = labels.copy()
        n_steps = len(steps)
        n_poison = int(n_steps * self.poison_rate)

        if self.attack_type == "flip":
            indices = random.sample(range(n_steps), min(n_poison, n_steps))
            for i in indices:
                poisoned[i] = 1.0 - poisoned[i]

        elif self.attack_type == "scale":
            indices = random.sample(range(n_steps), min(n_poison, n_steps))
            for i in indices:
                poisoned[i] = max(0.0, min(1.0, poisoned[i] * self.scale_factor))

        elif self.attack_type == "targeted":
            # Poison steps containing target keywords
            candidates = [
                i
                for i, step in enumerate(steps)
                if any(kw in step.lower() for kw in self.target_keywords)
            ]
            indices = random.sample(candidates, min(n_poison, len(candidates)))
            for i in indices:
                poisoned[i] = 1.0 - poisoned[i]

        return poisoned
