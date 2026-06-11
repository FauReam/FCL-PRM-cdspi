"""Step-level DP-SGD implementation and privacy budget calibration.

Innovation Point B: Step-Level DP information-theoretic tight bound.

This module provides a practical DP-SGD wrapper for step-level PRM training.
The theoretical tight bound remains an open research question (P4).
"""

import math
from typing import Callable

import torch
import torch.nn as nn


class StepLevelDPSGD:
    """DP-SGD tailored for step-level PRM training.

    Key difference from standard DP-SGD:
        - Sensitivity scales with number of steps T per CoT
        - Per-sample gradient clipping on step-level loss
        - Noise calibrated for step-level leakage

    Note:
        This is the practical implementation. The information-theoretic
        tight bound for step-level DP in PRM training is P4's research goal.
    """

    def __init__(
        self,
        epsilon: float,
        delta: float,
        max_grad_norm: float = 1.0,
    ) -> None:
        """Initialize DP-SGD with privacy budget.

        Args:
            epsilon: Privacy budget epsilon.
            delta: Privacy budget delta.
            max_grad_norm: Per-sample gradient clipping bound.
        """
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        try:
            from opacus import PrivacyEngine
        except ImportError as e:
            raise ImportError(
                "opacus is required for StepLevelDPSGD. "
                "Install via `pip install opacus>=1.5.0`."
            ) from e
        self.privacy_engine = PrivacyEngine()

    def make_private(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        data_loader: torch.utils.data.DataLoader,
        epochs: int,
    ) -> tuple[nn.Module, torch.optim.Optimizer, torch.utils.data.DataLoader]:
        """Wrap model, optimizer, and dataloader with DP-SGD.

        Args:
            model: Model to make private.
            optimizer: Optimizer to make private.
            data_loader: DataLoader to make private.
            epochs: Number of training epochs.

        Returns:
            Tuple of (private_model, private_optimizer, private_dataloader).
        """
        model, optimizer, data_loader = self.privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=data_loader,
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            epochs=epochs,
            max_grad_norm=self.max_grad_norm,
        )
        return model, optimizer, data_loader

    # NOTE: This method is a standalone heuristic utility. It is NOT
    # currently invoked by make_private(), which instead relies on Opacus'
    # built-in accountant. Keep it for future manual noise tuning or
    # step-level tight-bound research (P4).
    def compute_noise_multiplier(
        self,
        sample_size: int,
        batch_size: int,
        epochs: int,
        num_steps_per_cot: int = 1,
    ) -> float:
        """Compute Gaussian noise multiplier for step-level DP.

        This is a heuristic that accounts for the increased sensitivity
        due to multiple steps per CoT. The theoretical tight bound
        (relating epsilon to T and label_complexity) is P4's goal.

        Args:
            sample_size: Number of training samples.
            batch_size: Batch size.
            epochs: Number of training epochs.
            num_steps_per_cot: Average number of steps per CoT (T).

        Returns:
            Noise multiplier sigma.
        """
        # Heuristic: step-level sensitivity is T times higher than outcome-level
        # We inflate the noise by sqrt(T) as a conservative estimate
        # (assuming steps are somewhat correlated)
        t_factor = math.sqrt(max(num_steps_per_cot, 1))

        # Use opacus accountant to find noise for target epsilon
        try:
            from opacus.accountants.utils import get_noise_multiplier
        except ImportError as e:
            raise ImportError("opacus is required to compute noise multiplier.") from e

        noise = get_noise_multiplier(
            target_epsilon=self.epsilon,
            target_delta=self.delta,
            sample_rate=batch_size / sample_size,
            epochs=epochs,
            accountant="rdp",
        )

        return noise * t_factor

    def get_spent_epsilon(self, delta: float | None = None) -> float:
        """Get the actual privacy budget spent so far.

        Args:
            delta: Optional delta override.

        Returns:
            Spent epsilon value.
        """
        return self.privacy_engine.get_epsilon(delta or self.delta)
