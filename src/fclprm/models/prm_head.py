"""Step-level reward head (MLP or linear projection)."""

import torch
import torch.nn as nn


class PRMHead(nn.Module):
    """Lightweight head that maps hidden states to scalar step rewards.

    Architecture:
        hidden_state (D,) -> Linear(D, H) -> ReLU -> Linear(H, 1) -> scalar reward
    """

    def __init__(self, hidden_dim: int, head_dim: int = 256) -> None:
        """Initialize PRM head.

        Args:
            hidden_dim: Backbone hidden dimension.
            head_dim: Intermediate MLP dimension.
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        # Keep layers as named attributes (not Sequential) so anchor-alignment
        # can address mlp1 / mlp2 individually for permutation rebasin.
        self.mlp1 = nn.Linear(hidden_dim, head_dim)
        self.relu = nn.ReLU()
        self.mlp2 = nn.Linear(head_dim, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute reward from last hidden state.

        Args:
            hidden_states: Tensor of shape (B, L, D) or (B, D).

        Returns:
            Scalar rewards of shape (B,).
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]
        return self.mlp2(self.relu(self.mlp1(hidden_states))).squeeze(-1)

    def get_intermediate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return post-ReLU intermediate activations of the head.

        These are the head_dim-wide features used by Anchor-PRM for
        permutation-based cross-client alignment (commutes with ReLU).

        Args:
            hidden_states: Tensor of shape (B, L, D) or (B, D).

        Returns:
            Post-ReLU activations of shape (B, head_dim).
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]
        return self.relu(self.mlp1(hidden_states))
