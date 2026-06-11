"""Step-level reward head (MLP or linear projection).

Supports multiple activation functions for architecture ablation control
(required by expert panel P1-1: verify CD-SPI consistency across head
architectures).

Available activations:
    - "relu": ReLU (default, standard)
    - "gelu": GELU (smooth, commonly used in modern LLMs)
    - "identity": No activation (linear probe baseline)
"""

import torch
import torch.nn as nn


def _get_activation(name: str) -> nn.Module:
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "identity":
        return nn.Identity()
    else:
        raise ValueError(f"Unknown activation: {name}. Options: relu, gelu, identity")


class PRMHead(nn.Module):
    """Lightweight head that maps hidden states to scalar step rewards.

    Architecture:
        hidden_state (D,) -> Linear(D, H) -> [Activation] -> Linear(H, 1) -> scalar reward

    Activation ablation: relu (default), gelu, or identity (linear probe).
    """

    def __init__(
        self,
        hidden_dim: int,
        head_dim: int = 256,
        activation: str = "relu",
    ) -> None:
        """Initialize PRM head.

        Args:
            hidden_dim: Backbone hidden dimension.
            head_dim: Intermediate MLP dimension.
            activation: Activation function. One of "relu", "gelu", "identity".
        """
        super().__init__()
        self.hidden_dim = hidden_dim
        self.head_dim = head_dim
        self.activation_name = activation
        # Keep layers as named attributes (not Sequential) so anchor-alignment
        # can address mlp1 / mlp2 individually for permutation rebasin.
        self.mlp1 = nn.Linear(hidden_dim, head_dim)
        self.act = _get_activation(activation)
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
        return self.mlp2(self.act(self.mlp1(hidden_states))).squeeze(-1)

    def get_intermediate(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return post-activation intermediate features of the head.

        These are the head_dim-wide features used by Anchor-PRM for
        permutation-based cross-client alignment (commutes with activation).

        Args:
            hidden_states: Tensor of shape (B, L, D) or (B, D).

        Returns:
            Post-activation features of shape (B, head_dim).
        """
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]
        return self.act(self.mlp1(hidden_states))
