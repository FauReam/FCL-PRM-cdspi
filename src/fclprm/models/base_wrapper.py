"""LLM backbone wrapper with frozen weights + trainable PRM head."""

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from fclprm.models.prm_head import PRMHead


class StepRewardModel(nn.Module):
    """Wrapper: frozen or trainable LLM backbone + trainable PRM head.

    Forward pass:
        step_tokens -> backbone(last_hidden_state) -> last-non-PAD token -> PRMHead -> reward

    When freeze_backbone=True (default), only the PRMHead (256-dim MLP) is trained.
    When freeze_backbone=False, both backbone and head are fine-tuned end-to-end.
    The latter requires ~21 GB memory on Pythia-1.4B (BF16+Adam), feasible only
    on unified-memory devices like NVIDIA GB10.
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        head_dim: int = 256,
        freeze_backbone: bool = True,
    ) -> None:
        """Initialize wrapper.

        Args:
            backbone: Pre-trained LLM (e.g., Pythia 1.4B, LLaMA-3.1 8B).
            head_dim: PRM head intermediate dimension.
            freeze_backbone: If True, backbone params are frozen (head-only FT);
                if False, full-parameter fine-tuning.
        """
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        self.head = PRMHead(
            hidden_dim=backbone.config.hidden_size,
            head_dim=head_dim,
        )

    @staticmethod
    def _last_non_pad_hidden(
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Pick hidden state at the last position where attention_mask == 1.
        # Avoids using PAD token's hidden state, which would silently corrupt
        # CD-SPI measurements and PRM rewards on right-padded batches.
        seq_lens = attention_mask.sum(dim=1).clamp(min=1) - 1  # (B,)
        batch_idx = torch.arange(hidden_states.size(0), device=hidden_states.device)
        return hidden_states[batch_idx, seq_lens]

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Compute step reward.

        Args:
            input_ids: Token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L), 1 for tokens, 0 for PAD.

        Returns:
            Scalar rewards of shape (B,).
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = self._last_non_pad_hidden(
                outputs.last_hidden_state, attention_mask
            )
        return self.head(last_hidden)

    def get_step_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract step embedding for CD-SPI computation.

        Returns the hidden state at the last non-PAD token position.

        Args:
            input_ids: Token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).

        Returns:
            Step embeddings of shape (B, D).
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self._last_non_pad_hidden(outputs.last_hidden_state, attention_mask)

    def get_head_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract head-intermediate embedding for Anchor-PRM alignment.

        Differs from get_step_embedding: this passes the backbone hidden state
        through the trainable mlp1 + ReLU layer of the PRM head, so the
        resulting features actually depend on client-specific head weights
        (the backbone is frozen and identical across clients).

        Args:
            input_ids: Token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).

        Returns:
            Post-ReLU head features of shape (B, head_dim).
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = self._last_non_pad_hidden(
                outputs.last_hidden_state, attention_mask
            )
        return self.head.get_intermediate(last_hidden)
