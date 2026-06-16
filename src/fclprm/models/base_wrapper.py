"""LLM backbone wrapper with frozen weights + trainable PRM head.

Supports:
  - Standard residual connections and Block Attention Residuals
    (AttnRes, Kimi Team arXiv:2603.15031, 2026).
  - LoRA (Low-Rank Adaptation) via peft for parameter-efficient FT.
  - Partial FT (last-N-layers) for capacity-continuum baselines.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from fclprm.models.attnres_backbone import AttnResBackboneModel
from fclprm.models.prm_head import PRMHead

logger = logging.getLogger(__name__)

# Number of trainable parameters (informational only)
_TRAINABLE_PARAMS: dict[str, int] = {}


def _count_trainable(module: nn.Module, label: str) -> int:
    n = sum(p.numel() for p in module.parameters() if p.requires_grad)
    _TRAINABLE_PARAMS[label] = n
    return n


def get_trainable_param_count(label: str) -> int:
    return _TRAINABLE_PARAMS.get(label, 0)


class StepRewardModel(nn.Module):
    """Wrapper: backbone (vanilla / AttnRes / LoRA / partial-FT) + PRM head.

    Forward pass:
        step_tokens -> backbone(last_hidden_state) -> last-non-PAD token -> PRMHead -> reward

    Training modes (mutually informative, not mutually exclusive):
      - head-only:          freeze_backbone=True  (256-dim MLP only)
      - LoRA:               lora_config={r, alpha, ...}  (low-rank adapter)
      - partial-FT:         partial_ft_layers=N  (unfreeze last N layers)
      - full-FT:            freeze_backbone=False  (all backbone + head)
      - full-FT + AttnRes:  freeze_backbone=False, attnres={...}

    The capacity continuum enables fair comparison across the parameter-efficiency
    spectrum, rebutting the "false dichotomy" critique identified by the expert panel.
    """

    def __init__(
        self,
        backbone: PreTrainedModel,
        head_dim: int = 256,
        freeze_backbone: bool = True,
        attnres: Optional[dict] = None,
        lora_config: Optional[dict] = None,
        partial_ft_layers: int = 0,
        partial_ft_mode: str = "last_n",
        head_activation: str = "relu",
    ) -> None:
        """Initialize wrapper.

        Args:
            backbone: Pre-trained LLM (e.g., Pythia 1.4B, LLaMA-3.1 8B).
                When attnres is enabled, this is wrapped in AttnResBackboneModel.
            head_dim: PRM head intermediate dimension.
            freeze_backbone: If True, backbone params are frozen (head-only FT);
                if False, full-parameter fine-tuning. Overridden by lora_config
                and partial_ft_layers for finer-grained control.
            attnres: Optional dict with AttnRes configuration:
                - num_blocks (int): Number of AttnRes blocks, default 8.
                - zero_init (bool): Init pseudo-queries to zero, default True.
                If None, standard residuals are used.
            lora_config: Optional dict for PEFT LoRA configuration:
                - r (int): LoRA rank, default 8.
                - alpha (int): LoRA alpha, default 16.
                - dropout (float): LoRA dropout, default 0.05.
                - target_modules (list[str]): Module name patterns to apply LoRA.
                  Default: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"].
                If None, LoRA is not applied.
            partial_ft_layers: Number of final transformer layers to unfreeze.
                0 means no partial unfreezing. When >0 and freeze_backbone=True,
                the last N layers are unfrozen while earlier layers stay frozen.
            partial_ft_mode: Module selection strategy for partial FT:
                - "last_n": Unfreeze all submodules in the last N layers (default).
                - "mlp_only": Unfreeze only MLP sublayers in the last N layers.
                - "attn_only": Unfreeze only attention sublayers in the last N layers.
                - "all_except_embed": Unfreeze everything except embedding layer
                  (ignores partial_ft_layers).
            head_activation: PRM head activation function for architecture ablation.
                One of "relu" (default), "gelu", "identity". Used in P1-1 control
                experiment to verify CD-SPI consistency across head architectures.
        """
        super().__init__()
        self.freeze_backbone = freeze_backbone
        self.attnres_config = attnres
        self.lora_config = lora_config
        self.partial_ft_layers = partial_ft_layers
        self.partial_ft_mode = partial_ft_mode
        self.head_activation = head_activation
        self._lora_applied = False

        # Wrap backbone with AttnRes if enabled
        if attnres is not None:
            num_blocks = attnres.get("num_blocks", 8)
            zero_init = attnres.get("zero_init", True)
            logger.info(
                f"[StepRewardModel] Enabling Block AttnRes: "
                f"{num_blocks} blocks, zero_init={zero_init}, "
                f"model={backbone.__class__.__name__}"
            )
            self.backbone = AttnResBackboneModel(
                backbone=backbone,
                num_blocks=num_blocks,
                zero_init=zero_init,
            )
        else:
            self.backbone = backbone

        # --- LoRA: apply PEFT adapters to backbone ---
        if lora_config is not None:
            self._apply_lora(lora_config)

        # --- Partial FT: unfreeze selected layers/modules ---
        if partial_ft_layers > 0:
            self._apply_partial_ft(partial_ft_layers, mode=partial_ft_mode)

        # --- Freeze logic ---
        # If LoRA is applied, peft handles freezing non-adapter params.
        # If partial_ft_layers > 0, those layers were already unfrozen above.
        # Otherwise, apply the global freeze_backbone flag.
        if freeze_backbone and not self._lora_applied and partial_ft_layers == 0:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.head = PRMHead(
            hidden_dim=backbone.config.hidden_size,
            head_dim=head_dim,
            activation=head_activation,
        )

        _count_trainable(self, "total")
        n_backbone = sum(
            p.numel() for p in self.backbone.parameters() if p.requires_grad
        )
        n_head = sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        logger.info(
            f"[StepRewardModel] Trainable params: {n_backbone:,} backbone + "
            f"{n_head:,} head = {n_backbone + n_head:,} total"
        )

    def _apply_lora(self, cfg: dict) -> None:
        """Apply LoRA adapters to the backbone via peft."""
        try:
            from peft import LoraConfig, get_peft_model, TaskType
        except ImportError:
            raise ImportError(
                "peft is required for LoRA training. Install via `pip install peft`."
            )

        r = cfg.get("r", 8)
        alpha = cfg.get("alpha", 16)
        dropout = cfg.get("dropout", 0.05)
        target_modules = cfg.get(
            "target_modules",
            ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
        )

        # When AttnRes is active, the backbone is wrapped in AttnResBackboneModel.
        # LoRA must be applied to the inner backbone (HF model), not the wrapper.
        inner = self.backbone.backbone if self.attnres_config is not None else self.backbone

        peft_config = LoraConfig(
            task_type=TaskType.FEATURE_EXTRACTION,
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            target_modules=target_modules,
        )
        lora_backbone = get_peft_model(inner, peft_config)

        if self.attnres_config is not None:
            self.backbone.backbone = lora_backbone
        else:
            self.backbone = lora_backbone

        self._lora_applied = True
        logger.info(
            f"[StepRewardModel] LoRA applied: r={r}, alpha={alpha}, "
            f"target_modules={target_modules}"
        )

    def _apply_partial_ft(self, n_layers: int, mode: str = "last_n") -> None:
        """Unfreeze selected transformer layer parameters based on mode.

        The target for layer access is the inner HF backbone when AttnRes is
        active, otherwise self.backbone directly.

        Modes:
          - last_n:          Unfreeze all submodules in the last N layers (default).
          - mlp_only:        Unfreeze only MLP sublayers in the last N layers.
          - attn_only:       Unfreeze only attention sublayers in the last N layers.
          - all_except_embed: Unfreeze everything except embedding layer
                              (ignores n_layers/sets backbone fully trainable).
        """
        inner = self.backbone.backbone if self.attnres_config is not None else self.backbone

        # Mode: all_except_embed — freeze only embed, unfreeze everything else
        if mode == "all_except_embed":
            for param in inner.parameters():
                param.requires_grad = True
            for name, param in inner.named_parameters():
                if "embed" in name:
                    param.requires_grad = False
            logger.info(
                "[StepRewardModel] Partial FT: mode=all_except_embed — "
                "unfroze all backbone except embeddings"
            )
            return

        if not hasattr(inner, "layers") and hasattr(inner, "gpt_neox"):
            layers = inner.gpt_neox.layers
        elif hasattr(inner, "model") and hasattr(inner.model, "layers"):
            layers = inner.model.layers
        elif hasattr(inner, "layers"):
            layers = inner.layers
        else:
            logger.warning(
                "[StepRewardModel] Cannot identify transformer layers for partial FT; "
                "skipping."
            )
            return

        total_layers = len(layers)
        n_unfreeze = min(n_layers, total_layers)
        start_idx = total_layers - n_unfreeze

        # Mode: last_n — existing behaviour (unfreeze all submodules in selected layers)
        if mode == "last_n":
            for i, layer in enumerate(layers):
                for param in layer.parameters():
                    param.requires_grad = (i >= start_idx)
            logger.info(
                f"[StepRewardModel] Partial FT: mode=last_n, unfroze last "
                f"{n_unfreeze}/{total_layers} layers"
            )
            return

        # Modes that require selective unfreezing: first freeze all backbone params,
        # then selectively unfreeze within the selected layer range.
        for param in inner.parameters():
            param.requires_grad = False

        if mode == "mlp_only":
            unfrozen = 0
            for i in range(start_idx, total_layers):
                for name, param in layers[i].named_parameters():
                    if "mlp" in name:
                        param.requires_grad = True
                        unfrozen += 1
            logger.info(
                f"[StepRewardModel] Partial FT: mode=mlp_only, unfroze last "
                f"{n_unfreeze}/{total_layers} layers ({unfrozen} MLP params)"
            )
        elif mode == "attn_only":
            unfrozen = 0
            for i in range(start_idx, total_layers):
                for name, param in layers[i].named_parameters():
                    if "attention" in name:
                        param.requires_grad = True
                        unfrozen += 1
            logger.info(
                f"[StepRewardModel] Partial FT: mode=attn_only, unfroze last "
                f"{n_unfreeze}/{total_layers} layers ({unfrozen} attn params)"
            )
        else:
            logger.warning(
                f"[StepRewardModel] Unknown partial_ft_mode='{mode}'; "
                f"falling back to last_n behaviour"
            )
            for i, layer in enumerate(layers):
                for param in layer.parameters():
                    param.requires_grad = (i >= start_idx)

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
        # Frozen backbone: skip autograd graph construction entirely.
        # torch.no_grad() is faster than relying on requires_grad=False alone
        # because it avoids registering backward hooks on every op.
        backbone_needs_grad = (
            not self.freeze_backbone
            or self._lora_applied
            or self.partial_ft_layers > 0
        )
        # Always use autocast: the frozen-backbone path was previously
        # running in FP32 (torch.no_grad disables autograd, not dtype
        # casting), wasting ~2x compute and memory bandwidth vs BF16.
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            grad_ctx = torch.no_grad() if not backbone_needs_grad else torch.enable_grad()
            with grad_ctx:
                outputs = self.backbone(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
        last_hidden = self._last_non_pad_hidden(
            outputs.last_hidden_state, attention_mask
        )
        # Head weights are fp32, backbone output is bf16.
        # .float() cast is cheap (no copy needed if tensor is contiguous).
        return self.head(last_hidden.float())

    def get_step_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract step embedding for CD-SPI computation.

        Returns the hidden state at the last non-PAD token position
        from the backbone's FINAL layer.

        Note: This embedding comes from the backbone output (after all transformer
        layers). For head-only configs, backbone is frozen + identical across
        clients, so this will always yield CD-SPI ~ 0. For full FT, backbone
        differs across clients so CD-SPI > 0 is possible.

        For symmetrical measurement across configs, use get_backbone_embedding()
        instead, which extracts from the penultimate layer.

        Args:
            input_ids: Token IDs of shape (B, L).
            attention_mask: Attention mask of shape (B, L).

        Returns:
            Step embeddings of shape (B, D).
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        return self._last_non_pad_hidden(outputs.last_hidden_state, attention_mask)

    def get_backbone_embedding(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Extract embedding from backbone's SECOND-TO-LAST layer.

        This enables symmetrical CD-SPI measurement across head-only and full-FT
        configurations: both use the same extraction point (backbone penultimate
        layer hidden state), avoiding the embedding-space mismatch that occurs
        when using get_head_embedding() (which goes through randomly initialized
        head weights for head-only vs. trained head+backbone for full FT).

        Implementation uses output_hidden_states=True and takes hidden_states[-2],
        which works across all HuggingFace model architectures without requiring
        architecture-specific layer hooks.

        Returns:
            Penultimate-layer hidden states of shape (B, D) at last non-PAD position.
        """
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            outputs = self.backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
        # hidden_states is a tuple of (embed, layer1, ..., layerN)
        # hidden_states[-1] is the final layer output (= last_hidden_state)
        # hidden_states[-2] is the penultimate layer
        penultimate = outputs.hidden_states[-2]
        return self._last_non_pad_hidden(penultimate, attention_mask)

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
            outputs = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask
            )
            last_hidden = self._last_non_pad_hidden(
                outputs.last_hidden_state, attention_mask
            )
        return self.head.get_intermediate(last_hidden.float())
