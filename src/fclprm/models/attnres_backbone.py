"""Block Attention Residuals (Block AttnRes) backbone wrapper.

Reference: "Attention Residuals" (Kimi Team, arXiv:2603.15031, Mar 2026)
https://arxiv.org/abs/2603.15031

Replaces standard PreNorm residual connections:
    h_l = h_{l-1} + f_{l-1}(h_{l-1})

With learned softmax attention over block-level representations:
    h_l = sum_{i=0}^{l-1} alpha_{i->l} * v_i
    alpha_{i->l} = softmax(w_l^T * RMSNorm(v_i))

Standard residuals accumulate all layer outputs with fixed unit weights,
causing uncontrolled hidden-state growth and progressive dilution of each
layer's contribution (PreNorm dilution). AttnRes gives each layer selective,
content-aware access to all earlier representations via a single learned
pseudo-query vector w_l in R^d.

Block AttnRes partitions layers into N blocks to reduce memory from O(Ld)
to O(Nd). Within each block, outputs accumulate via standard residuals;
across blocks, softmax attention selects among block-level summaries.
With N ≈ 8 blocks, Block AttnRes recovers most of the gain of Full AttnRes.

Integration in FCL-PRM:
    StepRewardModel(backbone=AttnResBackboneModel(hf_backbone, num_blocks=8))
    The PRM head remains unchanged; only the backbone's residual pathway is
    modified. Full-parameter FT trains both the backbone and pseudo-query
    vectors; head-only FT trains only the PRM head (pseudo-queries frozen).
"""

from typing import Optional

import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast


class BlockAttnRes(nn.Module):
    """Single block attention residual computation.

    Attends over completed block representations + current partial block sum
    using a per-(sub)layer learned pseudo-query vector w_l.

    For each (sub)layer, the input is:
        h_l = sum_{i=0}^{N} alpha_{i->l} * v_i

    Where v_i are block-level representations (completed blocks + partial sum)
    and alpha_{i->l} = softmax(w_l^T * RMSNorm(v_i)).
    """

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.key_norm = nn.RMSNorm(hidden_dim)

    def forward(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        pseudo_query: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention residual.

        Args:
            blocks: List of [B, T, D] completed block representations.
            partial_block: [B, T, D] intra-block partial sum (b_n^i in paper).
            pseudo_query: [D] learned query vector w_l for this (sub)layer.

        Returns:
            [B, T, D] aggregated hidden state.
        """
        # Stack all sources: completed blocks + current partial sum.
        # Shape: [N_src, B, T, D] where N_src = len(blocks) + 1
        V = torch.stack(blocks + [partial_block], dim=0)
        K = self.key_norm(V)

        # Compute logits: einsum('d, n b t d -> n b t', w_l, K)
        logits = torch.einsum("d, n b t d -> n b t", pseudo_query, K)

        # Softmax over source dimension (depth), then aggregate.
        attn_weights = torch.softmax(logits, dim=0)  # [N_src, B, T]
        h = torch.einsum("n b t, n b t d -> b t d", attn_weights, V)
        return h


class AttnResBackboneModel(nn.Module):
    """Backbone wrapper implementing Block Attention Residuals.

    Wraps a HuggingFace GPTNeoX-like model (Pythia) and replaces its
    standard PreNorm residual connections with Block AttnRes.

    Architecture per transformer layer:
        1. Pre-ATTN AttnRes: attend over blocks + partial sum
        2. Block boundary check: save partial sum as new block if at boundary
        3. Self-attention: attn(PreNorm(h))
        4. Partial accumulation: partial += attn_out
        5. Pre-MLP AttnRes: attend over blocks + partial sum
        6. MLP: mlp(PreNorm(h))
        7. Partial accumulation: partial += mlp_out

    The backbone's original parameters (embeddings, attention, MLP) remain
    trainable or frozen as configured by StepRewardModel. The pseudo-query
    vectors w_l are always trained during full-parameter FT.

    Args:
        backbone: Pre-trained HuggingFace model (GPTNeoXModel or similar).
        num_blocks: Number of AttnRes blocks (N). Default 8.
            Each block contains num_layers / num_blocks transformer layers.
            The token embedding is an additional implicit source (b_0).
        zero_init: Initialize pseudo-queries to zero. Default True.
            Zero init ensures attention is uniform at start of training,
            matching standard residual behavior initially.
    """

    # Supported backbone architectures and their layer accessor paths.
    # Key: HuggingFace model class name, Value: (layers_attr, embed_attr, final_norm_attr)
    SUPPORTED_ARCHS = {
        "GPTNeoXForCausalLM": ("gpt_neox.layers", "gpt_neox.embed_in", "gpt_neox.final_layer_norm"),
        "GPTNeoXModel": ("layers", "embed_in", "final_layer_norm"),
    }

    def __init__(
        self,
        backbone: PreTrainedModel,
        num_blocks: int = 8,
        zero_init: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config
        hidden_size = backbone.config.hidden_size
        num_layers = backbone.config.num_hidden_layers
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # --- Architecture detection ---
        model_class = backbone.__class__.__name__
        if model_class in self.SUPPORTED_ARCHS:
            layers_attr, embed_attr, norm_attr = self.SUPPORTED_ARCHS[model_class]
            self._layers = self._resolve_attr(backbone, layers_attr)
            self._embed_in = self._resolve_attr(backbone, embed_attr)
            self._final_norm = self._resolve_attr(backbone, norm_attr)
        elif hasattr(backbone, "gpt_neox"):
            # Fallback: GPTNeoXForCausalLM-like
            self._layers = backbone.gpt_neox.layers
            self._embed_in = backbone.gpt_neox.embed_in
            self._final_norm = getattr(backbone.gpt_neox, "final_layer_norm", None)
        elif hasattr(backbone, "layers"):
            # Generic: try direct layers access (GPTNeoXModel, LlamaModel, etc.)
            self._layers = backbone.layers
            self._embed_in = backbone.get_input_embeddings()
            self._final_norm = getattr(backbone, "norm", None) or getattr(backbone, "final_layer_norm", None)
        else:
            raise ValueError(
                f"Unsupported backbone architecture: {model_class}. "
                f"AttnRes currently supports GPTNeoX (Pythia). "
                f"Expected backbone with `.layers` attribute."
            )

        # --- Block configuration ---
        # Each transformer layer has 2 sub-layers (ATTN + MLP).
        # block_size in transformer layers = how many layers per block.
        self.block_t_layers = max(1, num_layers // max(1, num_blocks))
        self.num_blocks_actual = num_layers // self.block_t_layers
        self.num_blocks = num_blocks

        # --- Pseudo-query vectors ---
        # One w_l per sub-layer: [2 * num_layers, hidden_size]
        # Even indices: pre-ATTN queries; Odd indices: pre-MLP queries.
        init_val = 0.0 if zero_init else 0.02
        self.pseudo_queries = nn.Parameter(
            torch.full((2 * num_layers, hidden_size), init_val, dtype=torch.float32)
        )

        # --- Shared key normalization (RMSNorm over source representations) ---
        self.key_norm = nn.RMSNorm(hidden_size)

    @staticmethod
    def _resolve_attr(obj, path: str):
        """Resolve a dot-separated attribute path on an object."""
        for attr in path.split("."):
            obj = getattr(obj, attr)
        return obj

    def _block_attn_res(
        self,
        blocks: list[torch.Tensor],
        partial_block: torch.Tensor,
        layer_idx: int,
        is_mlp: bool = False,
    ) -> torch.Tensor:
        """Apply Block AttnRes for a single (sub)layer.

        Args:
            blocks: Completed block representations (list of [B, T, D]).
            partial_block: Current intra-block partial sum [B, T, D].
            layer_idx: Transformer layer index (0..num_layers-1).
            is_mlp: If True, use the pre-MLP pseudo-query (odd index).

        Returns:
            [B, T, D] attention-weighted hidden state.
        """
        query_idx = 2 * layer_idx + (1 if is_mlp else 0)
        wl = self.pseudo_queries[query_idx]

        V = torch.stack(blocks + [partial_block], dim=0)  # [N_src, B, T, D]
        K = self.key_norm(V)
        logits = torch.einsum("d, n b t d -> n b t", wl, K)
        attn = torch.softmax(logits, dim=0)
        return torch.einsum("n b t, n b t d -> b t d", attn, V)

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> BaseModelOutputWithPast:
        """Forward pass with Block AttnRes residuals.

        Args:
            input_ids: [B, T] token IDs.
            attention_mask: [B, T] attention mask (1 for real tokens, 0 for PAD).

        Returns:
            BaseModelOutputWithPast with last_hidden_state being the
            accumulated block output after all layers.
        """
        # --- Token embeddings ---
        hidden_states = self._embed_in(input_ids)  # [B, T, D]

        # --- AttnRes state ---
        blocks: list[torch.Tensor] = []
        partial_block = hidden_states  # b_0^i starts as token embedding

        for layer_idx, layer in enumerate(self._layers):
            # ===== Pre-ATTN: Block AttnRes =====
            h = self._block_attn_res(blocks, partial_block, layer_idx, is_mlp=False)

            # ===== Block boundary check =====
            # Every `block_t_layers` transformer layers, save the current partial
            # sum as a completed block and reset for the next block.
            # The token embedding is automatically saved as b_0 at layer 0.
            if layer_idx % self.block_t_layers == 0:
                blocks.append(partial_block)  # completed block representation
                partial_block = None

            # ===== Self-attention (PreNorm) =====
            attn_out = layer.attention(
                layer.input_layernorm(h),
                attention_mask=attention_mask,
            )
            if isinstance(attn_out, (tuple, list)):
                attn_out = attn_out[0]
            partial_block = (
                partial_block + attn_out if partial_block is not None else attn_out
            )

            # ===== Pre-MLP: Block AttnRes =====
            h = self._block_attn_res(blocks, partial_block, layer_idx, is_mlp=True)

            # ===== MLP (PreNorm) =====
            mlp_out = layer.mlp(layer.post_attention_layernorm(h))
            partial_block = partial_block + mlp_out

        # --- Final layer norm (if the original backbone applies one) ---
        if self._final_norm is not None:
            partial_block = self._final_norm(partial_block)

        return BaseModelOutputWithPast(last_hidden_state=partial_block)

    def get_input_embeddings(self):
        """Delegate to backbone for compatibility."""
        return self.backbone.get_input_embeddings()

    @property
    def device(self):
        """Infer device from the first parameter."""
        return next(self.backbone.parameters()).device
