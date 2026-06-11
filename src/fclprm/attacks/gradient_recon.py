"""Gradient-based reasoning trace reconstruction attack (DLG variant).

Reference: Deep Leakage from Gradients (Zhu et al., NeurIPS 2019)
Adapted for step-level PRM gradients on a frozen-backbone + trainable-head
model. The attacker observes per-step PRM head gradients and tries to
recover the (input_ids, scalar label) pair that produced them.

Threat model:
    - Honest-but-curious server observes a client's PRM head gradients.
    - Backbone is frozen, so only head parameters carry signal.
    - Goal: reconstruct the private CoT step text and reward label.

Method:
    1. Initialise dummy `inputs_embeds` (B, L, D) and dummy `labels` (B,).
    2. Forward: backbone(inputs_embeds=..., attention_mask=ones) -> hidden
       at last token -> PRM head -> predictions.
    3. MSE loss against dummy labels.
    4. Differentiable head-gradient via `torch.autograd.grad(..., create_graph=True)`.
    5. Match objective: cosine distance between reconstructed and target
       head gradients (Geiping et al. 2020 "Inverting Gradients").
    6. Adam over (dummy_inputs_embeds, dummy_labels) for `num_iterations`.
    7. Map recovered embeddings back to nearest token IDs in the embedding
       table (cosine NN), and detokenise.
"""

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


def _trainable_head_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return (name, param) pairs whose gradients drive PRM head learning."""
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def _gradient_distance(
    recon_grads: Iterable[torch.Tensor],
    target_grads: Iterable[torch.Tensor],
) -> torch.Tensor:
    """Cosine distance over flattened concatenated gradients.

    This matches Geiping et al. 2020, which is markedly more stable than
    the squared-L2 of the original DLG when label magnitudes vary.
    """
    recon_flat = torch.cat([g.reshape(-1) for g in recon_grads])
    target_flat = torch.cat([g.reshape(-1) for g in target_grads])
    cos = F.cosine_similarity(
        recon_flat.unsqueeze(0), target_flat.unsqueeze(0)
    ).squeeze()
    return 1.0 - cos


class GradientReconstructionAttack:
    """Reconstruct reasoning traces from observed PRM gradients."""

    def __init__(self, model: nn.Module, tokenizer, device: str = "cuda") -> None:
        """Initialize attack with a reference model that mirrors the target.

        Args:
            model: Reference model (same architecture and weights as the
                target client at the moment the gradient was observed).
            tokenizer: HuggingFace tokenizer matching the model.
            device: Compute device.
        """
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def reconstruct(
        self,
        target_gradients: dict[str, torch.Tensor],
        max_steps: int = 1,
        seq_length: int = 32,
        num_iterations: int = 1000,
        lr: float = 0.1,
        verbose: bool = False,
    ) -> dict:
        """Reconstruct a single step (or short batch) from head gradients.

        Args:
            target_gradients: Dict of {param_name: gradient tensor} for the
                trainable head parameters.
            max_steps: Number of steps in the reconstructed batch (B).
            seq_length: Token length L of each reconstructed step.
            num_iterations: Adam steps over the dummy inputs.
            lr: Learning rate for the dummy-input optimiser.
            verbose: If True, print loss every ~10% of iterations.

        Returns:
            Dict with keys:
                - "reconstructed_text": list[str] of length max_steps.
                - "reconstructed_labels": list[float] of length max_steps.
                - "final_distance": final gradient-matching distance.
        """
        backbone = self.model.backbone
        head_params = [p for _, p in _trainable_head_params(self.model)]
        head_param_names = [n for n, _ in _trainable_head_params(self.model)]

        # Sanity: target_gradients must cover all trainable head params
        missing = [n for n in head_param_names if n not in target_gradients]
        if missing:
            raise KeyError(
                f"target_gradients is missing trainable head params: {missing}"
            )

        target_grad_list = [
            target_gradients[n].detach().to(self.device) for n in head_param_names
        ]

        embed_layer = backbone.get_input_embeddings()
        hidden_dim = embed_layer.weight.shape[1]

        # Dummy inputs to optimise. Initialise embeds with the same scale as
        # the embedding table to keep optimisation in-distribution.
        with torch.no_grad():
            init_scale = embed_layer.weight.std().item()
        dummy_embeds = (
            torch.randn(max_steps, seq_length, hidden_dim, device=self.device)
            * init_scale
        )
        dummy_embeds.requires_grad_(True)
        dummy_labels = torch.randn(max_steps, device=self.device, requires_grad=True)
        attention_mask = torch.ones(max_steps, seq_length, device=self.device)

        optimizer = torch.optim.Adam([dummy_embeds, dummy_labels], lr=lr)

        last_distance = float("inf")
        for it in range(num_iterations):
            optimizer.zero_grad()

            # Forward through frozen backbone via inputs_embeds API
            outputs = backbone(
                inputs_embeds=dummy_embeds,
                attention_mask=attention_mask,
            )
            hidden = outputs.last_hidden_state[
                :, -1, :
            ]  # all-ones mask -> last is non-PAD
            predictions = self.model.head(hidden)
            loss = F.mse_loss(predictions, dummy_labels)

            # Differentiable head-gradient
            recon_grads = torch.autograd.grad(
                loss, head_params, create_graph=True, retain_graph=True
            )
            distance = _gradient_distance(recon_grads, target_grad_list)
            distance.backward()
            optimizer.step()

            last_distance = float(distance.detach().item())
            if verbose and (it % max(1, num_iterations // 10) == 0):
                print(f"[DLG] iter={it} grad-distance={last_distance:.6f}")

        # Map recovered embeddings -> nearest tokens (cosine)
        with torch.no_grad():
            emb_table = embed_layer.weight  # (vocab, D)
            emb_table_norm = F.normalize(emb_table, dim=-1)
            recon = F.normalize(dummy_embeds.detach(), dim=-1)
            similarity = recon @ emb_table_norm.t()  # (B, L, vocab)
            token_ids = similarity.argmax(dim=-1)  # (B, L)
            texts = self.tokenizer.batch_decode(token_ids, skip_special_tokens=True)

        return {
            "reconstructed_text": texts,
            "reconstructed_labels": dummy_labels.detach().cpu().tolist(),
            "final_distance": last_distance,
        }
