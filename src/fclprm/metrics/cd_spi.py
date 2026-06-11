"""Cross-Domain Step Polysemy Index (CD-SPI).

Measures whether the same reasoning step shares embedding directions
across different client domains when trained in isolation.

Used in v2 to measure cross-client divergence under full-parameter FT.
"""

import torch
import torch.nn.functional as F


def compute_cd_spi(
    step_text: str,
    client_embeddings: dict[str, torch.Tensor],
) -> float:
    """Compute CD-SPI for a single step across clients.

    Definition:
        CD-SPI(s) = 1 - mean_{i,j} cos(h_i(s), h_j(s))

    Where h_i(s) is the hidden representation of step s from client i's PRM.

    Interpretation:
        - CD-SPI ~ 0: step embeddings are aligned (universal substrate)
        - CD-SPI -> 1: high polysemy (same form, different semantics)

    Args:
        step_text: The reasoning step text (for reference, not used in computation).
        client_embeddings: Dict mapping client_id -> embedding tensor of shape (D,).

    Returns:
        CD-SPI scalar in [0, 1].

    Raises:
        ValueError: If fewer than 2 client embeddings are provided.
    """
    if len(client_embeddings) < 2:
        raise ValueError("CD-SPI requires at least 2 client embeddings")

    embeddings = list(client_embeddings.values())
    num_clients = len(embeddings)

    # Normalize embeddings; handle zero vectors gracefully
    normalized = []
    for e in embeddings:
        norm = F.normalize(e.unsqueeze(0), dim=-1).squeeze(0)
        if torch.isnan(norm).any():
            norm = torch.zeros_like(norm)
        normalized.append(norm)

    # Compute pairwise cosine similarities
    total_sim = 0.0
    count = 0
    for i in range(num_clients):
        for j in range(i + 1, num_clients):
            sim = torch.dot(normalized[i], normalized[j]).item()
            if not torch.isnan(torch.tensor(sim)):
                total_sim += sim
                count += 1

    mean_sim = total_sim / count if count > 0 else 1.0
    cd_spi = 1.0 - mean_sim

    # Clamp to [0, 1] for numerical stability
    return float(torch.tensor(cd_spi).clamp(0.0, 1.0).item())


def compute_cd_spi_batch(
    step_list: list[str],
    all_client_embeddings: dict[str, list[torch.Tensor]],
) -> dict[str, float]:
    """Compute CD-SPI for a batch of steps.

    Args:
        step_list: List of step texts.
        all_client_embeddings: Dict mapping client_id -> list of embeddings,
            where list index corresponds to step_list index.

    Returns:
        Dict mapping step_text -> CD-SPI value.
    """
    results = {}
    for idx, step_text in enumerate(step_list):
        client_embs = {
            client_id: embeddings[idx]
            for client_id, embeddings in all_client_embeddings.items()
        }
        results[step_text] = compute_cd_spi(step_text, client_embs)
    return results


def compute_cd_spi_by_category(
    step_categories: dict[str, list[str]],
    all_client_embeddings: dict[str, dict[str, torch.Tensor]],
) -> dict[str, float]:
    """Compute mean CD-SPI per step category.

    Args:
        step_categories: Dict mapping category_name -> list of step texts.
        all_client_embeddings: Dict mapping client_id -> dict(step_text -> embedding).

    Returns:
        Dict mapping category_name -> mean CD-SPI.
    """
    category_cspi = {}
    for category, steps in step_categories.items():
        cspi_values = []
        for step in steps:
            client_embs = {
                client_id: embeddings.get(step)
                for client_id, embeddings in all_client_embeddings.items()
                if step in embeddings
            }
            if len(client_embs) >= 2:
                cspi_values.append(compute_cd_spi(step, client_embs))

        category_cspi[category] = (
            sum(cspi_values) / len(cspi_values) if cspi_values else 0.0
        )

    return category_cspi
