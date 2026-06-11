"""Cross-Domain Step Polysemy Index (CD-SPI).

Measures whether the same reasoning step shares embedding directions
across different client domains when trained in isolation.

Used in v2 to measure cross-client divergence under full-parameter FT.

Phase 2 of CD-SPI diagnostic protocol:
    compute_pca_evr() — Principal Component Explained Variance Ratio,
    complementing the permutation test (in cd_spi_stats.py) as the second
    stage of the CD-SPI diagnostic framework.

    EVR > 0.6 → divergence is structured (low-dimensional manifold)
    EVR < 0.4 → divergence is near-noise (high-dimensional sphere)
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


def compute_pca_evr(
    client_embeddings: dict[str, torch.Tensor],
) -> dict:
    """Compute Principal Component Explained Variance Ratio (PCA EVR).

    Phase 2 of the CD-SPI diagnostic protocol.  PCA EVR measures whether
    cross-client embedding variation is structured (low-rank, high EVR for
    first PC) or noise-like (isotropic, low EVR).

    Interpretation:
      - EVR > 0.6: divergence is STRUCTURED (low-dimensional manifold)
          → clients vary along consistent semantic axes
      - EVR < 0.4: divergence is NEAR-NOISE (high-dimensional sphere)
          → cross-client differences are essentially random
      - 0.4 <= EVR <= 0.6: ambiguous, requires permutation test

    Args:
        client_embeddings: Dict mapping client_id -> embedding tensor (D,).

    Returns:
        Dict with:
          - evr_first: float — EVR of the first principal component
          - evr_ratio: float — evr_first / evr_second (large = highly structured)
          - evr_all: list[float] — EVR for all components
          - interpretation: str — qualitative label
          - n_clients: int — number of clients
          - embedding_dim: int — dimension of each embedding

    Raises:
        ValueError: If fewer than 3 clients (need at least 3 for PCA).
    """
    if len(client_embeddings) < 3:
        raise ValueError("PCA EVR requires at least 3 client embeddings")

    embeddings = list(client_embeddings.values())
    n_clients = len(embeddings)
    d = embeddings[0].shape[0]

    # Stack embeddings: (n_clients, D)
    X = torch.stack(embeddings)

    # Center the data
    X_centered = X - X.mean(dim=0, keepdim=True)

    # Compute covariance matrix and eigen-decomposition
    # For n_clients < D, use SVD on X directly (more efficient)
    if n_clients < d:
        U, S, Vt = torch.linalg.svd(X_centered, full_matrices=False)
        # Explained variance = S^2 / (n_clients - 1)
        explained_var = S.pow(2) / (n_clients - 1)
        total_var = explained_var.sum()
    else:
        cov = (X_centered.T @ X_centered) / (n_clients - 1)
        eigenvalues = torch.linalg.eigvalsh(cov)
        # eigenvalues are in ascending order; reverse to descending
        eigenvalues = eigenvalues.flip(0)
        total_var = eigenvalues.sum()
        explained_var = eigenvalues

    evr = explained_var / total_var.clamp(min=1e-12)
    evr_first = float(evr[0].item()) if len(evr) > 0 else 0.0
    evr_second = float(evr[1].item()) if len(evr) > 1 else 0.0

    # Determine interpretation
    if evr_first > 0.6:
        interpretation = "structured"
    elif evr_first < 0.4:
        interpretation = "near_noise"
    else:
        interpretation = "ambiguous"

    return {
        "evr_first": evr_first,
        "evr_ratio": evr_first / max(evr_second, 1e-12),
        "evr_all": [float(v) for v in evr],
        "interpretation": interpretation,
        "n_clients": n_clients,
        "embedding_dim": d,
    }
