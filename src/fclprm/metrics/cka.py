"""Centered Kernel Alignment (CKA) — independent cross-validation for CD-SPI.

CKA measures the similarity between two representations by computing the
alignment of their Gram matrices after centering. It is invariant to
orthogonal transformations and isotropic scaling, making it a robust
complement to CD-SPI's cosine-similarity-based measurement.

CKA addresses the expert panel's concern that CD-SPI alone may be
sensitive to measurement asymmetry (different embedding spaces for
head-only vs full-FT). Unlike CD-SPI's element-wise cosine similarity,
CKA compares the *similarity structure within* each representation,
making it robust to:
  - Linear/affine transformations of the embedding space
  - Isotropic scaling differences between configs
  - Measurement point differences (as long as both use the same layer)

Reference:
    Kornblith et al., "Similarity of Neural Network Representations
    Revisited", ICML 2019. https://arxiv.org/abs/1905.00414
"""

from __future__ import annotations

import torch


def compute_cka(
    features_x: torch.Tensor,
    features_y: torch.Tensor,
) -> float:
    """Compute Centered Kernel Alignment between two feature matrices.

    CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))

    where HSIC is the Hilbert-Schmidt Independence Criterion (centered).

    Args:
        features_x: Feature matrix of shape (N, D1).
        features_y: Feature matrix of shape (N, D2). Must have same N.

    Returns:
        CKA similarity in [0, 1]. 1 = identical up to orthogonal transform.

    Raises:
        ValueError: If N < 2 or N differs between matrices.
    """
    if features_x.shape[0] != features_y.shape[0]:
        raise ValueError(
            f"Feature matrices must have same number of rows: "
            f"{features_x.shape[0]} vs {features_y.shape[0]}"
        )
    if features_x.shape[0] < 2:
        raise ValueError("Need at least 2 samples for CKA computation")

    n = features_x.shape[0]

    # Center the Gram matrices
    # K_centered = K - 1_n K / n - K 1_n / n + 1_n K 1_n / n^2
    # where K = X X^T (linear kernel)
    k_x = features_x @ features_x.T  # (N, N)
    k_y = features_y @ features_y.T  # (N, N)

    # Centering matrix: H = I - 1_n 1_n^T / n
    # K_centered = H K H
    # Efficient computation: K_centered = K - row_mean - col_mean + grand_mean
    mean_k_x = k_x.mean(dim=-1, keepdim=True)  # row means (N, 1)
    mean_k_x_col = k_x.mean(dim=-2, keepdim=True)  # col means (1, N)
    grand_k_x = k_x.mean()
    k_x_centered = k_x - mean_k_x - mean_k_x_col + grand_k_x

    mean_k_y = k_y.mean(dim=-1, keepdim=True)
    mean_k_y_col = k_y.mean(dim=-2, keepdim=True)
    grand_k_y = k_y.mean()
    k_y_centered = k_y - mean_k_y - mean_k_y_col + grand_k_y

    hsic_xy = (k_x_centered * k_y_centered).sum() / (n - 1) ** 2
    hsic_xx = (k_x_centered * k_x_centered).sum() / (n - 1) ** 2
    hsic_yy = (k_y_centered * k_y_centered).sum() / (n - 1) ** 2

    denominator = (hsic_xx * hsic_yy).clamp(min=1e-12).sqrt()
    cka = (hsic_xy / denominator).clamp(0.0, 1.0)

    return float(cka.item())


def compute_client_cka_matrix(
    client_features: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Compute pairwise CKA between all pairs of client feature matrices.

    Each client contributes a feature matrix of shape (N, D) where N is the
    number of anchor steps and D is the feature dimension.

    Args:
        client_features: Dict mapping client_id -> feature tensor (N, D).

    Returns:
        Dict with pairwise CKA values:
            "cka_mean": float — mean of all pairwise CKA values
            "cka_min": float — minimum pairwise CKA
            "cka_std": float — standard deviation
            "cka_client_{i}_vs_{j}": float — per-pair values
    """
    client_ids = list(client_features.keys())
    n_clients = len(client_ids)

    if n_clients < 2:
        return {"cka_mean": 1.0, "cka_min": 1.0, "cka_std": 0.0}

    cka_values = []
    results: dict[str, float] = {}

    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            cid_i = client_ids[i]
            cid_j = client_ids[j]
            cka_val = compute_cka(
                client_features[cid_i], client_features[cid_j]
            )
            cka_values.append(cka_val)
            results[f"cka_{cid_i}_vs_{cid_j}"] = cka_val

    cka_t = torch.tensor(cka_values)
    results["cka_mean"] = float(cka_t.mean().item())
    results["cka_min"] = float(cka_t.min().item())
    results["cka_std"] = float(cka_t.std().item())

    return results


def compute_cka_cd_spi_contrast(
    cka_mean: float,
    cd_spi_mean: float,
) -> dict[str, float]:
    """Compare CKA and CD-SPI for consistency check.

    If CKA and CD-SPI give contradictory signals (e.g., CKA high=similar
    while CD-SPI high=divergent), this flags a potential measurement bias.

    Args:
        cka_mean: Mean pairwise CKA (0=diverge, 1=aligned).
        cd_spi_mean: Mean CD-SPI (0=aligned, 1=diverge).

    Returns:
        Dict with:
        - "cka_cd_spi_consistency": bool — whether both agree
        - "cka_cd_spi_contrast": float — normalized difference
        - "interpretation": str — qualitative label
    """
    # CKA and CD-SPI are inversely related: CKA high = aligned = CD-SPI low
    # Normalize both to [0,1] with same polarity: 0=aligned, 1=diverge
    # cka_div = 1 - cka_mean
    cka_div = 1.0 - cka_mean
    contrast = abs(cd_spi_mean - cka_div)

    # Flags: if they differ by more than 0.3, something is inconsistent
    consistent = contrast < 0.3

    interpretation = (
        "consistent" if consistent
        else "inconsistent_cka_higher" if cd_spi_mean < cka_div
        else "inconsistent_spi_higher"
    )

    return {
        "cka_cd_spi_consistency": consistent,
        "cka_cd_spi_contrast": round(contrast, 4),
        "interpretation": interpretation,
    }
