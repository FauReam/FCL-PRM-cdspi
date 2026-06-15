"""CD-SPI statistical completeness — expert panel P0 requirements.

Adds:
  1. Permutation test for CD-SPI significance (p-value)
  2. Noise-injection ablation (distinguish signal from noise)
  3. Round-wise CD-SPI tracking curve
  4. Function-space output divergence (cosine similarity of reward outputs)
  5. JS divergence of per-client reward distributions
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


# ── 1. Permutation test ──────────────────────────────────────────────────

def permutation_test_cd_spi(
    client_embeddings: dict[str, torch.Tensor],
    n_permutations: int = 1000,
    seed: int = 42,
) -> dict:
    """Permutation test for CD-SPI statistical significance.

    H0: observed CD-SPI is consistent with random embedding assignment.
    HA: cross-client divergence is structural (not noise).

    Procedure:
      1. Compute observed CD-SPI from actual client groupings.
      2. Shuffle embedding-to-client assignment n_permutations times.
      3. Compute CD-SPI for each shuffle.
      4. p-value = fraction of shuffled CD-SPIs >= observed CD-SPI.

    Args:
        client_embeddings: Dict mapping client_id -> embedding tensor (D,).
        n_permutations: Number of random shuffles (min 100 for α=0.01).
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys: observed_cd_spi, p_value, null_mean, null_std,
        null_percentiles (5th, 95th), significant (bool at α=0.05),
        effect_size (Cohen's d), n_permutations.
    """
    if len(client_embeddings) < 2:
        return {"observed_cd_spi": 0.0, "p_value": 1.0, "significant": False}

    rng = torch.Generator()
    rng.manual_seed(seed)

    cids = sorted(client_embeddings.keys())
    C = len(cids)
    embs = torch.stack([client_embeddings[c] for c in cids])  # (C, D)

    # Observed CD-SPI
    observed = _pairwise_cd_spi(embs)
    if torch.isnan(observed):
        observed = torch.tensor(0.0)

    # Null distribution: shuffle client labels
    null_vals = []
    for _ in range(n_permutations):
        perm = torch.randperm(C)
        shuffled = embs[perm]
        null_cd = _pairwise_cd_spi(shuffled)
        if not torch.isnan(null_cd):
            null_vals.append(null_cd.item())

    null_t = torch.tensor(null_vals)
    p_value = (null_t >= observed.item()).float().mean().item()

    # Cohen's d effect size: (observed - null_mean) / null_std
    null_std_val = null_t.std().item()
    null_mean_val = null_t.mean().item()
    effect_size = (
        (observed.item() - null_mean_val) / null_std_val
        if null_std_val > 1e-12 else 0.0
    )

    return {
        "observed_cd_spi": round(observed.item(), 6),
        "p_value": round(p_value, 6),
        "null_mean": round(null_mean_val, 6),
        "null_std": round(null_std_val, 6),
        "null_p05": round(torch.quantile(null_t, 0.05).item(), 6),
        "null_p95": round(torch.quantile(null_t, 0.95).item(), 6),
        "significant": bool(p_value < 0.05),
        "effect_size": round(effect_size, 4),
        "n_permutations": n_permutations,
    }


def _pairwise_cd_spi(embs: torch.Tensor) -> torch.Tensor:
    """CD-SPI for stacked embeddings. embs: (C, D)."""
    C = embs.size(0)
    normed = F.normalize(embs, dim=-1)
    # Pairwise cosine similarity matrix
    sim_matrix = normed @ normed.T  # (C, C)
    # Upper triangle (i < j) mean
    mask = torch.triu(torch.ones(C, C), diagonal=1).bool()
    mean_sim = sim_matrix[mask].nan_to_num(0.0).mean()
    return (1.0 - mean_sim).clamp(0.0, 1.0)


# ── 2. Noise-injection ablation ──────────────────────────────────────────

def noise_injection_cd_spi(
    base_embeddings: dict[str, torch.Tensor],
    noise_scales: list[float] | None = None,
    n_trials: int = 10,
    seed: int = 42,
) -> dict[str, list[float]]:
    """Ablation: inject Gaussian noise at increasing scales to test CD-SPI robustness.

    Rationale: if CD-SPI differences are structural (semantic divergence), they
    should persist under moderate noise. If they are just noise, injection of
    small σ will erase the observed differences.

    Args:
        base_embeddings: Dict mapping client_id -> embedding tensor (D,).
        noise_scales: List of σ values for Gaussian noise. Default: [0.0, 0.01, 0.05, 0.1, 0.2, 0.5].
        n_trials: Trials per noise scale.
        seed: Random seed.

    Returns:
        Dict mapping noise_scale -> list of CD-SPI values across trials.
    """
    if noise_scales is None:
        noise_scales = [0.0, 0.01, 0.05, 0.1, 0.2, 0.5]

    rng = torch.Generator()
    rng.manual_seed(seed)

    cids = sorted(base_embeddings.keys())
    base = {c: base_embeddings[c].clone() for c in cids}
    emb_norm = max(e.norm().item() for e in base.values()) or 1.0

    results: dict[str, list[float]] = {}
    for sigma in noise_scales:
        trials = []
        for _ in range(n_trials):
            noisy = {}
            for c in cids:
                noise = torch.randn(base[c].shape) * sigma * emb_norm
                noisy[c] = base[c] + noise
            cd = _pairwise_cd_spi(torch.stack([noisy[c] for c in cids]))
            trials.append(round(cd.item(), 6))
        results[f"sigma_{sigma}"] = trials

    return results


def noise_threshold_cd_spi(
    base_embeddings: dict[str, torch.Tensor],
    target_cd_spi_drop: float = 0.5,
    seed: int = 42,
) -> dict:
    """Find the noise scale at which CD-SPI drops to `target_cd_spi_drop`
    of its original value — a single-number robustness score.

    Returns:
        Dict with 'cd_spi_clean', 'threshold_sigma', 'cd_spi_at_threshold'.
    """
    noise_scales = [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]
    results = noise_injection_cd_spi(
        base_embeddings, noise_scales=noise_scales, n_trials=5, seed=seed
    )

    clean_cd = torch.tensor(results["sigma_0.0"]).mean().item()
    target = clean_cd * (1.0 - target_cd_spi_drop)

    threshold = float("inf")
    cd_at_threshold = clean_cd
    for sigma in noise_scales:
        key = f"sigma_{sigma}"
        mean_cd = torch.tensor(results[key]).mean().item()
        if mean_cd <= target:
            threshold = sigma
            cd_at_threshold = mean_cd
            break

    return {
        "cd_spi_clean": round(clean_cd, 6),
        "threshold_sigma": threshold,
        "cd_spi_at_threshold": round(cd_at_threshold, 6),
    }


# ── 3. Function-space output divergence ──────────────────────────────────

def compute_output_cosine_divergence(
    client_predictions: dict[str, torch.Tensor],
) -> float:
    """Function-space divergence: 1 - mean pairwise cosine similarity of
    client reward predictions on a shared hold-out set.

    Unlike CD-SPI (parameter-space embeddings), this measures whether
    clients actually disagree in their *outputs*, not just their internal
    representations. This addresses the expert panel critique that CD-SPI
    may measure noise rather than functional divergence.

    Args:
        client_predictions: Dict mapping client_id -> reward predictions
            tensor of shape (N,) for N shared evaluation steps.

    Returns:
        Scalar in [0, 1] — 0 means identical outputs, 1 means orthogonal.
    """
    cids = sorted(client_predictions.keys())
    if len(cids) < 2:
        return 0.0

    preds = torch.stack([client_predictions[c] for c in cids])  # (C, N)
    normed = F.normalize(preds, dim=-1)
    sim = normed @ normed.T  # (C, C)
    mask = torch.triu(torch.ones(len(cids), len(cids)), diagonal=1).bool()
    mean_sim = sim[mask].nan_to_num(0.0).mean()
    return round((1.0 - mean_sim).clamp(0.0, 1.0).item(), 6)


def compute_js_output_divergence(
    client_predictions: dict[str, torch.Tensor],
    n_bins: int = 50,
) -> float:
    """Jensen-Shannon divergence between client reward prediction distributions.

    Uses histogram-based density estimation with n_bins over the pooled
    prediction range. Higher JS divergence means clients produce systematically
    different reward distributions — functional disagreement, not just noise.

    Args:
        client_predictions: Dict mapping client_id -> reward predictions (N,).
        n_bins: Number of histogram bins.

    Returns:
        Mean pairwise JS divergence in [0, ln(2)].
    """
    cids = sorted(client_predictions.keys())
    if len(cids) < 2:
        return 0.0

    # Shared bin edges across all clients
    all_preds = torch.cat([client_predictions[c] for c in cids])
    vmin, vmax = all_preds.min().item(), all_preds.max().item()
    if vmax <= vmin:
        vmax = vmin + 1e-6
    bin_edges = torch.linspace(vmin, vmax, n_bins + 1)

    # Per-client histograms (normalized to probability)
    histograms = {}
    for c in cids:
        h = torch.histc(client_predictions[c].float(), bins=n_bins, min=vmin, max=vmax)
        h = h / h.sum().clamp(min=1e-9)
        histograms[c] = h

    # Pairwise JS divergence
    total_js = 0.0
    count = 0
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            p = histograms[cids[i]]
            q = histograms[cids[j]]
            m = 0.5 * (p + q)
            kl_pm = (p * (p / m.clamp(min=1e-9)).log()).nan_to_num(0.0).sum()
            kl_qm = (q * (q / m.clamp(min=1e-9)).log()).nan_to_num(0.0).sum()
            total_js += 0.5 * (kl_pm + kl_qm).item()
            count += 1

    return round(total_js / max(count, 1), 6)


# ── 4. Round-wise CD-SPI tracking ────────────────────────────────────────

class CDSPITracker:
    """Track CD-SPI evolution across federated rounds.

    Stores per-round CD-SPI values and provides summary statistics
    (convergence rate, final value, stability metric).
    """

    def __init__(self) -> None:
        self._history: list[dict] = []

    def record(self, round_num: int, cd_spi: float) -> None:
        self._history.append({"round": round_num, "cd_spi": cd_spi})

    def get_curve(self) -> list[float]:
        return [h["cd_spi"] for h in self._history]

    def summary(self) -> dict:
        if not self._history:
            return {}
        values = torch.tensor(self.get_curve())
        return {
            "cd_spi_init": round(values[0].item(), 6),
            "cd_spi_final": round(values[-1].item(), 6),
            "cd_spi_mean": round(values.mean().item(), 6),
            "cd_spi_std": round(values.std().item(), 6),
            "cd_spi_trend": "increasing" if values[-1] > values[0] else "decreasing",
            "cd_spi_range": round((values.max() - values.min()).item(), 6),
            "n_rounds": len(values),
        }
