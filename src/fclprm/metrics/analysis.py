"""CD-SPI vs Performance correlation analysis.

Extracts CD-SPI metrics and task performance (MSE, accuracy) from federated
simulation history and computes statistical correlations.

Addresses expert panel P1 requirement (#6): CD-SPI vs Performance correlation
— scatter-plot-ready data and correlation coefficients that quantify whether
higher cross-client divergence (CD-SPI) actually tracks with better or worse
aggregation outcomes.

References:
    Pearson r:  measures linear relationship between CD-SPI and performance.
    Spearman ρ: measures monotonic relationship (robust to outliers).
"""

from __future__ import annotations

from typing import Any

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

try:
    from scipy.stats import pearsonr, spearmanr
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def extract_cd_spi_series(
    history: list[dict],
    key: str = "cd_spi",
    subkey: str = "cd_spi_mean",
) -> list[float]:
    """Extract per-round CD-SPI mean values from simulator history.

    Args:
        history: Simulator history list (each entry is a round dict).
        key: Top-level key in each round dict (default "cd_spi").
        subkey: Nested key for the scalar value (default "cd_spi_mean").

    Returns:
        List of CD-SPI values, one per round. Rounds without CD-SPI data
        are filtered out.
    """
    series: list[float] = []
    for entry in history:
        if key in entry and isinstance(entry[key], dict):
            val = entry[key].get(subkey)
            if val is not None:
                series.append(float(val))
    return series


def extract_performance_series(
    history: list[dict],
    metric: str = "avg_loss",
    per_domain_suffix: str | None = None,
) -> list[float]:
    """Extract per-round performance metrics from simulator history.

    Args:
        history: Simulator history list.
        metric: Which top-level metric to extract:
            - "avg_loss": average client loss (default).
            - "per_domain_mse": mean per-domain MSE (aggregated).
        per_domain_suffix: When metric="per_domain_mse", optionally select
            a specific client domain metric by suffix (e.g., "client_0_mse").
            If None, the mean across all available per-domain entries is used.

    Returns:
        List of scalar performance values, one per round.
    """
    series: list[float] = []
    for entry in history:
        if metric == "per_domain_mse" and "per_domain_mse" in entry:
            pdm = entry["per_domain_mse"]
            if per_domain_suffix is not None:
                val = pdm.get(per_domain_suffix)
            else:
                vals = [v for v in pdm.values() if isinstance(v, (int, float))]
                val = float(np.mean(vals)) if vals and HAS_NUMPY else (
                    sum(vals) / len(vals) if vals else None
                )
            if val is not None:
                series.append(float(val))
        elif metric in entry:
            val = entry[metric]
            if val is not None:
                series.append(float(val))
    return series


def prepare_aligned_series(
    history: list[dict],
    cd_spi_key: str = "cd_spi",
    cd_spi_subkey: str = "cd_spi_mean",
    perf_metric: str = "avg_loss",
    per_domain_suffix: str | None = None,
) -> dict[str, Any]:
    """Align CD-SPI and performance series by round index.

    Some rounds may have CD-SPI but no eval (or vice versa).  This function
    ensures the two series correspond to the same round indices.

    Returns:
        dict with:
            "cd_spi": list[float] — CD-SPI values
            "performance": list[float] — aligned performance values
            "rounds": list[int] — round indices of the aligned data
            "n": int — number of aligned data points
            "cd_spi_mean": float, "cd_spi_std": float
            "perf_mean": float, "perf_std": float
    """
    rounds: list[int] = []
    cd_spi_vals: list[float] = []
    perf_vals: list[float] = []

    for entry in history:
        # Extract CD-SPI
        cd_spi: float | None = None
        if cd_spi_key in entry and isinstance(entry[cd_spi_key], dict):
            cd_spi = entry[cd_spi_key].get(cd_spi_subkey)
        if cd_spi is None:
            continue

        # Extract performance
        perf: float | None = None
        if perf_metric == "per_domain_mse" and "per_domain_mse" in entry:
            pdm = entry["per_domain_mse"]
            if per_domain_suffix is not None:
                perf = pdm.get(per_domain_suffix)
            else:
                vals = [v for v in pdm.values() if isinstance(v, (int, float))]
                perf = float(np.mean(vals)) if vals and HAS_NUMPY else (
                    sum(vals) / len(vals) if vals else None
                )
        elif perf_metric in entry:
            perf = entry[perf_metric]
        if perf is None:
            continue

        rounds.append(entry.get("round", len(rounds)))
        cd_spi_vals.append(float(cd_spi))
        perf_vals.append(float(perf))

    result: dict[str, Any] = {
        "cd_spi": cd_spi_vals,
        "performance": perf_vals,
        "rounds": rounds,
        "n": len(cd_spi_vals),
    }

    if cd_spi_vals and HAS_NUMPY:
        result["cd_spi_mean"] = float(np.mean(cd_spi_vals))
        result["cd_spi_std"] = float(np.std(cd_spi_vals))
        result["perf_mean"] = float(np.mean(perf_vals))
        result["perf_std"] = float(np.std(perf_vals))
    elif cd_spi_vals:
        n = len(cd_spi_vals)
        result["cd_spi_mean"] = sum(cd_spi_vals) / n
        result["cd_spi_std"] = (
            (sum(v * v for v in cd_spi_vals) / n - result["cd_spi_mean"] ** 2) ** 0.5
        )
        result["perf_mean"] = sum(perf_vals) / n
        result["perf_std"] = (
            (sum(v * v for v in perf_vals) / n - result["perf_mean"] ** 2) ** 0.5
        )

    return result


# ---------------------------------------------------------------------------
# Correlation computation
# ---------------------------------------------------------------------------

def compute_correlation(
    x: list[float],
    y: list[float],
    method: str = "pearson",
) -> dict[str, Any]:
    """Compute correlation between two series.

    Args:
        x: First variable (e.g., CD-SPI).
        y: Second variable (e.g., performance metric).
        method: "pearson" (linear) or "spearman" (monotonic).

    Returns:
        dict with:
            "method": str — the method used.
            "r": float — correlation coefficient (NaN on failure).
            "p_value": float — two-sided p-value (NaN if scipy unavailable).
            "n": int — number of data points.
            "interpretation": str — qualitative label.

    Raises:
        ValueError: On unknown method or empty input.
    """
    if not x or not y:
        raise ValueError("Cannot compute correlation on empty series")
    if len(x) != len(y):
        raise ValueError(f"Series length mismatch: {len(x)} vs {len(y)}")
    if len(x) < 3:
        return {
            "method": method,
            "r": float("nan"),
            "p_value": float("nan"),
            "n": len(x),
            "interpretation": "insufficient_samples",
        }

    n = len(x)

    if method == "pearson":
        if HAS_SCIPY:
            r, p = pearsonr(x, y)
        elif HAS_NUMPY:
            r = float(np.corrcoef(x, y)[0, 1])
            p = float("nan")
        else:
            # Pure-Python fallback
            x_arr = list(x)
            y_arr = list(y)
            mx = sum(x_arr) / n
            my = sum(y_arr) / n
            num = sum((xi - mx) * (yi - my) for xi, yi in zip(x_arr, y_arr))
            den = (
                sum((xi - mx) ** 2 for xi in x_arr)
                * sum((yi - my) ** 2 for yi in y_arr)
            ) ** 0.5
            r = num / den if den > 0 else 0.0
            p = float("nan")
    elif method == "spearman":
        if HAS_SCIPY:
            r, p = spearmanr(x, y)
        else:
            # Fallback: compute Pearson on ranks
            x_rank = _rank(x)
            y_rank = _rank(y)
            if HAS_NUMPY:
                r = float(np.corrcoef(x_rank, y_rank)[0, 1])
            else:
                n = len(x_rank)
                mx = sum(x_rank) / n
                my = sum(y_rank) / n
                num = sum((xi - mx) * (yi - my) for xi, yi in zip(x_rank, y_rank))
                den = (
                    sum((xi - mx) ** 2 for xi in x_rank)
                    * sum((yi - my) ** 2 for yi in y_rank)
                ) ** 0.5
                r = num / den if den > 0 else 0.0
            p = float("nan")
    else:
        raise ValueError(f"Unknown correlation method: '{method}'. "
                         f"Use 'pearson' or 'spearman'.")

    # Qualitative interpretation
    abs_r = abs(r)
    if abs_r >= 0.8:
        interpretation = "very_strong"
    elif abs_r >= 0.6:
        interpretation = "strong"
    elif abs_r >= 0.4:
        interpretation = "moderate"
    elif abs_r >= 0.2:
        interpretation = "weak"
    else:
        interpretation = "very_weak"

    return {
        "method": method,
        "r": float(r),
        "p_value": float(p),
        "n": n,
        "interpretation": interpretation,
    }


def _rank(values: list[float]) -> list[float]:
    """Compute rank (1-based, ties get average rank)."""
    indexed = sorted((v, i) for i, v in enumerate(values))
    n = len(values)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        # Find tie group
        while j + 1 < n and indexed[j + 1][0] == indexed[i][0]:
            j += 1
        avg_rank = (i + 1 + j + 1) / 2.0
        for k in range(i, j + 1):
            ranks[indexed[k][1]] = avg_rank
        i = j + 1
    return ranks


# ---------------------------------------------------------------------------
# Multi-experiment comparison
# ---------------------------------------------------------------------------

def cross_experiment_correlation(
    experiments: dict[str, list[dict]],
    cd_spi_subkey: str = "cd_spi_mean",
    perf_metric: str = "avg_loss",
    method: str = "pearson",
) -> list[dict[str, Any]]:
    """Compute CD-SPI vs performance correlation for multiple experiments.

    Each experiment's final-round (or mean) CD-SPI is paired with the
    corresponding performance metric, and the cross-experiment correlation
    is computed.  This answers: *across different training configurations,
    does higher CD-SPI correlate with better/worse performance?*

    Args:
        experiments: Dict mapping experiment_name -> simulator history.
        cd_spi_subkey: Which CD-SPI variant to use.
        perf_metric: Which performance metric to use.
        method: Correlation method.

    Returns:
        List of dicts, one per experiment, plus a final entry with the
        cross-experiment correlation result.
    """
    results: list[dict[str, Any]] = []
    cd_spi_values: list[float] = []
    perf_values: list[float] = []
    names: list[str] = []

    for name, history in experiments.items():
        aligned = prepare_aligned_series(
            history,
            cd_spi_subkey=cd_spi_subkey,
            perf_metric=perf_metric,
        )
        if aligned["n"] == 0:
            continue

        # Use final-round values for cross-experiment comparison
        final_cd_spi = aligned["cd_spi"][-1]
        final_perf = aligned["performance"][-1]

        results.append({
            "experiment": name,
            "n_rounds": aligned["n"],
            "cd_spi_mean": aligned.get("cd_spi_mean"),
            "perf_mean": aligned.get("perf_mean"),
            "final_cd_spi": final_cd_spi,
            "final_perf": final_perf,
        })
        cd_spi_values.append(final_cd_spi)
        perf_values.append(final_perf)
        names.append(name)

    if len(names) >= 3:
        corr = compute_correlation(cd_spi_values, perf_values, method=method)
        results.append({
            "experiment": f"cross_experiment_{method}",
            "method": method,
            "r": corr["r"],
            "p_value": corr["p_value"],
            "n": corr["n"],
            "interpretation": corr["interpretation"],
        })

    return results


# ---------------------------------------------------------------------------
# Convenience: per-round delta analysis
# ---------------------------------------------------------------------------

def cd_spi_performance_delta(
    history: list[dict],
    window: int = 1,
) -> dict[str, Any]:
    """Analyse whether within-experiment CD-SPI changes track performance.

    Computes the correlation between *changes* in CD-SPI and *changes* in
    the performance metric from round to round (or over a sliding window).

    This addresses: "does increasing client divergence over training rounds
    correspond to improving or degrading aggregation quality?"

    Args:
        history: Simulator history.
        window: Delta window size (1 = round-to-round change).

    Returns:
        dict with delta series and correlation result.
    """
    aligned = prepare_aligned_series(history)
    cd_spi = aligned["cd_spi"]
    perf = aligned["performance"]

    if len(cd_spi) < window + 2:
        return {"error": f"insufficient rounds for window={window}"}

    d_cd_spi = [cd_spi[i] - cd_spi[i - window]
                for i in range(window, len(cd_spi))]
    d_perf = [perf[i] - perf[i - window]
              for i in range(window, len(perf))]

    corr = compute_correlation(d_cd_spi, d_perf)

    return {
        "window": window,
        "n_deltas": len(d_cd_spi),
        "d_cd_spi_mean": float(np.mean(d_cd_spi)) if HAS_NUMPY else (
            sum(d_cd_spi) / len(d_cd_spi)
        ),
        "d_perf_mean": float(np.mean(d_perf)) if HAS_NUMPY else (
            sum(d_perf) / len(d_perf)
        ),
        "correlation": corr,
        "d_cd_spi": d_cd_spi,
        "d_perf": d_perf,
    }
