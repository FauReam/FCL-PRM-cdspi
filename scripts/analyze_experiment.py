#!/usr/bin/env python3
"""Post-experiment analysis entry point.

Loads federated simulator history from logs/checkpoints and produces:
  - CD-SPI vs Performance correlation (Pearson/Spearman)
  - Symmetrical vs asymmetrical CD-SPI comparison
  - PCA EVR + permutation test summary across rounds
  - CKA cross-validation contrast with CD-SPI
  - Experiment comparison (multiple configs)

Usage:
    # Single experiment analysis
    python scripts/analyze_experiment.py --history experiments/M3_full_1.4b/results/logs/history.json

    # Cross-experiment comparison
    python scripts/analyze_experiment.py --compare-dir experiments/
"""

import argparse
import json
import sys
from pathlib import Path


def load_history(path: str) -> list[dict]:
    """Load simulator history from JSON."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"History file not found: {p}")
    with open(p) as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "history" in data:
        return data["history"]
    return data


def analyze_single_experiment(history: list[dict], label: str = "") -> dict:
    """Run full analysis suite on a single experiment history."""
    from fclprm.metrics.analysis import (
        prepare_aligned_series,
        compute_correlation,
        cd_spi_performance_delta,
        extract_cd_spi_series,
    )

    results: dict = {}
    label = label or "experiment"

    # CD-SPI vs loss correlation
    aligned_loss = prepare_aligned_series(
        history, cd_spi_subkey="cd_spi_mean", perf_metric="avg_loss"
    )
    if aligned_loss["n"] >= 3:
        pearson = compute_correlation(
            aligned_loss["cd_spi"], aligned_loss["performance"], method="pearson"
        )
        spearman = compute_correlation(
            aligned_loss["cd_spi"], aligned_loss["performance"], method="spearman"
        )
        results["cd_spi_vs_loss"] = {
            "n_rounds": aligned_loss["n"],
            "cd_spi_range": [round(min(aligned_loss["cd_spi"]), 4),
                             round(max(aligned_loss["cd_spi"]), 4)],
            "loss_range": [round(min(aligned_loss["performance"]), 4),
                           round(max(aligned_loss["performance"]), 4)],
            "pearson": pearson,
            "spearman": spearman,
        }

    # CD-SPI vs per-domain MSE correlation
    aligned_mse = prepare_aligned_series(
        history, cd_spi_subkey="cd_spi_mean", perf_metric="per_domain_mse"
    )
    if aligned_mse["n"] >= 3:
        pearson_mse = compute_correlation(
            aligned_mse["cd_spi"], aligned_mse["performance"], method="pearson"
        )
        results["cd_spi_vs_mse"] = {
            "n_rounds": aligned_mse["n"],
            "cd_spi_range": [round(min(aligned_mse["cd_spi"]), 4),
                             round(max(aligned_mse["cd_spi"]), 4)],
            "mse_range": [round(min(aligned_mse["performance"]), 4),
                          round(max(aligned_mse["performance"]), 4)],
            "pearson": pearson_mse,
        }

    # Per-round delta analysis
    delta = cd_spi_performance_delta(history, window=1)
    if "correlation" in delta:
        results["cd_spi_delta_vs_loss_delta"] = {
            "n_deltas": delta["n_deltas"],
            "pearson": delta["correlation"],
        }

    # Symmetrical CD-SPI analysis (if available in history)
    sym_cd_spi = extract_cd_spi_series(history, key="cd_spi_sym",
                                        subkey="cd_spi_sym_mean")
    if sym_cd_spi:
        asym_cd_spi = extract_cd_spi_series(history, key="cd_spi",
                                             subkey="cd_spi_mean")
        results["sym_vs_asym"] = {
            "sym_cd_spi_mean": round(sum(sym_cd_spi) / len(sym_cd_spi), 4),
            "sym_cd_spi_final": round(sym_cd_spi[-1], 4),
            "asym_cd_spi_mean": round(sum(asym_cd_spi) / len(asym_cd_spi), 4) if asym_cd_spi else None,
            "n_rounds": len(sym_cd_spi),
        }

    # Extract PCA EVR and CKA from final round if available
    for entry in reversed(history):
        cd_spi_sym = entry.get("cd_spi_sym", {})
        if cd_spi_sym:
            results["final_round"] = {
                "round": entry.get("round", "N/A"),
                "pca_evr": cd_spi_sym.get("pca_evr", {}),
                "cka": cd_spi_sym.get("cka", {}),
                "permutation_test": cd_spi_sym.get("permutation_test", {}),
            }
            break

    results["n_history_entries"] = len(history)
    return results


def compare_experiments(experiment_dir: str) -> dict:
    """Compare multiple experiments in a directory tree."""
    from fclprm.metrics.analysis import cross_experiment_correlation

    base = Path(experiment_dir)
    if not base.exists():
        raise FileNotFoundError(f"Experiment directory not found: {base}")

    # Find all history.json files
    history_files = {}
    for p in base.rglob("history.json"):
        exp_name = p.parent.parent.parent.name  # e.g., M3_full_1.4b
        history_files[exp_name] = p

    if not history_files:
        raise FileNotFoundError(f"No history.json found under {base}")

    # Load all histories
    experiments = {}
    for name, path in history_files.items():
        try:
            experiments[name] = load_history(str(path))
            print(f"  Loaded {name} ({len(experiments[name])} rounds)")
        except Exception as e:
            print(f"  Warning: Could not load {name}: {e}")

    if len(experiments) < 2:
        return {"error": "Need at least 2 experiments for comparison",
                "loaded": list(experiments.keys())}

    # Cross-experiment CD-SPI vs performance correlation
    cross = cross_experiment_correlation(
        experiments, cd_spi_subkey="cd_spi_mean", perf_metric="avg_loss"
    )

    # Per-experiment analysis
    individual = {}
    for name, history in experiments.items():
        individual[name] = analyze_single_experiment(history, label=name)

    return {
        "comparison": cross,
        "experiments": individual,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze federated experiment results")
    parser.add_argument(
        "--history", type=str, default=None,
        help="Path to single experiment history.json"
    )
    parser.add_argument(
        "--compare-dir", type=str, default=None,
        help="Directory containing multiple experiment results"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output JSON path (default: stdout)"
    )
    args = parser.parse_args()

    if not args.history and not args.compare_dir:
        print("ERROR: Provide --history or --compare-dir")
        sys.exit(1)

    result: dict = {}

    if args.history:
        print(f"[ANALYSIS] Loading single experiment: {args.history}")
        history = load_history(args.history)
        exp_name = Path(args.history).parent.parent.name
        result = analyze_single_experiment(history, label=exp_name)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.compare_dir:
        print(f"[ANALYSIS] Comparing experiments in: {args.compare_dir}")
        result = compare_experiments(args.compare_dir)
        print(json.dumps(result, indent=2, ensure_ascii=False))

    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"[ANALYSIS] Results saved to: {out_path}")


if __name__ == "__main__":
    main()
