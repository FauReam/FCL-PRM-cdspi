"""CD-SPI diagnostic framework metrics.

CD-SPI (Client Divergence Signal-Noise Partition Index):
    Two-stage diagnostic protocol:
    Stage 1 — Permutation test (cd_spi_stats): statistical significance
    Stage 2 — PCA EVR (cd_spi): structural vs noise divergence

Independent cross-validation:
    CKA (cka.py) — Centered Kernel Alignment, robust to linear transforms
    Function-space divergence (cd_spi_stats): output cosine + JS divergence

Statistical completeness:
    Noise injection, CD-SPI tracker, function-space divergence

Analysis:
    CD-SPI vs performance correlation, cross-experiment comparison
"""

from fclprm.metrics.bon import best_of_n_accuracy
from fclprm.metrics.cd_spi import (
    compute_cd_spi,
    compute_cd_spi_batch,
    compute_cd_spi_by_category,
    compute_pca_evr,
)
from fclprm.metrics.cd_spi_stats import (
    CDSPITracker,
    compute_js_output_divergence,
    compute_output_cosine_divergence,
    noise_injection_cd_spi,
    noise_threshold_cd_spi,
    permutation_test_cd_spi,
)
from fclprm.metrics.prm_bench import ProcessBenchEvaluator
from fclprm.metrics.privacy import evaluate_reconstruction_attack
from fclprm.metrics.analysis import (
    compute_correlation,
    extract_cd_spi_series,
    extract_performance_series,
    prepare_aligned_series,
    cross_experiment_correlation,
    cd_spi_performance_delta,
)
from fclprm.metrics.cka import (
    compute_cka,
    compute_client_cka_matrix,
    compute_cka_cd_spi_contrast,
)

__all__ = [
    # CD-SPI core + PCA EVR (Phase 2 diagnostic)
    "compute_cd_spi",
    "compute_cd_spi_batch",
    "compute_cd_spi_by_category",
    "compute_pca_evr",
    # Statistical completeness (Phase 1 diagnostic)
    "permutation_test_cd_spi",
    "noise_injection_cd_spi",
    "noise_threshold_cd_spi",
    "CDSPITracker",
    # Function-space divergence
    "compute_output_cosine_divergence",
    "compute_js_output_divergence",
    # Independent cross-validation (CKA)
    "compute_cka",
    "compute_client_cka_matrix",
    "compute_cka_cd_spi_contrast",
    # Benchmarks
    "ProcessBenchEvaluator",
    "best_of_n_accuracy",
    "evaluate_reconstruction_attack",
    # Correlation analysis
    "compute_correlation",
    "extract_cd_spi_series",
    "extract_performance_series",
    "prepare_aligned_series",
    "cross_experiment_correlation",
    "cd_spi_performance_delta",
]
