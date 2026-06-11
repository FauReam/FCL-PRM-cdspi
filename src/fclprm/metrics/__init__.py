"""CD-SPI and function-space divergence metrics.

CD-SPI (Cross-Domain Step Polysemy Index):
    Measures parameter-space embedding divergence across federated clients.
    Lower = better alignment; higher = clients have diverged.

Statistical completeness (cd_spi_stats):
    Permutation test, noise-injection ablation, output divergence,
    JS divergence — addressing expert panel P0 requirements.

Function-space divergence:
    Output cosine similarity and JS divergence of reward distributions
    distinguish semantic divergence from overfitting noise.
"""

from fclprm.metrics.cd_spi import (
    compute_cd_spi,
    compute_cd_spi_batch,
    compute_cd_spi_by_category,
)
from fclprm.metrics.cd_spi_stats import (
    CDSPITracker,
    compute_js_output_divergence,
    compute_output_cosine_divergence,
    noise_injection_cd_spi,
    noise_threshold_cd_spi,
    permutation_test_cd_spi,
)

__all__ = [
    # Core CD-SPI
    "compute_cd_spi",
    "compute_cd_spi_batch",
    "compute_cd_spi_by_category",
    # Statistical completeness
    "permutation_test_cd_spi",
    "noise_injection_cd_spi",
    "noise_threshold_cd_spi",
    "CDSPITracker",
    # Function-space divergence
    "compute_output_cosine_divergence",
    "compute_js_output_divergence",
]
