"""Cross-Domain Step Polysemy Index (CD-SPI) — measures client representation divergence.

CD-SPI quantifies whether the same reasoning step produces similar
embeddings across different federated clients.  Lower values mean
better alignment; higher values mean clients have diverged.

Used in v2 to diagnose the effect of full-parameter fine-tuning on
cross-client representation drift.
"""

from fclprm.metrics.cd_spi import (
    compute_cd_spi,
    compute_cd_spi_batch,
    compute_cd_spi_by_category,
)

__all__ = [
    "compute_cd_spi",
    "compute_cd_spi_batch",
    "compute_cd_spi_by_category",
]
