"""Federated learning core: clients, server, aggregators, DP."""

from fclprm.federated.aggregators import (
    anchor_prm_aggregate,
    fedavg_prm,
    robust_aggregate_trimmed_mean,
)
from fclprm.federated.client import FederatedClient
from fclprm.federated.server import FederatedServer
from fclprm.federated.simulator import FederatedSimulator

__all__ = [
    "FederatedClient",
    "FederatedServer",
    "FederatedSimulator",
    "fedavg_prm",
    "anchor_prm_aggregate",
    "robust_aggregate_trimmed_mean",
]
