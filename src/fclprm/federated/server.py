"""Federated server: coordinates aggregation and maintains global model."""

import copy
from typing import Callable

import torch
import torch.nn as nn

from fclprm.federated.aggregators import anchor_prm_aggregate, fedavg_prm


class FederatedServer:
    """Server-side orchestration.

    Responsibilities:
        - Maintain global PRM model
        - Receive client updates (deltas + step embeddings)
        - Apply aggregation rule (FedAvg / Anchor-PRM / robust)
        - Broadcast updated global model
    """

    def __init__(
        self,
        global_model: nn.Module,
        aggregation_rule: str = "fedavg",
        anchor_steps: list[str] | None = None,
    ) -> None:
        """Initialize server.

        Args:
            global_model: Initial global model.
            aggregation_rule: Aggregation strategy name.
            anchor_steps: Shared anchor step texts (Anchor-PRM only;
                logged for reproducibility).
        """
        self.global_model = global_model
        self.aggregation_rule = aggregation_rule
        self.anchor_steps = anchor_steps or []
        self.round_num = 0
        self.history: list[dict] = []

    def aggregate(self, client_updates: list[dict]) -> nn.Module:
        """Aggregate client updates into new global model.

        Args:
            client_updates: List of update dicts from clients.
                Each dict must contain 'state_dict'. For 'anchor_prm'
                aggregation, each update must additionally carry an
                'anchor_embeddings' tensor of shape (N, head_dim).

        Returns:
            Updated global model.
        """
        state_dicts = [u["state_dict"] for u in client_updates]

        if self.aggregation_rule == "fedavg":
            self.global_model = fedavg_prm(self.global_model, state_dicts)
        elif self.aggregation_rule == "trimmed_mean":
            from fclprm.federated.aggregators import robust_aggregate_trimmed_mean

            self.global_model = robust_aggregate_trimmed_mean(
                self.global_model, state_dicts
            )
        elif self.aggregation_rule == "anchor_prm":
            client_embeddings = {
                u.get("client_id", i): u["anchor_embeddings"]
                for i, u in enumerate(client_updates)
                if "anchor_embeddings" in u and u["anchor_embeddings"] is not None
            }
            self.global_model = anchor_prm_aggregate(
                global_model=self.global_model,
                client_updates=client_updates,
                client_embeddings=client_embeddings,
                anchor_steps=self.anchor_steps,
            )
        else:
            raise ValueError(f"Unknown aggregation rule: {self.aggregation_rule}")

        self.round_num += 1
        self.history.append(
            {
                "round": self.round_num,
                "num_clients": len(client_updates),
                "aggregation": self.aggregation_rule,
            }
        )

        return self.global_model

    def broadcast(self) -> dict:
        """Broadcast current global model state to clients.

        Returns:
            Global model state dict.
        """
        return copy.deepcopy(self.global_model.state_dict())

    def get_global_model(self) -> nn.Module:
        """Return the current global model.

        Returns:
            Global model instance.
        """
        return self.global_model
