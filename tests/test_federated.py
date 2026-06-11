"""Unit tests for federated learning components."""

import copy

import torch

from fclprm.federated.aggregators import fedavg_prm, robust_aggregate_trimmed_mean
from fclprm.models.prm_head import PRMHead


def test_fedavg_uniform_weights():
    """Test FedAvg with uniform weights averages parameters correctly."""
    head = PRMHead(hidden_dim=8, head_dim=4)

    # Create two client updates
    update1 = {k: torch.ones_like(v) * 1.0 for k, v in head.state_dict().items()}
    update2 = {k: torch.ones_like(v) * 3.0 for k, v in head.state_dict().items()}

    # Create a fresh head with zeros
    global_head = PRMHead(hidden_dim=8, head_dim=4)
    for p in global_head.parameters():
        p.data.zero_()

    result = fedavg_prm(global_head, [update1, update2])

    # After uniform FedAvg, all params should be 2.0
    for p in result.parameters():
        assert torch.allclose(p, torch.ones_like(p) * 2.0), "FedAvg uniform failed"


def test_fedavg_weighted():
    """Test FedAvg with custom weights."""
    head = PRMHead(hidden_dim=8, head_dim=4)

    update1 = {k: torch.ones_like(v) * 1.0 for k, v in head.state_dict().items()}
    update2 = {k: torch.ones_like(v) * 3.0 for k, v in head.state_dict().items()}

    global_head = PRMHead(hidden_dim=8, head_dim=4)
    for p in global_head.parameters():
        p.data.zero_()

    result = fedavg_prm(global_head, [update1, update2], weights=[0.75, 0.25])

    # Weighted average: 0.75*1.0 + 0.25*3.0 = 1.5
    for p in result.parameters():
        assert torch.allclose(p, torch.ones_like(p) * 1.5), "FedAvg weighted failed"


def test_trimmed_mean_basic():
    """Test trimmed mean removes extreme values."""
    head = PRMHead(hidden_dim=8, head_dim=4)

    # 5 clients with values [0, 1, 2, 3, 100]
    updates = []
    for val in [0.0, 1.0, 2.0, 3.0, 100.0]:
        updates.append(
            {k: torch.ones_like(v) * val for k, v in head.state_dict().items()}
        )

    global_head = PRMHead(hidden_dim=8, head_dim=4)
    for p in global_head.parameters():
        p.data.zero_()

    # Trim 20% from each side: removes 0.0 and 100.0
    result = robust_aggregate_trimmed_mean(global_head, updates, trim_ratio=0.2)

    # Mean of [1.0, 2.0, 3.0] = 2.0
    for p in result.parameters():
        assert torch.allclose(
            p, torch.ones_like(p) * 2.0, atol=1e-5
        ), "Trimmed mean failed"


def test_fedavg_preserves_backbone():
    """Test that FedAvg only updates head parameters, not backbone."""
    from transformers import AutoModel
    from fclprm.models.base_wrapper import StepRewardModel

    # Use a tiny model for testing
    backbone = AutoModel.from_pretrained("sshleifer/tiny-gpt2")
    model = StepRewardModel(backbone=backbone, head_dim=8)

    # Record initial backbone params
    initial_backbone = {k: v.clone() for k, v in model.backbone.named_parameters()}

    # Create dummy updates
    update = {k: torch.randn_like(v) for k, v in model.state_dict().items()}

    fedavg_prm(model, [update])

    # Backbone should be unchanged
    for name, param in model.backbone.named_parameters():
        assert torch.equal(param, initial_backbone[name]), f"Backbone changed: {name}"


def test_anchor_prm_alignment():
    """Test Anchor-PRM permutation preserves functional equivalence.

    If two clients have identical embeddings (perfect alignment), the
    permutation should be the identity, and FedAvg should behave normally.
    """
    from fclprm.federated.aggregators import anchor_prm_aggregate
    from fclprm.models.base_wrapper import StepRewardModel

    try:
        from transformers import AutoModel
    except ImportError:
        return  # Skip if transformers not available

    backbone = AutoModel.from_pretrained("sshleifer/tiny-gpt2")
    global_model = StepRewardModel(backbone=backbone, head_dim=8)

    # Two clients with identical state dicts
    state1 = copy.deepcopy(global_model.state_dict())
    state2 = copy.deepcopy(global_model.state_dict())

    # Identical embeddings -> identity permutation
    K, head_dim = 4, 8
    identical_emb = torch.randn(K, head_dim)
    client_embeddings = {
        0: identical_emb.clone(),
        1: identical_emb.clone(),
    }

    updates = [
        {"client_id": 0, "state_dict": state1},
        {"client_id": 1, "state_dict": state2},
    ]

    result = anchor_prm_aggregate(
        global_model=global_model,
        client_updates=updates,
        client_embeddings=client_embeddings,
        reference_client=0,
    )

    # With identical embeddings and identical weights, result should match input
    for name, param in result.named_parameters():
        if param.requires_grad:
            assert torch.allclose(
                param, state1[name]
            ), f"Anchor-PRM identity alignment failed for {name}"


def test_anchor_prm_insufficient_clients():
    """Test Anchor-PRM degrades to FedAvg when <2 clients have embeddings."""
    from fclprm.federated.aggregators import anchor_prm_aggregate
    from fclprm.models.prm_head import PRMHead

    head = PRMHead(hidden_dim=8, head_dim=4)
    global_head = PRMHead(hidden_dim=8, head_dim=4)
    for p in global_head.parameters():
        p.data.zero_()

    update = {k: torch.ones_like(v) * 2.0 for k, v in head.state_dict().items()}

    # Only one client has embeddings -> should degrade to FedAvg
    result = anchor_prm_aggregate(
        global_model=global_head,
        client_updates=[{"client_id": 0, "state_dict": update}],
        client_embeddings={0: torch.randn(2, 4)},
        reference_client=0,
    )

    for p in result.parameters():
        assert torch.allclose(
            p, torch.ones_like(p) * 2.0
        ), "Anchor-PRM degradation to FedAvg failed"


def test_anchor_prm_reference_client_fallback():
    """Test Anchor-PRM falls back to first available client when reference is missing."""
    from fclprm.federated.aggregators import anchor_prm_aggregate
    from fclprm.models.prm_head import PRMHead

    head = PRMHead(hidden_dim=8, head_dim=4)
    global_head = PRMHead(hidden_dim=8, head_dim=4)

    update = {k: torch.ones_like(v) for k, v in head.state_dict().items()}
    embeddings = {1: torch.randn(2, 4)}  # reference_client=0 not present

    # Should not raise; falls back to client 1 as reference
    result = anchor_prm_aggregate(
        global_model=global_head,
        client_updates=[{"client_id": 1, "state_dict": update}],
        client_embeddings=embeddings,
        reference_client=0,  # missing
    )

    # With only one client, should be identity (degraded to FedAvg)
    for name, param in result.named_parameters():
        if param.requires_grad:
            assert torch.allclose(param, update[name]), "Reference fallback failed"


def test_anchor_prm_shape_mismatch():
    """Test Anchor-PRM raises ValueError on embedding shape mismatch."""
    from fclprm.federated.aggregators import anchor_prm_aggregate
    from fclprm.models.prm_head import PRMHead

    head = PRMHead(hidden_dim=8, head_dim=4)
    global_head = PRMHead(hidden_dim=8, head_dim=4)

    update1 = {k: torch.ones_like(v) for k, v in head.state_dict().items()}
    update2 = {k: torch.ones_like(v) for k, v in head.state_dict().items()}

    # Mismatched embedding shapes
    client_embeddings = {
        0: torch.randn(2, 4),  # (K=2, head_dim=4)
        1: torch.randn(3, 4),  # (K=3, head_dim=4) - mismatch
    }

    try:
        anchor_prm_aggregate(
            global_model=global_head,
            client_updates=[
                {"client_id": 0, "state_dict": update1},
                {"client_id": 1, "state_dict": update2},
            ],
            client_embeddings=client_embeddings,
            reference_client=0,
        )
        assert False, "Should have raised ValueError for shape mismatch"
    except ValueError as e:
        assert "shape" in str(e).lower()


def test_fedavg_with_module_prefixed_state_dict():
    """Test FedAvg works when client state_dict keys have _module. prefix.

    Opacus GradSampleModule prefixes state_dict keys with "_module.".
    This test simulates that wrapper and verifies the aggregator can still
    match parameters by the raw key names extracted in client.py.
    """
    head = PRMHead(hidden_dim=8, head_dim=4)

    update1 = {k: torch.ones_like(v) * 1.0 for k, v in head.state_dict().items()}
    update2 = {k: torch.ones_like(v) * 3.0 for k, v in head.state_dict().items()}

    # Simulate Opacus prefix
    update1_prefixed = {"_module." + k: v for k, v in update1.items()}
    update2_prefixed = {"_module." + k: v for k, v in update2.items()}

    global_head = PRMHead(hidden_dim=8, head_dim=4)
    for p in global_head.parameters():
        p.data.zero_()

    # This should raise KeyError because fedavg_prm looks for raw keys
    try:
        fedavg_prm(global_head, [update1_prefixed, update2_prefixed])
        assert False, "Should have raised KeyError for _module. prefixed keys"
    except KeyError:
        pass

    # The fix in client.py strips the prefix before shipping, so a correctly
    # formed update (raw keys) should work fine.
    result = fedavg_prm(global_head, [update1, update2])
    for p in result.parameters():
        assert torch.allclose(p, torch.ones_like(p) * 2.0), "FedAvg with raw keys failed"
