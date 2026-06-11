"""End-to-end smoke test: tiny model, tiny data, 1 round federated training.

This test validates that the full federated pipeline executes without error
using the smallest possible configuration, serving as a regression checkpoint
for code-path integrity.
"""

import tempfile
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from fclprm.data.utils import collate_step_batch
from fclprm.federated.simulator import FederatedSimulator
from fclprm.models.base_wrapper import StepRewardModel
from fclprm.utils.seed import set_seed


def _make_tiny_step_data(tokenizer, num_samples: int = 4) -> list[dict]:
    """Create tiny step-level samples for smoke testing."""
    samples = []
    for i in tqdm(range(num_samples), desc="  Generating tiny test data", leave=False):
        text = f"Step {i}: Let x = {i}. Therefore, x + 1 = {i + 1}."
        encoded = tokenizer(
            text,
            padding=False,
            truncation=True,
            max_length=32,
            return_tensors=None,
        )
        samples.append(
            {
                "input_ids": torch.tensor(encoded["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(
                    encoded["attention_mask"], dtype=torch.long
                ),
                "label": 1.0 if i % 2 == 0 else 0.0,
            }
        )
    return samples


def test_end_to_end_federated_smoke():
    """Run a full federated simulation with tiny-gpt2 backbone."""
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModel.from_pretrained("sshleifer/tiny-gpt2")
    global_model = StepRewardModel(backbone=backbone, head_dim=8)

    # Two clients with tiny data
    client_data = [
        _make_tiny_step_data(tokenizer, num_samples=3),
        _make_tiny_step_data(tokenizer, num_samples=3),
    ]

    simulator = FederatedSimulator(
        num_clients=2,
        num_rounds=1,
        global_model=global_model,
        client_data=client_data,
        aggregation_rule="fedavg",
        seed=42,
    )

    results = simulator.run(
        local_epochs=1,
        local_batch_size=2,
        local_lr=1e-3,
        device="cpu",
    )

    assert results["num_rounds"] == 1
    assert results["num_clients"] == 2
    assert len(results["history"]) == 1
    assert "avg_loss" in results["history"][0]
    assert results["history"][0]["avg_loss"] >= 0.0

    # Verify final model exists and has trainable head
    final_model = results["final_model"]
    assert final_model is not None
    head_params = [p for p in final_model.head.parameters() if p.requires_grad]
    assert len(head_params) > 0


def test_end_to_end_anchor_prm_smoke():
    """Run Anchor-PRM aggregation with tiny-gpt2."""
    set_seed(42)

    tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    backbone = AutoModel.from_pretrained("sshleifer/tiny-gpt2")
    global_model = StepRewardModel(backbone=backbone, head_dim=8)

    client_data = [
        _make_tiny_step_data(tokenizer, num_samples=3),
        _make_tiny_step_data(tokenizer, num_samples=3),
    ]

    anchor_steps = [
        "Let x be the variable.",
        "Therefore, the answer is positive.",
    ]
    anchor_encoded = tokenizer(
        anchor_steps,
        padding=True,
        truncation=True,
        max_length=32,
        return_tensors="pt",
    )
    anchor_inputs = {
        "input_ids": anchor_encoded["input_ids"],
        "attention_mask": anchor_encoded["attention_mask"],
    }

    simulator = FederatedSimulator(
        num_clients=2,
        num_rounds=1,
        global_model=global_model,
        client_data=client_data,
        aggregation_rule="anchor_prm",
        seed=42,
        anchor_inputs=anchor_inputs,
        anchor_steps=anchor_steps,
    )

    results = simulator.run(
        local_epochs=1,
        local_batch_size=2,
        local_lr=1e-3,
        device="cpu",
    )

    assert results["num_rounds"] == 1
    assert len(results["history"]) == 1
    # Anchor-PRM should not raise and should produce a valid loss
    assert results["history"][0]["avg_loss"] >= 0.0


def test_collate_integration():
    """Test that collate_step_batch works end-to-end with DataLoader."""
    from torch.utils.data import DataLoader

    samples = [
        {
            "input_ids": torch.tensor([1, 2, 3]),
            "attention_mask": torch.tensor([1, 1, 1]),
            "label": 1.0,
        },
        {
            "input_ids": torch.tensor([4, 5]),
            "attention_mask": torch.tensor([1, 1]),
            "label": 0.0,
        },
        {
            "input_ids": torch.tensor([6, 7, 8, 9]),
            "attention_mask": torch.tensor([1, 1, 1, 1]),
            "label": 1.0,
        },
    ]

    loader = DataLoader(samples, batch_size=2, collate_fn=collate_step_batch)
    batches = list(loader)

    assert len(batches) == 2
    # First batch: 2 samples, max length 3
    assert batches[0]["input_ids"].shape == (2, 3)
    assert batches[0]["attention_mask"].shape == (2, 3)
    assert batches[0]["labels"].shape == (2,)
    # Second batch: 1 sample, length 4
    assert batches[1]["input_ids"].shape == (1, 4)


def test_checkpoint_save_load():
    """Test checkpoint round-trip."""
    from fclprm.models.checkpoint import load_checkpoint, save_checkpoint
    from fclprm.models.prm_head import PRMHead

    head = PRMHead(hidden_dim=8, head_dim=4)
    optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)

    with tempfile.TemporaryDirectory() as tmpdir:
        path = save_checkpoint(
            model=head,
            optimizer=optimizer,
            round_num=5,
            client_id=2,
            milestone="M3",
            save_dir=tmpdir,
        )
        assert Path(path).exists()

        head2 = PRMHead(hidden_dim=8, head_dim=4)
        optimizer2 = torch.optim.Adam(head2.parameters(), lr=1e-3)
        meta = load_checkpoint(path, head2, optimizer2)

        assert meta["round_num"] == 5
        assert meta["client_id"] == 2
        assert meta["milestone"] == "M3"

        # Parameters should match
        for p1, p2 in zip(head.parameters(), head2.parameters()):
            assert torch.allclose(p1, p2)


def test_config_hash_reproducibility():
    """Test that identical configs produce identical hashes."""
    import tempfile

    from fclprm.utils.config import ExperimentConfig

    config_dict = {
        "model": {"backbone": "test-model", "head_dim": 256},
        "training": {"lr": 1e-4, "epochs": 3},
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        path1 = Path(tmpdir) / "cfg1.yaml"
        path2 = Path(tmpdir) / "cfg2.yaml"

        import yaml

        with open(path1, "w") as f:
            yaml.dump(config_dict, f)
        with open(path2, "w") as f:
            yaml.dump(config_dict, f)

        cfg1 = ExperimentConfig(str(path1))
        cfg2 = ExperimentConfig(str(path2))

        assert cfg1.hash() == cfg2.hash()
        assert cfg1.get("model.backbone") == "test-model"
        assert cfg1.require("model.head_dim") == 256

        missing = cfg1.validate_keys(["model.backbone", "nonexistent.key"])
        assert missing == ["nonexistent.key"]
