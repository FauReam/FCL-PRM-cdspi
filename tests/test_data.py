"""Unit tests for data loading and preprocessing."""

import json
import tempfile
from pathlib import Path

import torch


def test_split_cot_into_steps():
    """Test basic CoT step splitting."""
    from fclprm.data.utils import split_cot_into_steps

    cot = "Step 1: Define x.\n\nStep 2: Calculate y.\n\nStep 3: Conclude."
    steps = split_cot_into_steps(cot)
    assert len(steps) == 3
    assert steps[0] == "Step 1: Define x."


def test_collate_step_batch():
    """Test batch collation with padding."""
    from fclprm.data.utils import collate_step_batch

    batch = [
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
    ]
    collated = collate_step_batch(batch)
    assert collated["input_ids"].shape == (2, 3)
    assert collated["labels"].shape == (2,)


def test_normalize_labels():
    """Test label normalization handles multiple formats."""
    from fclprm.data.utils import _normalize_labels

    assert _normalize_labels([1, 0, -1, 2]) == [1.0, 0.0, 0.0, 1.0]
    assert _normalize_labels(["+", "-", "correct", "incorrect", "1", "0"]) == [
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
    ]
    assert _normalize_labels([1.0, 0.0]) == [1.0, 0.0]
    assert _normalize_labels([None, []]) == [0.0, 0.0]


def test_load_jsonl_or_json():
    """Test loading from both JSONL and JSON formats."""
    from fclprm.data.utils import _load_jsonl_or_json

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # JSONL format
        jsonl_file = tmp_path / "test.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            f.write(json.dumps({"a": 1}) + "\n")
            f.write(json.dumps({"b": 2}) + "\n")

        samples = _load_jsonl_or_json(tmp_path, "test")
        assert len(samples) == 2
        assert samples[0]["a"] == 1

        # JSON format fallback
        json_file = tmp_path / "fallback.json"
        with open(json_file, "w", encoding="utf-8") as f:
            json.dump([{"c": 3}], f)

        samples = _load_jsonl_or_json(tmp_path, "fallback")
        assert len(samples) == 1
        assert samples[0]["c"] == 3

        # Missing file
        try:
            _load_jsonl_or_json(tmp_path, "nonexistent")
            assert False, "Should raise FileNotFoundError"
        except FileNotFoundError:
            pass


def test_normalize_dataset():
    """Test in-place dataset normalization."""
    from fclprm.data.utils import _normalize_dataset

    samples = [
        {"question": "q1", "labels": [1, 0]},
        {"question": "q2"},  # no labels key
    ]
    result = _normalize_dataset(samples)
    assert result[0]["labels"] == [1.0, 0.0]
    assert "labels" not in result[1]


def test_prm800k_loader():
    """Test PRM800KLoader loads and normalizes data correctly."""
    from fclprm.data.prm800k import PRM800KLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "train.jsonl"
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "question": "What is 2+2?",
                        "steps": ["Let x = 2.", "Add 2 to x.", "x = 4."],
                        "labels": [1, 1, 1],
                    }
                )
                + "\n"
            )

        loader = PRM800KLoader(data_dir=tmpdir, split="train")
        samples = loader.load()
        assert len(samples) == 1
        assert samples[0]["question"] == "What is 2+2?"
        assert len(samples[0]["steps"]) == 3
        assert samples[0]["labels"] == [1.0, 1.0, 1.0]

        # Test caching
        samples2 = loader.load()
        assert samples2 is samples


def test_versa_loader_domains():
    """Test VersaPRMLoader discovers domains dynamically."""
    from fclprm.data.versa_loader import VersaPRMLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "versa_prm.jsonl"
        with open(data_file, "w", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "domain": "math",
                        "question": "q1",
                        "steps": ["s1"],
                        "labels": [1],
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "domain": "code",
                        "question": "q2",
                        "steps": ["s2"],
                        "labels": [0],
                    }
                )
                + "\n"
            )
            f.write(
                json.dumps(
                    {
                        "domain": "math",
                        "question": "q3",
                        "steps": ["s3"],
                        "labels": [1],
                    }
                )
                + "\n"
            )

        loader = VersaPRMLoader(data_dir=tmpdir)
        assert loader.domains == ["code", "math"]

        math_samples = loader.load_domain("math")
        assert len(math_samples) == 2

        code_samples = loader.load_domain("code")
        assert len(code_samples) == 1


def test_versa_loader_federated_splits():
    """Test federated splitting logic for various client/domain ratios."""
    from fclprm.data.versa_loader import VersaPRMLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "versa_prm.jsonl"
        with open(data_file, "w", encoding="utf-8") as f:
            for i in range(6):
                domain = "math" if i < 4 else "code"
                f.write(
                    json.dumps(
                        {
                            "domain": domain,
                            "question": f"q{i}",
                            "steps": [f"s{i}"],
                            "labels": [1],
                        }
                    )
                    + "\n"
                )

        loader = VersaPRMLoader(data_dir=tmpdir)

        # 2 clients <= 2 domains: one domain per client
        splits = loader.get_federated_splits(num_clients=2, seed=42)
        assert len(splits) == 2
        assert len(splits[0]) + len(splits[1]) == 6

        # 4 clients > 2 domains: split domain data
        splits = loader.get_federated_splits(num_clients=4, seed=42)
        assert len(splits) == 4
        total = sum(len(s) for s in splits)
        assert total == 6

        # 0 clients edge case
        splits = loader.get_federated_splits(num_clients=0)
        assert splits == []


def test_med_loader():
    """Test MedPRMBenchLoader loads medical data correctly."""
    from fclprm.data.med_loader import MedPRMBenchLoader

    with tempfile.TemporaryDirectory() as tmpdir:
        data_file = Path(tmpdir) / "med_prm.json"
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(
                [
                    {
                        "question": "Patient presents with fever...",
                        "steps": ["Check vitals.", "Order blood test."],
                        "labels": [1, 0],
                    }
                ],
                f,
            )

        loader = MedPRMBenchLoader(data_dir=tmpdir)
        samples = loader.load()
        assert len(samples) == 1
        assert samples[0]["labels"] == [1.0, 0.0]
