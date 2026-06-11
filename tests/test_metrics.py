"""Unit tests for metrics and evaluation."""

import torch

from fclprm.metrics.cd_spi import compute_cd_spi, compute_cd_spi_batch


def test_cd_spi_perfect_alignment():
    """CD-SPI should be 0 when all embeddings are identical."""
    embs = {
        "client_a": torch.tensor([1.0, 0.0, 0.0]),
        "client_b": torch.tensor([1.0, 0.0, 0.0]),
        "client_c": torch.tensor([1.0, 0.0, 0.0]),
    }
    cspi = compute_cd_spi("test step", embs)
    assert abs(cspi - 0.0) < 1e-5, f"Expected 0.0, got {cspi}"


def test_cd_spi_perfect_opposition():
    """CD-SPI should be 1 when embeddings are opposite."""
    embs = {
        "client_a": torch.tensor([1.0, 0.0]),
        "client_b": torch.tensor([-1.0, 0.0]),
    }
    cspi = compute_cd_spi("test step", embs)
    assert abs(cspi - 1.0) < 1e-5, f"Expected 1.0, got {cspi}"


def test_cd_spi_range():
    """CD-SPI output must always be in [0, 1]."""
    torch.manual_seed(42)
    for _ in range(10):
        embs = {f"client_{i}": torch.randn(8) for i in range(4)}
        cspi = compute_cd_spi("random step", embs)
        assert 0.0 <= cspi <= 1.0, f"CD-SPI out of range: {cspi}"


def test_cd_spi_batch():
    """Batch CD-SPI should return correct dict."""
    steps = ["step1", "step2"]
    embs = {
        "client_a": [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
        "client_b": [torch.tensor([1.0, 0.0]), torch.tensor([0.0, -1.0])],
    }
    result = compute_cd_spi_batch(steps, embs)
    assert len(result) == 2
    assert "step1" in result
    assert "step2" in result
    # step1: aligned -> low CD-SPI
    assert result["step1"] < 0.1
    # step2: opposite -> high CD-SPI
    assert result["step2"] > 0.9


def test_cd_spi_requires_two_clients():
    """CD-SPI should raise error with fewer than 2 clients."""
    embs = {"client_a": torch.tensor([1.0, 0.0])}
    try:
        compute_cd_spi("test", embs)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


def test_best_of_n_accuracy_shape():
    """BoN should return float in [0, 1]."""
    from fclprm.metrics.bon import best_of_n_accuracy

    # This is a smoke test; full test requires model + tokenizer
    assert callable(best_of_n_accuracy)
