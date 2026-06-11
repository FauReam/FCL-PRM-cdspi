"""Unit tests for PRM model components."""


def test_prm_head_forward():
    """Test PRM head produces scalar rewards."""
    import torch
    from fclprm.models.prm_head import PRMHead

    head = PRMHead(hidden_dim=128, head_dim=64)
    hidden = torch.randn(4, 128)  # (batch, hidden)
    rewards = head(hidden)
    assert rewards.shape == (4,)


def test_prm_head_3d_input():
    """Test PRM head handles 3D hidden_states by taking the last token."""
    import torch
    from fclprm.models.prm_head import PRMHead

    head = PRMHead(hidden_dim=128, head_dim=64)
    hidden_3d = torch.randn(4, 10, 128)  # (batch, seq_len, hidden)
    rewards = head(hidden_3d)
    assert rewards.shape == (4,)


def test_prm_head_intermediate():
    """Test get_intermediate returns post-ReLU activations of correct shape."""
    import torch
    from fclprm.models.prm_head import PRMHead

    head = PRMHead(hidden_dim=128, head_dim=64)
    hidden = torch.randn(4, 128)
    inter = head.get_intermediate(hidden)
    assert inter.shape == (4, 64)
    # ReLU should zero out negative values
    assert (inter >= 0).all()


def test_last_non_pad_hidden():
    """Test _last_non_pad_hidden picks the correct token for each sequence."""
    import torch
    from fclprm.models.base_wrapper import StepRewardModel

    try:
        from transformers import AutoModel
    except ImportError:
        return  # Skip if transformers not available

    backbone = AutoModel.from_pretrained("sshleifer/tiny-gpt2")
    model = StepRewardModel(backbone=backbone, head_dim=8)

    hidden = torch.randn(2, 5, backbone.config.hidden_size)
    attention_mask = torch.tensor(
        [
            [1, 1, 1, 0, 0],  # seq_len = 3, last valid at index 2
            [1, 1, 1, 1, 1],  # seq_len = 5, last valid at index 4
        ]
    )

    result = StepRewardModel._last_non_pad_hidden(hidden, attention_mask)
    assert result.shape == (2, backbone.config.hidden_size)
    assert torch.equal(result[0], hidden[0, 2])
    assert torch.equal(result[1], hidden[1, 4])


def test_step_embedding_vs_head_embedding():
    """Test get_step_embedding returns backbone hidden; get_head_embedding returns head features."""
    import torch
    from fclprm.models.base_wrapper import StepRewardModel

    try:
        from transformers import AutoModel
    except ImportError:
        return  # Skip if transformers not available

    backbone = AutoModel.from_pretrained("sshleifer/tiny-gpt2")
    model = StepRewardModel(backbone=backbone, head_dim=8)

    input_ids = torch.tensor([[1, 2, 3, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 0, 0]])

    step_emb = model.get_step_embedding(input_ids, attention_mask)
    head_emb = model.get_head_embedding(input_ids, attention_mask)

    # Step embedding: backbone hidden_dim
    assert step_emb.shape == (1, backbone.config.hidden_size)
    # Head embedding: head_dim (post-ReLU)
    assert head_emb.shape == (1, 8)
    # They should differ because head_emb passes through trainable MLP + ReLU
    assert not torch.equal(step_emb, head_emb)
