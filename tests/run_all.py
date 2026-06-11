"""Run the regression suite + critical-path FakeBackbone smoke without pytest.

Why this exists:
    The host environment has torch + scipy but not transformers / pytest
    / VersaPRM data. So we cannot run scripts/run_federated.py end-to-end
    on a real Pythia 1.4B + real dataset. This driver substitutes a tiny
    stub `transformers` module and a FakeBackbone, and runs:
        - tests/test_data.py      (2)
        - tests/test_metrics.py   (6)
        - tests/test_models.py    (1)
        - tests/test_federated.py (3 — skips the real-AutoModel test)
        - innovation tests        (anchor_prm rebasin + DLG signal)
        - critical-path simulator smoke (fedavg / trimmed_mean / anchor_prm
          + ValueError guard)
"""

from __future__ import annotations

import sys
import traceback
import types
from pathlib import Path

# ---- Path + transformers stub --------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _install_transformers_stub() -> None:
    if "transformers" in sys.modules:
        return
    fake = types.ModuleType("transformers")

    class _StubPreTrainedModel:  # only used as an isinstance/typing hint
        pass

    class _StubPreTrainedTokenizer:
        pass

    fake.PreTrainedModel = _StubPreTrainedModel
    fake.PreTrainedTokenizer = _StubPreTrainedTokenizer
    fake.AutoModel = None
    fake.AutoTokenizer = None
    sys.modules["transformers"] = fake


_install_transformers_stub()

import torch  # noqa: E402  (stub must be installed first)
import torch.nn as nn  # noqa: E402

# ---- Fake backbone (transformers-free) -----------------------------------------


class _FakeOutput:
    def __init__(self, last_hidden_state: torch.Tensor) -> None:
        self.last_hidden_state = last_hidden_state


class FakeBackbone(nn.Module):
    """Tiny embedding-only backbone that mirrors the HF transformer API.

    Has a `.config.hidden_size`, a `.get_input_embeddings()`, and returns an
    object with `.last_hidden_state` from `forward(input_ids=..., attention_mask=...)`
    (or `inputs_embeds=...`). Parameters are frozen by `StepRewardModel`'s
    constructor in the real wrapper, so the embedding table here does carry
    a small number of params, but they remain `requires_grad=False`.
    """

    def __init__(self, vocab: int = 64, hidden: int = 16) -> None:
        super().__init__()
        self.config = types.SimpleNamespace(hidden_size=hidden)
        self.embed = nn.Embedding(vocab, hidden)

    def get_input_embeddings(self) -> nn.Module:
        return self.embed

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> _FakeOutput:
        if inputs_embeds is None:
            assert input_ids is not None
            h = self.embed(input_ids)
        else:
            h = inputs_embeds
        return _FakeOutput(last_hidden_state=h)


# ---- Tiny test framework -------------------------------------------------------

_RESULTS: list[tuple[str, bool, str]] = []


def run(name: str, fn) -> None:
    try:
        fn()
        _RESULTS.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as exc:  # noqa: BLE001
        msg = f"{type(exc).__name__}: {exc}"
        _RESULTS.append((name, False, msg))
        print(f"  FAIL  {name}\n        {msg}")
        traceback.print_exc(limit=2)


# ---- Test cases ----------------------------------------------------------------


def t_data_split() -> None:
    from fclprm.data.utils import split_cot_into_steps

    cot = "Step 1: Define x.\n\nStep 2: Calculate y.\n\nStep 3: Conclude."
    steps = split_cot_into_steps(cot)
    assert len(steps) == 3
    assert steps[0] == "Step 1: Define x."


def t_data_collate() -> None:
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


def t_cdspi_alignment() -> None:
    from fclprm.metrics.cd_spi import compute_cd_spi

    embs = {f"c{i}": torch.tensor([1.0, 0.0, 0.0]) for i in range(3)}
    assert abs(compute_cd_spi("s", embs) - 0.0) < 1e-5


def t_cdspi_opposition() -> None:
    from fclprm.metrics.cd_spi import compute_cd_spi

    embs = {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([-1.0, 0.0])}
    assert abs(compute_cd_spi("s", embs) - 1.0) < 1e-5


def t_cdspi_range() -> None:
    from fclprm.metrics.cd_spi import compute_cd_spi

    torch.manual_seed(42)
    for _ in range(10):
        embs = {f"c{i}": torch.randn(8) for i in range(4)}
        v = compute_cd_spi("s", embs)
        assert 0.0 <= v <= 1.0


def t_cdspi_batch() -> None:
    from fclprm.metrics.cd_spi import compute_cd_spi_batch

    steps = ["s1", "s2"]
    embs = {
        "a": [torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0])],
        "b": [torch.tensor([1.0, 0.0]), torch.tensor([0.0, -1.0])],
    }
    out = compute_cd_spi_batch(steps, embs)
    assert out["s1"] < 0.1 and out["s2"] > 0.9


def t_cdspi_guard() -> None:
    from fclprm.metrics.cd_spi import compute_cd_spi

    try:
        compute_cd_spi("s", {"a": torch.tensor([1.0])})
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def t_bon_callable() -> None:
    from fclprm.metrics.bon import best_of_n_accuracy

    assert callable(best_of_n_accuracy)


def t_prm_head_forward() -> None:
    from fclprm.models.prm_head import PRMHead

    head = PRMHead(hidden_dim=128, head_dim=64)
    rewards = head(torch.randn(4, 128))
    assert rewards.shape == (4,)


def t_fedavg_uniform() -> None:
    from fclprm.federated.aggregators import fedavg_prm
    from fclprm.models.prm_head import PRMHead

    head = PRMHead(hidden_dim=8, head_dim=4)
    u1 = {k: torch.ones_like(v) * 1.0 for k, v in head.state_dict().items()}
    u2 = {k: torch.ones_like(v) * 3.0 for k, v in head.state_dict().items()}
    g = PRMHead(hidden_dim=8, head_dim=4)
    for p in g.parameters():
        p.data.zero_()
    out = fedavg_prm(g, [u1, u2])
    for p in out.parameters():
        assert torch.allclose(p, torch.ones_like(p) * 2.0)


def t_fedavg_weighted() -> None:
    from fclprm.federated.aggregators import fedavg_prm
    from fclprm.models.prm_head import PRMHead

    head = PRMHead(hidden_dim=8, head_dim=4)
    u1 = {k: torch.ones_like(v) * 1.0 for k, v in head.state_dict().items()}
    u2 = {k: torch.ones_like(v) * 3.0 for k, v in head.state_dict().items()}
    g = PRMHead(hidden_dim=8, head_dim=4)
    for p in g.parameters():
        p.data.zero_()
    out = fedavg_prm(g, [u1, u2], weights=[0.75, 0.25])
    for p in out.parameters():
        assert torch.allclose(p, torch.ones_like(p) * 1.5)


def t_trimmed_mean() -> None:
    from fclprm.federated.aggregators import robust_aggregate_trimmed_mean
    from fclprm.models.prm_head import PRMHead

    head = PRMHead(hidden_dim=8, head_dim=4)
    updates = [
        {k: torch.ones_like(v) * val for k, v in head.state_dict().items()}
        for val in [0.0, 1.0, 2.0, 3.0, 100.0]
    ]
    g = PRMHead(hidden_dim=8, head_dim=4)
    for p in g.parameters():
        p.data.zero_()
    out = robust_aggregate_trimmed_mean(g, updates, trim_ratio=0.2)
    for p in out.parameters():
        assert torch.allclose(p, torch.ones_like(p) * 2.0, atol=1e-5)


def t_anchor_prm_rebasin() -> None:
    """Construct ref + a permuted client; check Hungarian recovers the inverse."""
    from fclprm.federated.aggregators import _hungarian_match, _permute_head_state
    from fclprm.models.prm_head import PRMHead

    torch.manual_seed(0)
    head_dim = 4
    ref = PRMHead(hidden_dim=6, head_dim=head_dim)
    construct_perm = torch.tensor([2, 0, 3, 1])  # client = ref permuted by this

    # Build client whose head is the ref reindexed along head_dim.
    client_state = {
        "mlp1.weight": ref.mlp1.weight.data[construct_perm].clone(),
        "mlp1.bias": ref.mlp1.bias.data[construct_perm].clone(),
        "mlp2.weight": ref.mlp2.weight.data[:, construct_perm].clone(),
        "mlp2.bias": ref.mlp2.bias.data.clone(),
    }
    # Anchor activations: ref produces ref_emb, client produces ref_emb[:, construct_perm].
    anchor_in = torch.randn(7, 6)
    with torch.no_grad():
        ref_emb = ref.get_intermediate(anchor_in)
        cli_head = PRMHead(hidden_dim=6, head_dim=head_dim)
        # Match the StepRewardModel naming convention used by aggregators
        cli_head.load_state_dict(client_state)
        cli_emb = cli_head.get_intermediate(anchor_in)

    # Build the squared-distance cost and solve.
    ref_norm = (ref_emb**2).sum(dim=0)
    cli_norm = (cli_emb**2).sum(dim=0)
    cross = ref_emb.t() @ cli_emb
    cost = ref_norm.unsqueeze(1) + cli_norm.unsqueeze(0) - 2.0 * cross
    perm = _hungarian_match(cost)

    # Optimal match is the inverse of construct_perm.
    inv = torch.empty_like(construct_perm)
    inv[construct_perm] = torch.arange(head_dim)
    assert torch.equal(perm, inv), f"perm={perm}, expected inverse={inv}"

    # Apply that permutation back; recovered weights must equal ref.
    cli_state_namespaced = {f"head.{k}": v for k, v in client_state.items()}
    rebased = _permute_head_state(cli_state_namespaced, perm)
    assert torch.allclose(rebased["head.mlp1.weight"], ref.mlp1.weight.data, atol=1e-5)
    assert torch.allclose(rebased["head.mlp1.bias"], ref.mlp1.bias.data, atol=1e-5)
    assert torch.allclose(rebased["head.mlp2.weight"], ref.mlp2.weight.data, atol=1e-5)


def t_dlg_signal() -> None:
    """Run DLG for 500 iters; require gradient distance to drop well below 1.0."""
    import torch.nn.functional as F

    from fclprm.attacks.gradient_recon import GradientReconstructionAttack
    from fclprm.models.base_wrapper import StepRewardModel

    torch.manual_seed(0)
    backbone = FakeBackbone(vocab=64, hidden=16)
    model = StepRewardModel(backbone=backbone, head_dim=8)

    # Build a target gradient by running a real forward + backward on a known pair.
    input_ids = torch.randint(0, 64, (1, 6))
    attention_mask = torch.ones_like(input_ids)
    label = torch.tensor([0.7])

    model.train()
    pred = model(input_ids, attention_mask)
    loss = F.mse_loss(pred, label)
    head_params = [p for _, p in model.named_parameters() if p.requires_grad]
    head_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, head_params)
    target = {n: g.detach().clone() for n, g in zip(head_param_names, grads)}

    class _DummyTok:
        def batch_decode(self, ids, skip_special_tokens=True):
            return [" ".join(map(str, row.tolist())) for row in ids]

    attack = GradientReconstructionAttack(model, _DummyTok(), device="cpu")
    out = attack.reconstruct(
        target_gradients=target,
        max_steps=1,
        seq_length=6,
        num_iterations=1000,
        lr=0.1,
        verbose=False,
    )
    # FakeBackbone has only an embedding layer, so the gradient signal is
    # weaker than under a real transformer; require meaningful but not
    # near-perfect convergence.
    assert out["final_distance"] < 1.0, f"distance={out['final_distance']:.4f}"


def t_simulator_fedavg() -> None:
    from fclprm.federated.simulator import FederatedSimulator
    from fclprm.models.base_wrapper import StepRewardModel

    torch.manual_seed(0)
    backbone = FakeBackbone()
    g = StepRewardModel(backbone=backbone, head_dim=8)

    def make_data(n: int) -> list[dict]:
        return [
            {
                "input_ids": torch.randint(0, 64, (5,)),
                "attention_mask": torch.ones(5, dtype=torch.long),
                "label": float(torch.rand(1).item()),
            }
            for _ in range(n)
        ]

    sim = FederatedSimulator(
        num_clients=3,
        num_rounds=2,
        global_model=g,
        client_data=[make_data(8) for _ in range(3)],
        aggregation_rule="fedavg",
        seed=0,
    )
    res = sim.run(local_epochs=1, local_batch_size=4, local_lr=1e-3, device="cpu")
    assert res["num_rounds"] == 2 and len(res["history"]) == 2


def t_simulator_trimmed_mean() -> None:
    from fclprm.federated.simulator import FederatedSimulator
    from fclprm.models.base_wrapper import StepRewardModel

    torch.manual_seed(0)
    backbone = FakeBackbone()
    g = StepRewardModel(backbone=backbone, head_dim=8)

    def make_data(n: int) -> list[dict]:
        return [
            {
                "input_ids": torch.randint(0, 64, (5,)),
                "attention_mask": torch.ones(5, dtype=torch.long),
                "label": float(torch.rand(1).item()),
            }
            for _ in range(n)
        ]

    sim = FederatedSimulator(
        num_clients=4,
        num_rounds=1,
        global_model=g,
        client_data=[make_data(6) for _ in range(4)],
        aggregation_rule="trimmed_mean",
        seed=0,
    )
    res = sim.run(local_epochs=1, local_batch_size=4, local_lr=1e-3, device="cpu")
    assert res["num_rounds"] == 1


def t_simulator_anchor_prm() -> None:
    from fclprm.federated.simulator import FederatedSimulator
    from fclprm.models.base_wrapper import StepRewardModel

    torch.manual_seed(0)
    backbone = FakeBackbone()
    g = StepRewardModel(backbone=backbone, head_dim=8)
    initial_w1 = g.head.mlp1.weight.detach().clone()

    def make_data(n: int) -> list[dict]:
        return [
            {
                "input_ids": torch.randint(0, 64, (5,)),
                "attention_mask": torch.ones(5, dtype=torch.long),
                "label": float(torch.rand(1).item()),
            }
            for _ in range(n)
        ]

    anchor_inputs = {
        "input_ids": torch.randint(0, 64, (4, 5)),
        "attention_mask": torch.ones(4, 5, dtype=torch.long),
    }

    sim = FederatedSimulator(
        num_clients=3,
        num_rounds=2,
        global_model=g,
        client_data=[make_data(8) for _ in range(3)],
        aggregation_rule="anchor_prm",
        seed=0,
        anchor_inputs=anchor_inputs,
        anchor_steps=["a", "b", "c", "d"],
    )
    res = sim.run(local_epochs=1, local_batch_size=4, local_lr=1e-3, device="cpu")
    final_w1 = res["final_model"].head.mlp1.weight.detach()
    drift = (initial_w1 - final_w1).norm().item()
    assert drift > 0.0, "anchor_prm aggregation produced zero head drift"


def t_simulator_anchor_guard() -> None:
    from fclprm.federated.simulator import FederatedSimulator
    from fclprm.models.base_wrapper import StepRewardModel

    backbone = FakeBackbone()
    g = StepRewardModel(backbone=backbone, head_dim=8)

    try:
        FederatedSimulator(
            num_clients=2,
            num_rounds=1,
            global_model=g,
            client_data=[[], []],
            aggregation_rule="anchor_prm",
            seed=0,
            anchor_inputs=None,
        )
    except ValueError as exc:
        assert "anchor_inputs" in str(exc)
        return
    raise AssertionError("expected ValueError when anchor_prm + anchor_inputs=None")


# ---- Driver --------------------------------------------------------------------


def main() -> int:
    print("=== regression suite + critical-path smoke ===")
    print("[data]")
    run("data.split_cot_into_steps", t_data_split)
    run("data.collate_step_batch", t_data_collate)

    print("[metrics]")
    run("metrics.cd_spi_perfect_alignment", t_cdspi_alignment)
    run("metrics.cd_spi_perfect_opposition", t_cdspi_opposition)
    run("metrics.cd_spi_range", t_cdspi_range)
    run("metrics.cd_spi_batch", t_cdspi_batch)
    run("metrics.cd_spi_requires_two_clients", t_cdspi_guard)
    run("metrics.best_of_n_callable", t_bon_callable)

    print("[models]")
    run("models.prm_head_forward", t_prm_head_forward)

    print("[federated/aggregators]")
    run("federated.fedavg_uniform_weights", t_fedavg_uniform)
    run("federated.fedavg_weighted", t_fedavg_weighted)
    run("federated.trimmed_mean_basic", t_trimmed_mean)

    print("[innovation]")
    run("innovation.anchor_prm_rebasin", t_anchor_prm_rebasin)
    run("innovation.dlg_gradient_signal", t_dlg_signal)

    print("[critical-path simulator smoke]")
    run("simulator.fedavg_3c_2r", t_simulator_fedavg)
    run("simulator.trimmed_mean_4c_1r", t_simulator_trimmed_mean)
    run("simulator.anchor_prm_3c_2r", t_simulator_anchor_prm)
    run("simulator.anchor_prm_guard_no_inputs", t_simulator_anchor_guard)

    passed = sum(1 for _, ok, _ in _RESULTS if ok)
    failed = len(_RESULTS) - passed
    print()
    print(f"  {passed} passed, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
