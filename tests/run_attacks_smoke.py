"""End-to-end privacy-attack smoke on a Pythia-1.4B-backed StepRewardModel.

scripts/run_federated.py does not currently read `dp.*` or `attacks.*` from
the YAML config, so this driver exercises the attack code paths directly:

    1. Load Pythia-1.4B + tokenizer (real).
    2. Wrap it in StepRewardModel; train the head for a couple of steps on a
       known "member" CoT sample so the model has actually fit something.
    3. Snapshot per-parameter head gradients on that member sample.
    4. Run GradientReconstructionAttack against those gradients (real Pythia
       embedding table for nearest-token decode).
    5. Run MembershipInferenceAttack on (member, non-member) and check the
       member score is higher (or report the gap).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fclprm.attacks.gradient_recon import GradientReconstructionAttack  # noqa: E402
from fclprm.attacks.membership import MembershipInferenceAttack  # noqa: E402
from fclprm.models.base_wrapper import StepRewardModel  # noqa: E402


def main() -> int:
    torch.manual_seed(0)
    device = "cpu"

    print("[load] Pythia-1.4b + StepRewardModel")
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained("EleutherAI/pythia-1.4b")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    backbone = AutoModel.from_pretrained(
        "EleutherAI/pythia-1.4b", dtype=torch.float32
    )
    model = StepRewardModel(backbone=backbone, head_dim=64).to(device)
    print(
        f"  loaded in {time.time()-t0:.1f}s "
        f"(trainable={sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6:.2f}M)"
    )

    member_text = "Step: Let x be the variable. Then x equals five."
    nonmember_text = "Step: The capital of France is the largest city in Europe."
    label_member = torch.tensor([1.0])
    label_nonmember = torch.tensor([0.0])

    member_enc = tok(
        member_text, return_tensors="pt", padding=True, truncation=True, max_length=32
    )
    nonmember_enc = tok(
        nonmember_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=32,
    )

    # --- DLG snapshot: use a FRESH-init head so the gradient signal is rich. -----
    # If we pre-train first, the head can fall into a dying-ReLU regime where
    # every pre-activation on this sample is negative → all upstream grads are
    # zero except mlp2.bias, and DLG has no per-token signal to invert.
    print("[snapshot] head gradients on the member sample (fresh head)")
    pred = model(member_enc["input_ids"], member_enc["attention_mask"])
    loss = F.mse_loss(pred, label_member)
    head_params = [p for n, p in model.named_parameters() if p.requires_grad]
    head_param_names = [n for n, p in model.named_parameters() if p.requires_grad]
    grads = torch.autograd.grad(loss, head_params)
    target_grads = {n: g.detach().clone() for n, g in zip(head_param_names, grads)}
    for n, g in target_grads.items():
        print(f"  {n:30s} shape={tuple(g.shape)}  norm={g.norm().item():.4f}")
    nonzero = [n for n, g in target_grads.items() if g.norm() > 1e-6]
    assert len(nonzero) >= 3, (
        f"only {len(nonzero)}/{len(target_grads)} head grads are non-zero — "
        "head is in a dying-ReLU state, DLG cannot recover anything"
    )

    # --- DLG reconstruction -----------------------------------------------------
    print("[DLG] reconstructing from head gradients (500 iters, lr=0.1)")
    t0 = time.time()
    attack = GradientReconstructionAttack(model, tok, device=device)
    out = attack.reconstruct(
        target_gradients=target_grads,
        max_steps=1,
        seq_length=int(member_enc["input_ids"].shape[1]),
        num_iterations=500,
        lr=0.1,
        verbose=False,
    )
    print(f"  done in {time.time()-t0:.1f}s")
    print(f"  final grad-distance: {out['final_distance']:.4f}")
    print(f"  reconstructed text:  {out['reconstructed_text']!r}")
    print(f"  reconstructed label: {out['reconstructed_labels']}")
    assert (
        out["final_distance"] < 1.0
    ), f"DLG distance unexpectedly stuck at {out['final_distance']:.4f}"

    # --- Train the head briefly so MIA has signal to detect ---------------------
    # lr=1e-3, 20 steps: gentle enough to actually fit the member sample
    # without blowing up the ReLU.
    print("[train] gentle fit for MIA (lr=1e-3, 20 steps)")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    pre_loss = post_loss = 0.0
    for it in range(20):
        opt.zero_grad()
        pred = model(member_enc["input_ids"], member_enc["attention_mask"])
        loss = F.mse_loss(pred, label_member)
        loss.backward()
        opt.step()
        if it == 0:
            pre_loss = float(loss.detach())
        post_loss = float(loss.detach())
    print(f"  member-loss {pre_loss:.4f} -> {post_loss:.4f}")
    assert post_loss < pre_loss, "member loss did not decrease — fit is broken"

    # --- Membership inference ---------------------------------------------------
    print("[MIA] loss-based scores on member vs non-member")
    mia = MembershipInferenceAttack(shadow_model=model, device=device)
    sample_member = {
        "input_ids": member_enc["input_ids"][0],
        "attention_mask": member_enc["attention_mask"][0],
        "label": float(label_member.item()),
    }
    sample_nonmember = {
        "input_ids": nonmember_enc["input_ids"][0],
        "attention_mask": nonmember_enc["attention_mask"][0],
        "label": float(label_nonmember.item()),
    }
    s_mem = mia.infer(model, sample_member, method="loss")
    s_non = mia.infer(model, sample_nonmember, method="loss")
    gap = s_mem - s_non
    print(f"  member score    : {s_mem:.4f}")
    print(f"  non-member score: {s_non:.4f}")
    print(f"  gap (mem - non) : {gap:+.4f}")

    print()
    print(
        "[done] privacy-attack smoke OK "
        f"(DLG dist={out['final_distance']:.4f}, MIA gap={gap:+.4f})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
