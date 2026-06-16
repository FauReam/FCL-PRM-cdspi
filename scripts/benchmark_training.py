#!/usr/bin/env python3
"""Training speed benchmark for Pythia-1.4B head-only on GB10.

Measures per-batch time after the efficiency fixes (autocast, in-memory
eval, inference_mode, pin_memory).  Compare against pre-fix baseline:
~10-12 s/batch, ~11.5 samples/s (from live run logs).
"""

import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoModel
from tqdm import tqdm

from fclprm.models.base_wrapper import StepRewardModel

# ---------------------------------------------------------------------------
# Config — matches m3_fedavg_head_1.4b_gelu.yaml
# ---------------------------------------------------------------------------
MODEL_NAME = "EleutherAI/pythia-1.4b"
HEAD_DIM = 256
HEAD_ACT = "gelu"
BATCH_SIZE = 128
MAX_LENGTH = 384
WARMUP_BATCHES = 10
TIMED_BATCHES = 50
DEVICE = "cuda"
SEED = 42

torch.manual_seed(SEED)
torch.backends.cudnn.benchmark = True


def make_synthetic_data(num_samples: int, vocab_size: int, max_len: int):
    """Create synthetic step-level data matching the real dataset format."""
    input_ids = torch.randint(1, vocab_size - 1, (num_samples, max_len))
    # Variable-length: 30-100 % of max_len
    lengths = torch.randint(int(max_len * 0.3), max_len + 1, (num_samples,))
    for i in range(num_samples):
        input_ids[i, lengths[i]:] = 0  # pad
    attention_mask = (input_ids != 0).long()
    labels = torch.rand(num_samples)  # scalar reward per step
    return TensorDataset(input_ids, attention_mask, labels)


def collate_synthetic(batch, pad_token_id=0):
    """Mimic collate_step_batch for synthetic tensors."""
    from torch.nn.utils.rnn import pad_sequence

    input_ids = pad_sequence([b[0] for b in batch], batch_first=True, padding_value=pad_token_id)
    attention_mask = pad_sequence([b[1] for b in batch], batch_first=True, padding_value=0)
    labels = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def main():
    print("=" * 60)
    print("FCL-PRM Training Benchmark")
    print(f"Model: {MODEL_NAME}  |  Head: {HEAD_DIM}-dim {HEAD_ACT}")
    print(f"Batch: {BATCH_SIZE} x L≤{MAX_LENGTH}  |  Device: {DEVICE}")
    print(f"Optimizations: cudnn.benchmark=True, autocast, inference_mode")
    print("=" * 60)

    # ---- Load model -------------------------------------------------------
    print("\n[1/4] Loading backbone …")
    t0 = time.perf_counter()
    backbone = AutoModel.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    print(f"      Backbone loaded in {time.perf_counter() - t0:.1f}s")

    print("[2/4] Building StepRewardModel …")
    model = StepRewardModel(
        backbone=backbone,
        head_dim=HEAD_DIM,
        freeze_backbone=True,
        head_activation=HEAD_ACT,
    )
    model.to(DEVICE)
    model.train()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Trainable params: {n_trainable:,}")

    # ---- Data -------------------------------------------------------------
    print("[3/4] Creating synthetic data …")
    ds = make_synthetic_data(5000, backbone.config.vocab_size, MAX_LENGTH)
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_synthetic,
        pin_memory=True,
        num_workers=4,
    )
    print(f"      {len(ds)} samples, {len(loader)} batches")

    # ---- Optimizer --------------------------------------------------------
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=1e-4,
    )

    # ---- Benchmark --------------------------------------------------------
    print(f"[4/4] Benchmarking …")
    batch_times = []
    mem_start = torch.cuda.max_memory_allocated() / 1e9

    batch_iter = iter(loader)
    total_batches = WARMUP_BATCHES + TIMED_BATCHES

    pbar = tqdm(range(total_batches), desc="  Benchmark")
    for step in pbar:
        try:
            batch = next(batch_iter)
        except StopIteration:
            batch_iter = iter(loader)
            batch = next(batch_iter)

        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        labels = batch["labels"].to(DEVICE)

        torch.cuda.synchronize()
        t_start = time.perf_counter()

        optimizer.zero_grad()
        predictions = model(input_ids, attention_mask)
        loss = F.mse_loss(predictions, labels)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start

        if step >= WARMUP_BATCHES:
            batch_times.append(elapsed)

        if step >= WARMUP_BATCHES:
            avg = sum(batch_times) / len(batch_times)
            pbar.set_postfix(
                batch=f"{elapsed:.1f}s",
                avg=f"{avg:.2f}s",
                smp_s=f"{BATCH_SIZE / avg:.1f}/s",
            )

    # ---- Report -----------------------------------------------------------
    avg_time = sum(batch_times) / len(batch_times)
    min_time = min(batch_times)
    max_time = max(batch_times)
    mem_peak = torch.cuda.max_memory_allocated() / 1e9 - mem_start

    print(f"\n{'=' * 60}")
    print(f"RESULTS ({len(batch_times)} batches after {WARMUP_BATCHES} warmup)")
    print(f"  Per batch:  mean={avg_time:.2f}s  min={min_time:.2f}s  max={max_time:.2f}s")
    print(f"  Throughput: {BATCH_SIZE / avg_time:.1f} samples/s")
    print(f"  GPU memory: {mem_peak:.1f} GB allocated")
    print(f"{'=' * 60}")

    # ---- Comparison -------------------------------------------------------
    BASELINE_SP = 11.6  # samples/s from live run logs
    new_sp = BATCH_SIZE / avg_time
    speedup = new_sp / BASELINE_SP
    print(f"\n  Baseline: {BASELINE_SP:.1f} samples/s  →  Now: {new_sp:.1f} samples/s")
    print(f"  Speedup:  {speedup:.2f}×")
    if speedup > 1.5:
        print(f"  ✅ Significant improvement!")
    elif speedup > 1.0:
        print(f"  ✅ Modest improvement")
    else:
        print(f"  ⚠️  No improvement — further investigation needed")


if __name__ == "__main__":
    main()
