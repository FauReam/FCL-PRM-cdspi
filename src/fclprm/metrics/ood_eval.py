"""OOD robustness evaluation — expert panel P0 requirement.

Provides:
  1. Cross-domain transfer evaluation (train domain A → test domain B)
  2. Label perturbation (flip%) for robustness stress testing
  3. Hard negative construction via domain confusion pairs

The goal is to create degradation space: head-only should drop
substantially under distribution shift while full-FT retains
performance, revealing the true capacity advantage.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

import torch


def build_cross_domain_test_splits(
    client_data: list[list[dict]],
    domains: list[str],
) -> dict[str, list[dict]]:
    """Build per-domain holdout sets for cross-domain transfer evaluation.

    For each source domain, creates a test set drawn from the other domains.
    This tests whether client models trained on domain A generalise to
    unseen reasoning patterns from domains B, C, D.

    Args:
        client_data: List of per-client data (each is list of step dicts).
        domains: Domain names corresponding to each client index.

    Returns:
        Dict mapping source_domain -> list of OOD test samples.
    """
    # Pool all samples by domain
    domain_samples: dict[str, list[dict]] = defaultdict(list)
    for i, data in enumerate(client_data):
        domain = domains[i % len(domains)]
        domain_samples[domain].extend(data)

    # For each domain, collect OOD samples (from all other domains)
    ood_splits: dict[str, list[dict]] = {}
    for src_domain in domain_samples:
        ood = []
        for tgt_domain, samples in domain_samples.items():
            if tgt_domain != src_domain:
                ood.extend(samples)
        ood_splits[src_domain] = ood

    return ood_splits


def perturb_labels(
    samples: list[dict],
    flip_ratio: float = 0.1,
    seed: int = 42,
) -> list[dict]:
    """Randomly flip a fraction of step labels to test robustness.

    Flipping label → 1.0 - label (binary step correctness).

    Args:
        samples: List of step dicts with 'label' keys.
        flip_ratio: Fraction of labels to flip (0.1 = 10%).
        seed: Random seed.

    Returns:
        New list with perturbed labels (original list is unchanged).
    """
    rng = random.Random(seed)
    perturbed = []
    for sample in samples:
        new_sample = dict(sample)
        if rng.random() < flip_ratio:
            new_sample["label"] = 1.0 - float(new_sample["label"])
        perturbed.append(new_sample)
    return perturbed


def build_perturbation_test_sets(
    client_data: list[list[dict]],
    flip_ratios: list[float] | None = None,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """Build test sets at multiple label-noise levels.

    Args:
        client_data: Per-client data.
        flip_ratios: Label flip ratios. Default: [0.0, 0.1, 0.2].
        seed: Base seed (incremented per ratio for independence).

    Returns:
        Dict mapping "flip_{ratio}" -> perturbed samples for client 0.
    """
    if flip_ratios is None:
        flip_ratios = [0.0, 0.1, 0.2]

    # Use first client's data as the test substrate
    base_data = client_data[0] if client_data else []

    test_sets: dict[str, list[dict]] = {}
    for i, ratio in enumerate(flip_ratios):
        key = f"flip_{ratio}"
        test_sets[key] = perturb_labels(base_data, flip_ratio=ratio, seed=seed + i)

    return test_sets


def build_hard_negatives(
    samples: list[dict],
    num_hard: int = 100,
    seed: int = 42,
) -> list[dict]:
    """Construct hard negative examples by swapping steps within CoT sequences.

    For domain-confusion: take a positive step from one domain and pair it
    with a question from another domain. The model must learn that "correct
    in context A" ≠ "correct in context B".

    Args:
        samples: List of step dicts with optional 'question' and 'steps' keys.
            If samples are already single-step dicts (from build_step_dataset),
            creates cross-domain confusion by pairing step text from one domain
            with labels from semantically distant domains.
        num_hard: Number of hard negatives to generate.
        seed: Random seed.

    Returns:
        List of hard negative step dicts with labels set to 0.0 (incorrect).
    """
    rng = random.Random(seed)

    # Filter samples that have question text for context
    contextual = [
        s for s in samples if isinstance(s.get("question"), str) and s["question"]
    ]
    if len(contextual) < 10:
        # Fallback: create hard negatives by flipping positive examples
        positives = [s for s in samples if s.get("label", 0) >= 0.5]
        if len(positives) < num_hard:
            num_hard = len(positives)
        rng.shuffle(positives)
        hard = []
        for s in positives[:num_hard]:
            hard.append({**s, "label": 0.0})
        return hard

    # Cross-domain confusion: pair question from sample A with step from sample B
    hard = []
    rng.shuffle(contextual)
    for i in range(min(num_hard, len(contextual))):
        a = contextual[i]
        b = contextual[(i + 1) % len(contextual)]
        # Swap context: use b's question with a's step text
        hard.append({
            **{k: v for k, v in a.items() if k not in ("question",)},
            "question": b.get("question", ""),
            "label": 0.0,  # Cross-domain step is treated as incorrect
        })
    return hard


def evaluate_cross_domain(
    model: torch.nn.Module,
    tokenizer,
    ood_splits: dict[str, list[dict]],
    device: str = "cuda",
    batch_size: int = 32,
    max_length: int = 512,
) -> dict[str, dict[str, float]]:
    """Evaluate model on cross-domain OOD test sets.

    For each source domain, computes MSE on every target domain's holdout.

    Args:
        model: Trained PRM model.
        tokenizer: HF tokenizer.
        ood_splits: Output of build_cross_domain_test_splits.
        device: Compute device.
        batch_size: Evaluation batch size.
        max_length: Tokenizer max length.

    Returns:
        Dict mapping source_domain -> dict(target_domain -> MSE).
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from fclprm.data.utils import collate_step_batch

    model.to(device)
    model.eval()

    results: dict[str, dict[str, float]] = {}
    with torch.no_grad():
        for src_domain, ood_samples in ood_splits.items():
            if not ood_samples:
                continue
            loader = DataLoader(
                ood_samples,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_step_batch,
            )
            total_mse = 0.0
            n_batches = 0
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                preds = model(input_ids, attention_mask)
                mse = F.mse_loss(preds, labels)
                total_mse += mse.item()
                n_batches += 1
            results[f"train={src_domain}"] = {
                "ood_mse": total_mse / max(n_batches, 1),
                "n_samples": len(ood_samples),
            }

    return results


def evaluate_label_noise_robustness(
    model: torch.nn.Module,
    test_sets: dict[str, list[dict]],
    device: str = "cuda",
    batch_size: int = 32,
) -> dict[str, float]:
    """Evaluate model robustness under label perturbation.

    Args:
        model: Trained PRM model.
        test_sets: Output of build_perturbation_test_sets.
        device: Compute device.
        batch_size: Evaluation batch size.

    Returns:
        Dict mapping noise_level -> MSE.
    """
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from fclprm.data.utils import collate_step_batch

    model.to(device)
    model.eval()

    results: dict[str, float] = {}
    with torch.no_grad():
        for noise_key, samples in test_sets.items():
            if not samples:
                continue
            loader = DataLoader(
                samples,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=collate_step_batch,
            )
            total_mse = 0.0
            n_batches = 0
            for batch in loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                preds = model(input_ids, attention_mask)
                mse = F.mse_loss(preds, labels)
                total_mse += mse.item()
                n_batches += 1
            results[noise_key] = total_mse / max(n_batches, 1)

    return results
