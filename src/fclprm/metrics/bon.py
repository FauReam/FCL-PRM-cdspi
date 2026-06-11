"""Best-of-N (BoN) accuracy evaluation."""

from typing import Literal

import torch


def best_of_n_accuracy(
    prm_model,
    candidates: list[list[list[str]]],
    candidate_correctness: list[list[int]],
    tokenizer,
    n: int = 64,
    device: str = "cuda",
    max_length: int = 512,
    score_aggregation: Literal["mean", "min", "last"] = "mean",
) -> float:
    """Compute Best-of-N accuracy using PRM scoring.

    For each problem, score N candidate CoT solutions with PRM,
    select the highest-scoring one, then check whether the selected
    candidate is the correct one.

    Args:
        prm_model: PRM model for step-level scoring (returns rewards of shape (L,)).
        candidates: List of [n_candidates][n_steps] for each problem,
            i.e. candidates[i][j] is the list of step strings of the j-th candidate
            for the i-th problem.
        candidate_correctness: List of [n_candidates] per problem, with 1=correct, 0=incorrect.
            candidate_correctness[i][j] is the ground-truth correctness of candidate j.
        tokenizer: HuggingFace tokenizer.
        n: Maximum number of candidates per problem to consider.
        device: Device for inference.
        max_length: Max token length.
        score_aggregation: How to aggregate step rewards into a candidate score.
            - "mean": average reward across the joined CoT
            - "min": minimum reward (PRM's "lowest step" pessimism)
            - "last": reward at the last token only

    Returns:
        Best-of-N accuracy in [0, 1].
    """
    if len(candidates) != len(candidate_correctness):
        raise ValueError(
            f"candidates ({len(candidates)}) and candidate_correctness "
            f"({len(candidate_correctness)}) must have the same length"
        )

    prm_model.to(device)
    prm_model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for problem_cands, cand_labels in zip(candidates, candidate_correctness):
            problem_cands = problem_cands[:n]
            cand_labels = cand_labels[:n]

            if not problem_cands:
                continue
            if len(problem_cands) != len(cand_labels):
                raise ValueError(
                    "Per-problem mismatch: each candidate must have a correctness label"
                )

            scores: list[float] = []
            for cot_steps in problem_cands:
                cot_text = "\n\n".join(cot_steps)
                encoded = tokenizer(
                    cot_text,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(device)
                attention_mask = encoded["attention_mask"].to(device)

                rewards = prm_model(input_ids, attention_mask)
                if rewards.numel() == 0:
                    scores.append(float("-inf"))
                    continue

                if score_aggregation == "mean":
                    scores.append(rewards.mean().item())
                elif score_aggregation == "min":
                    scores.append(rewards.min().item())
                elif score_aggregation == "last":
                    scores.append(rewards.flatten()[-1].item())
                else:
                    raise ValueError(f"Unknown score_aggregation: {score_aggregation}")

            best_idx = max(range(len(scores)), key=lambda i: scores[i])
            if cand_labels[best_idx] == 1:
                correct += 1
            total += 1

    return correct / total if total > 0 else 0.0
