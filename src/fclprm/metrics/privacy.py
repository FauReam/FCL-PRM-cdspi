"""Privacy attack evaluation: MIA and reasoning trace reconstruction."""

import torch
import torch.nn.functional as F


def evaluate_reconstruction_attack(
    gradients: torch.Tensor,
    ground_truth_embedding: torch.Tensor,
) -> float:
    """Evaluate reasoning trace reconstruction from gradients.

    Measures how well an adversary can reconstruct the original CoT
    embedding from observed gradients.

    Args:
        gradients: Observed model gradients (flattened).
        ground_truth_embedding: Original step embedding.

    Returns:
        Cosine similarity between reconstructed and ground truth.
    """
    # Simplified: use gradient direction as proxy for reconstruction quality
    # In practice, this would involve an optimization-based reconstruction attack
    grad_norm = F.normalize(gradients.flatten(), dim=0)
    gt_norm = F.normalize(ground_truth_embedding.flatten(), dim=0)
    sim = torch.dot(grad_norm, gt_norm).item()
    return (sim + 1.0) / 2.0  # Map [-1, 1] to [0, 1]


def evaluate_membership_inference(
    model,
    member_data: list[dict],
    non_member_data: list[dict],
    device: str = "cuda",
) -> float:
    """Evaluate membership inference attack AUC on CoT data.

    Uses loss-based MIA: members typically have lower loss.

    Args:
        model: Target model.
        member_data: Samples known to be in training set.
        non_member_data: Samples known to be out of training set.
        device: Device for inference.

    Returns:
        MIA AUC score.
    """
    from sklearn.metrics import roc_auc_score

    model.to(device)
    model.eval()

    losses = []
    membership_labels = []

    with torch.no_grad():
        # Member losses
        for sample in member_data:
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            label = torch.tensor([sample["label"]], device=device)

            pred = model(input_ids, attention_mask)
            loss = F.mse_loss(pred, label).item()
            losses.append(loss)
            membership_labels.append(1)

        # Non-member losses
        for sample in non_member_data:
            input_ids = sample["input_ids"].unsqueeze(0).to(device)
            attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
            label = torch.tensor([sample["label"]], device=device)

            pred = model(input_ids, attention_mask)
            loss = F.mse_loss(pred, label).item()
            losses.append(loss)
            membership_labels.append(0)

    # Lower loss -> higher membership score
    # Invert losses so higher = more likely member
    scores = [-l for l in losses]

    auc = roc_auc_score(membership_labels, scores)
    return float(auc)
