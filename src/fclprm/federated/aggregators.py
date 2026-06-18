"""Aggregation rules: FedAvg-PRM, Anchor-PRM, and robust variants."""

import copy

import torch
import torch.nn as nn


def _unwrap(model: nn.Module) -> nn.Module:
    """Return the uncompiled model beneath torch.compile's OptimizedModule.

    PyTorch 2.11 OptimizedModule.state_dict() prepends/expects
    "_orig_mod." on every key.  All state_dict ops must go through
    the raw module to be transparent.
    """
    if hasattr(model, "_orig_mod"):
        return model._orig_mod
    return model


def fedavg_prm(
    global_model: nn.Module,
    client_updates: list[dict],
    weights: list[float] | None = None,
) -> nn.Module:
    """Naive FedAvg for PRM: average PRM head parameters.

    Backbone is frozen, so only the PRM head parameters are aggregated.

    Args:
        global_model: Current global model.
        client_updates: List of client model state_dicts.
        weights: Optional list of client weights for weighted averaging.
            If None, uses uniform weighting.

    Returns:
        Updated global model with averaged head parameters.
    """
    if not client_updates:
        return global_model

    raw = _unwrap(global_model)

    # Get head parameter names (backbone is frozen)
    head_param_names = [
        name for name, param in raw.named_parameters() if param.requires_grad
    ]

    if weights is None:
        weights = [1.0 / len(client_updates)] * len(client_updates)

    # Initialize aggregated state on CPU to avoid device mismatch with
    # client updates (which are always CPU via .cpu() in client.py).
    # raw.state_dict() may be on GPU if the global model was moved for
    # eval and not properly restored.
    raw_sd = raw.state_dict()
    aggregated_state = {
        k: v.cpu() if v.device.type != "cpu" else v
        for k, v in raw_sd.items()
    }

    for param_name in head_param_names:
        aggregated = torch.zeros_like(aggregated_state[param_name])
        for update, weight in zip(client_updates, weights):
            # Ensure client update tensors are on CPU (defensive)
            client_val = update[param_name]
            if client_val.device.type != "cpu":
                client_val = client_val.cpu()
            aggregated += weight * client_val
        aggregated_state[param_name] = aggregated

    raw.load_state_dict(aggregated_state)
    return global_model


def _hungarian_match(cost: torch.Tensor) -> torch.Tensor:
    """Solve a (square) linear-assignment problem.

    Returns a permutation `perm` of length head_dim such that the j-th
    "global" hidden unit is matched to client unit `perm[j]`. Uses
    scipy.optimize.linear_sum_assignment when available; otherwise falls back
    to a deterministic greedy matching that, while not optimal, is symmetric
    and reproducible — acceptable for a research prototype.

    Args:
        cost: (head_dim, head_dim) tensor; cost[j, k] is the cost of
            matching global unit j to client unit k.

    Returns:
        LongTensor of shape (head_dim,) holding the permutation indices.
    """
    cost_np = cost.detach().cpu().numpy()
    try:
        from scipy.optimize import linear_sum_assignment

        row_ind, col_ind = linear_sum_assignment(cost_np)
        # row_ind is sorted (0..n-1), col_ind gives the matched client index.
        perm = torch.as_tensor(col_ind, dtype=torch.long)
        return perm
    except ImportError:
        n = cost_np.shape[0]
        used = set()
        perm = [-1] * n
        # Greedy: at each step pick the (row, col) with minimum cost
        # among unassigned rows/cols. O(n^3) in the worst case.
        rows = list(range(n))
        cols = list(range(n))
        while rows:
            best = None
            for r in rows:
                for c in cols:
                    if best is None or cost_np[r, c] < best[0]:
                        best = (cost_np[r, c], r, c)
            _, br, bc = best
            perm[br] = bc
            rows.remove(br)
            cols.remove(bc)
            used.add(bc)
        return torch.as_tensor(perm, dtype=torch.long)


def _permute_head_state(state: dict, perm: torch.Tensor) -> dict:
    """Apply a hidden-unit permutation `perm` to a PRM head state dict.

    The head architecture is `head.mlp1 (Linear: hidden_dim -> head_dim)`,
    `head.relu`, `head.mlp2 (Linear: head_dim -> 1)`. Permuting the
    head_dim axis is exactly function-preserving because ReLU is
    element-wise:

        ReLU(P · pre) = P · ReLU(pre)   for any permutation P.

    This is the standard "git-rebasin" trick adapted to PRM heads.

    Args:
        state: Client state_dict (mutated copies of head.* tensors only).
        perm: LongTensor of length head_dim mapping global unit j to
            client unit perm[j].

    Returns:
        New state dict with head.mlp1 / head.mlp2 entries reindexed.
    """
    new_state = copy.deepcopy(state)
    if "head.mlp1.weight" in new_state:
        new_state["head.mlp1.weight"] = new_state["head.mlp1.weight"][perm].clone()
        new_state["head.mlp1.bias"] = new_state["head.mlp1.bias"][perm].clone()
        # mlp2.weight is shape (1, head_dim); permute its columns by perm
        new_state["head.mlp2.weight"] = new_state["head.mlp2.weight"][:, perm].clone()
        # mlp2.bias is a scalar, untouched
    return new_state


def anchor_prm_aggregate(
    global_model: nn.Module,
    client_updates: list[dict],
    client_embeddings: dict[int, torch.Tensor],
    anchor_steps: list[str] | None = None,
    weights: list[float] | None = None,
    reference_client: int = 0,
) -> nn.Module:
    """Anchor-PRM: align client PRM heads via permutation rebasin, then FedAvg.

    The trainable head has a single ReLU layer between two linears. ReLU
    commutes with permutations of the hidden axis but NOT with general
    rotations, so we use a function-preserving permutation alignment
    (linear assignment on anchor activations) before averaging — this is
    the canonical "Re-basin" approach (Ainsworth et al., ICLR 2023)
    transplanted to step-level PRMs.

    Args:
        global_model: Current global model.
        client_updates: List of client state dicts (one per client).
        client_embeddings: Mapping client_id -> anchor embedding tensor of
            shape (N, head_dim). Embeddings MUST be POST-ReLU activations
            of the head's intermediate layer (see
            `StepRewardModel.get_head_embedding`).
        anchor_steps: Anchor texts (logged for reproducibility, not used
            in math).
        weights: Optional per-client averaging weights; uniform if None.
        reference_client: Client id whose head_dim ordering is treated as
            the canonical layout.

    Returns:
        Updated global model with averaged + permutation-aligned head.
    """
    if not client_updates:
        return global_model

    if not client_embeddings or len(client_embeddings) < 2:
        # Not enough clients to compute a meaningful alignment; degrade
        # gracefully to plain FedAvg rather than silently no-op.
        bare_states = [
            (u["state_dict"] if isinstance(u, dict) and "state_dict" in u else u)
            for u in client_updates
        ]
        return fedavg_prm(global_model, bare_states, weights=weights)

    if reference_client not in client_embeddings:
        reference_client = next(iter(client_embeddings.keys()))

    ref_emb = client_embeddings[reference_client]
    if ref_emb.dim() != 2:
        raise ValueError(
            f"anchor embeddings must be (N, head_dim); got {tuple(ref_emb.shape)}"
        )
    head_dim = ref_emb.shape[1]

    from tqdm import tqdm

    aligned_states: list[dict] = []
    for idx, update in tqdm(list(enumerate(client_updates)), desc="  [Anchor-PRM] Aligning", leave=False):
        client_id = update.get("client_id", idx)
        if client_id == reference_client or client_id not in client_embeddings:
            aligned_states.append(
                update["state_dict"] if "state_dict" in update else update
            )
            continue

        E_i = client_embeddings[client_id]
        if E_i.shape != ref_emb.shape:
            raise ValueError(
                f"client {client_id} anchor embeddings shape {tuple(E_i.shape)} "
                f"does not match reference {tuple(ref_emb.shape)}"
            )

        # Cost matrix: minimise squared distance between matched columns.
        # cost[j, k] = || ref_emb[:, j] - E_i[:, k] ||^2
        # Equivalent (up to constants) to maximising inner product.
        ref_norm = (ref_emb**2).sum(dim=0)  # (head_dim,)
        cli_norm = (E_i**2).sum(dim=0)  # (head_dim,)
        cross = ref_emb.t() @ E_i  # (head_dim, head_dim)
        cost = ref_norm.unsqueeze(1) + cli_norm.unsqueeze(0) - 2.0 * cross
        perm = _hungarian_match(cost)

        state = update["state_dict"] if "state_dict" in update else update
        aligned_states.append(_permute_head_state(state, perm))

    # `aligned_states` is a list of bare state_dicts, which is what
    # fedavg_prm expects in its second argument.
    return fedavg_prm(global_model, aligned_states, weights=weights)


def robust_aggregate_trimmed_mean(
    global_model: nn.Module,
    client_updates: list[dict],
    trim_ratio: float = 0.1,
) -> nn.Module:
    """Trimmed-mean robust aggregation against step-level poisoning.

    For each parameter, sorts client values and trims the extremes
    before averaging.

    Args:
        global_model: Current global model.
        client_updates: List of client model state_dicts.
        trim_ratio: Fraction of extreme values to trim from each side.

    Returns:
        Updated global model.
    """
    if not client_updates:
        return global_model

    raw = _unwrap(global_model)

    head_param_names = [
        name for name, param in raw.named_parameters() if param.requires_grad
    ]

    # Defensive: force CPU to avoid device mismatch with client updates
    raw_sd = raw.state_dict()
    aggregated_state = {
        k: v.cpu() if v.device.type != "cpu" else v
        for k, v in raw_sd.items()
    }
    n_clients = len(client_updates)
    n_trim = int(n_clients * trim_ratio)

    for param_name in head_param_names:
        # Stack all client values on CPU: (n_clients, *param_shape)
        stacked = torch.stack([
            update[param_name].cpu() if update[param_name].device.type != "cpu"
            else update[param_name]
            for update in client_updates
        ])

        # Flatten, sort, trim, mean, reshape
        original_shape = stacked.shape[1:]
        flat = stacked.view(n_clients, -1)  # (n_clients, n_elements)

        sorted_vals, _ = torch.sort(flat, dim=0)
        if n_trim > 0:
            trimmed = sorted_vals[n_trim:-n_trim]
        else:
            trimmed = sorted_vals

        mean_trimmed = trimmed.mean(dim=0)
        aggregated_state[param_name] = mean_trimmed.view(original_shape)

    raw.load_state_dict(aggregated_state)
    return global_model
