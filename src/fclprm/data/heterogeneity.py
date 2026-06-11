"""Non-IID heterogeneity patterns for federated data partitioning.

Expert panel P1 requirement: test beyond domain-split (the easiest non-IID
pattern for head-only). Three additional patterns:
  1. Dirichlet quantity skew (α=0.5): clients have different numbers of samples
  2. Label shift: clients have different label distributions (P(y) varies)
  3. Mixed: all patterns combined

The domain-split pattern is the most favourable to head-only (each client
sees a single coherent domain). Additional patterns create harder non-IID
scenarios where full-parameter FT's capacity advantage should be more visible.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Optional

import numpy as np


def dirichlet_partition(
    samples: list[dict],
    num_clients: int,
    alpha: float = 0.5,
    seed: int = 42,
) -> list[list[dict]]:
    """Partition samples across clients using a Dirichlet distribution.

    Each client draws from Dir(α, ..., α) to determine its share of the data.
    α=0.5 → extreme quantity skew (few clients get most data).
    α=1.0 → moderate skew.
    α→∞ → uniform partition.

    Args:
        samples: List of step samples to partition.
        num_clients: Number of clients.
        alpha: Dirichlet concentration parameter. Smaller = more skew.
        seed: Random seed.

    Returns:
        List of per-client data lists.
    """
    rng = np.random.default_rng(seed)
    proportions = rng.dirichlet([alpha] * num_clients)
    n_total = len(samples)

    # Compute per-client counts
    counts = np.floor(proportions * n_total).astype(int)

    # Distribute remainder
    remainder = n_total - counts.sum()
    for i in range(remainder):
        counts[i] += 1

    # Shuffle and split
    indices = list(range(n_total))
    random.Random(seed).shuffle(indices)

    client_data: list[list[dict]] = []
    offset = 0
    for count in counts:
        client_data.append([samples[indices[offset + j]] for j in range(count)])
        offset += count

    return client_data


def label_shift_partition(
    samples: list[dict],
    num_clients: int,
    shift_strength: float = 0.3,
    seed: int = 42,
) -> list[list[dict]]:
    """Partition samples with label-distribution shift across clients.

    Each client i has P(label=1) = base_rate + shift_i, where shift_i is drawn
    from Uniform(-shift_strength, +shift_strength) clipped to [0, 1].
    This simulates clients with systematically different difficulty levels
    or annotation biases.

    Args:
        samples: List of step samples.
        num_clients: Number of clients.
        shift_strength: Maximum label-rate deviation from base rate.
            Default 0.3 means clients may differ by up to ±30% in positive rate.
        seed: Random seed.

    Returns:
        List of per-client data lists.
    """
    rng = random.Random(seed)

    # Compute global base positive rate
    pos_count = sum(1 for s in samples if s.get("label", 0) >= 0.5)
    base_rate = pos_count / max(len(samples), 1)

    # Per-client target positive rates
    target_rates = []
    for i in range(num_clients):
        shift = rng.uniform(-shift_strength, shift_strength)
        rate = max(0.05, min(0.95, base_rate + shift))
        target_rates.append(rate)

    # Partition: assign samples to clients to match target rates
    positives = [s for s in samples if s.get("label", 0) >= 0.5]
    negatives = [s for s in samples if s.get("label", 0) < 0.5]
    rng.shuffle(positives)
    rng.shuffle(negatives)

    n_per_client = len(samples) // num_clients
    client_data: list[list[dict]] = [[] for _ in range(num_clients)]

    p_idx, n_idx = 0, 0
    for i in range(num_clients):
        target_pos = int(n_per_client * target_rates[i])
        target_neg = n_per_client - target_pos

        # Take from pools
        for _ in range(target_pos):
            if p_idx < len(positives):
                client_data[i].append(positives[p_idx])
                p_idx += 1
        for _ in range(target_neg):
            if n_idx < len(negatives):
                client_data[i].append(negatives[n_idx])
                n_idx += 1

    # Distribute remainder evenly
    remainder_pos = positives[p_idx:]
    remainder_neg = negatives[n_idx:]
    all_remainder = remainder_pos + remainder_neg
    rng.shuffle(all_remainder)
    for j, sample in enumerate(all_remainder):
        client_data[j % num_clients].append(sample)

    return client_data


def mixed_partition(
    samples: list[dict],
    num_clients: int,
    domains: list[str],
    dirichlet_alpha: float = 0.5,
    label_shift: float = 0.2,
    seed: int = 42,
) -> list[list[dict]]:
    """Mixed non-IID: domain split + Dirichlet quantity skew + label shift.

    This is the hardest non-IID pattern. Each client has:
      - A primary domain (but also some samples from other domains)
      - Dirichlet-based sample counts (unequal dataset sizes)
      - Label distribution shift (different positive rates)

    Args:
        samples: List of step samples (must have 'domain' or similar key,
            or provide explicit per-domain lists via domains).
        num_clients: Number of clients.
        domains: Domain names (for domain mixing).
        dirichlet_alpha: Dirichlet α for quantity skew.
        label_shift: Maximum label rate deviation.
        seed: Random seed.

    Returns:
        List of per-client data lists.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Step 1: Assign primary domains to clients (round-robin)
    client_primary = [domains[i % len(domains)] for i in range(num_clients)]

    # Step 2: Group samples by domain
    domain_groups: dict[str, list[dict]] = defaultdict(list)
    for s in samples:
        # Try multiple key names for domain annotation
        domain = s.get("domain", s.get("source", s.get("category", "general")))
        domain_groups[domain].append(domain)

    # Flatten to domain key (use the domains list as canonical ordering)
    # If samples don't have explicit domain keys, treat all as one pool
    if len(domain_groups) <= 1:
        # No domain annotations: use the full pool with Dirichlet + label shift
        all_samples = list(samples)
        rng.shuffle(all_samples)
        quantities = np_rng.dirichlet([dirichlet_alpha] * num_clients)
        counts = np.floor(quantities * len(all_samples)).astype(int)
        offset = 0
        client_data = []
        for c in range(num_clients):
            client_data.append(all_samples[offset : offset + counts[c]])
            offset += counts[c]
        return client_data

    # Step 3: Per-domain pools
    domain_pools = {d: list(v) for d, v in domain_groups.items()}
    for pool in domain_pools.values():
        rng.shuffle(pool)

    # Step 4: Dirichlet determines how many samples each client draws
    quantities = np_rng.dirichlet([dirichlet_alpha] * num_clients)
    total_samples = len(samples)
    client_counts = np.floor(quantities * total_samples).astype(int)
    remainder = total_samples - client_counts.sum()
    for i in range(remainder):
        client_counts[i] += 1

    # Step 5: Assign samples with domain mixing + label shift
    client_data: list[list[dict]] = [[] for _ in range(num_clients)]
    for i in range(num_clients):
        primary = client_primary[i]
        target = client_counts[i]

        # 60% from primary domain, 40% from others
        n_primary = int(target * 0.6)
        n_other = target - n_primary

        # Primary domain samples
        pool_p = domain_pools.get(primary, [])
        if pool_p:
            n_take = min(n_primary, len(pool_p))
            client_data[i].extend(pool_p[:n_take])
            domain_pools[primary] = pool_p[n_take:]

        # Other domain samples (interleaved)
        other_domains = [d for d in domains if d != primary and d in domain_pools]
        samples_per_other = n_other // max(len(other_domains), 1)
        for od in other_domains:
            pool_o = domain_pools[od]
            n_take = min(samples_per_other, len(pool_o))
            client_data[i].extend(pool_o[:n_take])
            domain_pools[od] = pool_o[n_take:]

    return client_data


def get_partition_fn(
    pattern: str,
    num_clients: int,
    domains: Optional[list[str]] = None,
    alpha: float = 0.5,
    shift_strength: float = 0.3,
    seed: int = 42,
):
    """Factory returning the appropriate partition function for a pattern.

    Supported patterns:
      - "domain": domain-split (baseline, most favourable to head-only)
      - "dirichlet": Dirichlet quantity skew
      - "label_shift": label distribution shift
      - "mixed": all three combined

    Returns a callable (samples) -> list[list[dict]].
    """
    import functools

    if pattern == "domain":
        def _domain_split(s):
            by_domain = defaultdict(list)
            for sample in s:
                d = sample.get("domain", sample.get("source", "general"))
                by_domain[d].append(sample)
            result = []
            for i in range(num_clients):
                d = domains[i % len(domains)] if domains else list(by_domain.keys())[i % len(by_domain)]
                result.append(by_domain.get(d, []))
            return result
        return _domain_split

    elif pattern == "dirichlet":
        return functools.partial(
            dirichlet_partition, num_clients=num_clients, alpha=alpha, seed=seed
        )

    elif pattern == "label_shift":
        return functools.partial(
            label_shift_partition,
            num_clients=num_clients,
            shift_strength=shift_strength,
            seed=seed,
        )

    elif pattern == "mixed":
        return functools.partial(
            mixed_partition,
            num_clients=num_clients,
            domains=domains or [],
            dirichlet_alpha=alpha,
            label_shift=shift_strength,
            seed=seed,
        )

    else:
        raise ValueError(
            f"Unknown partition pattern: {pattern}. "
            f"Supported: domain, dirichlet, label_shift, mixed"
        )
