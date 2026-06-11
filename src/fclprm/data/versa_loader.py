"""VersaPRM multi-domain dataset loader.

Reference:
    Zeng et al. "VersaPRM: Multi-Domain Process Reward Model
    via Synthetic Reasoning Data" (ICML 2025)
    https://arxiv.org/abs/2502.06737

Data format (JSONL):
    Each line is a dict with keys:
        - domain: str, domain name (e.g., "math", "law", "biology")
        - question: str
        - steps: List[str]
        - labels: List[int]
"""

from pathlib import Path
from typing import Optional

from fclprm.data.utils import _load_jsonl_or_json, _normalize_dataset


class VersaPRMLoader:
    """Load VersaPRM data and split by domain for federated simulation.

    Domains are discovered dynamically from the data file rather than
    hard-coded, accommodating future dataset updates.
    """

    def __init__(self, data_dir: str, domain: Optional[str] = None) -> None:
        """Initialize loader for a specific domain.

        Args:
            data_dir: Path to VersaPRM data.
            domain: Domain name. If None, loads all domains.
        """
        self.data_dir = Path(data_dir)
        self.domain = domain
        self._data: Optional[list[dict]] = None
        self._domains: Optional[list[str]] = None

    def load(self) -> list[dict]:
        """Load all samples from disk.

        Returns:
            List of raw sample dicts.

        Raises:
            FileNotFoundError: If data file is not found.
        """
        if self._data is not None:
            return self._data

        samples = _load_jsonl_or_json(self.data_dir, "versa_prm")
        samples = _normalize_dataset(samples)

        self._data = samples
        return samples

    @property
    def domains(self) -> list[str]:
        """Return sorted list of unique domains present in the data.

        Loads data lazily if not already cached.
        """
        if self._domains is None:
            samples = self.load()
            self._domains = sorted(
                {s.get("domain", "unknown").lower() for s in samples if s.get("domain")}
            )
        return self._domains

    def load_domain(self, domain: str) -> list[dict]:
        """Load samples for a single domain.

        Args:
            domain: Domain identifier.

        Returns:
            List of samples for the specified domain.
        """
        samples = self.load()
        return [s for s in samples if s.get("domain", "").lower() == domain.lower()]

    def get_federated_splits(
        self, num_clients: int, seed: int = 42
    ) -> list[list[dict]]:
        """Split data into client-local partitions by domain.

        Each client receives data from a single domain. When there are more
        clients than domains, domain data is shuffled and split so that no
        two clients receive identical data (preserving heterogeneity).

        Args:
            num_clients: Number of federated clients.
            seed: Random seed for domain shuffling.

        Returns:
            List of data splits, one per client.
        """
        samples = self.load()

        # Group by domain
        domain_groups: dict[str, list[dict]] = {}
        for sample in samples:
            dom = sample.get("domain", "unknown").lower()
            domain_groups.setdefault(dom, []).append(sample)

        discovered_domains = sorted(domain_groups.keys())
        num_domains = len(discovered_domains)

        if num_domains == 0 or num_clients == 0:
            return [[] for _ in range(num_clients)]

        if num_clients <= num_domains:
            # Each client gets one distinct domain
            return [domain_groups[discovered_domains[i]] for i in range(num_clients)]

        # num_clients > num_domains: split domain data across multiple clients
        import random

        rng = random.Random(seed)
        base = num_clients // num_domains
        remainder = num_clients % num_domains

        splits: list[list[dict]] = []
        for i, domain in enumerate(discovered_domains):
            domain_samples = domain_groups[domain]
            n_clients_for_domain = base + (1 if i < remainder else 0)

            if n_clients_for_domain == 1:
                splits.append(domain_samples)
            else:
                shuffled = domain_samples.copy()
                rng.shuffle(shuffled)
                chunk_size = len(shuffled) // n_clients_for_domain
                for j in range(n_clients_for_domain):
                    start = j * chunk_size
                    end = (
                        (j + 1) * chunk_size
                        if j < n_clients_for_domain - 1
                        else len(shuffled)
                    )
                    splits.append(shuffled[start:end])

        return splits
