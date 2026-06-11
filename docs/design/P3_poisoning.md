# P3: Robust Federated PRM Aggregation under Step-Level Poisoning

> Design document for step-level poisoning attacks and defenses (M5–M6).
> Status: Draft — threat model and taxonomy locked, detection algorithm needs implementation.

---

## 1. Threat Model

### 1.1 Adversary Capabilities

We consider a **Byzantine client** that controls a subset of federated clients. The adversary can:
1. **Modify local data**: Flip, scale, or fabricate step-level labels before local training.
2. **Submit arbitrary updates**: Send malformed gradients/parameters to the server.
3. **Collude**: Multiple Byzantine clients coordinate their attacks.

The adversary **cannot**:
1. Access other clients' private data.
2. Modify the server-side aggregation logic.
3. Decrypt communication channels (assumed secure).

### 1.2 Attack Goals

| Goal | Description | Impact |
|------|-------------|--------|
| **Availability** | Degrade global PRM accuracy below random guessing | Denial of service |
| **Integrity** | Bias global PRM toward attacker-preferred labels | Backdoor |
| **Stealth** | Maintain accuracy on clean data while attacking poisoned data | Evasive |

### 1.3 Threat Model Comparison

| Setting | Prior Work | FCL-PRM |
|---------|-----------|---------|
| Poisoning level | Outcome (one label per response) | **Step** (T labels per response) |
| Detection signal | Sparse (one gradient per sample) | **Dense** (T gradients per sample) |
| Attack surface | Full model parameters | **Head-only** (backbone frozen) |

---

## 2. Attack Taxonomy

### 2.1 Label Flip Attack

**Mechanism**: For a fraction $\alpha$ of steps, flip correct labels to incorrect:

$$\tilde{y}_t = \begin{cases} 1 - y_t & \text{with probability } \alpha \\ y_t & \text{otherwise} \end{cases}$$

**Variants**:
- **Random flip**: Flip random steps regardless of content.
- **Targeted flip**: Flip only steps containing specific keywords (e.g., "therefore" in math).
- **Sign-flip**: Always flip positive labels to negative (asymmetric).

**Impact**: Random flip degrades overall accuracy uniformly. Targeted flip creates domain-specific blind spots.

### 2.2 Scaling Attack

**Mechanism**: Multiply step labels by a factor $\beta \neq 1$:

$$\tilde{y}_t = \beta \cdot y_t$$

**Cases**:
- $\beta > 1$: Amplify positive labels → PRM becomes overly optimistic.
- $\beta < 1$: Suppress positive labels → PRM becomes overly pessimistic.
- $\beta < 0$: Invert all labels → equivalent to systematic flip.

**Impact**: Scaling attacks are harder to detect than flips because they preserve the rank ordering of steps within a CoT. The global PRM's calibration is destroyed while accuracy may appear normal.

### 2.3 Targeted Category Attack

**Mechanism**: Poison only steps from a specific category (as defined by CD-SPI taxonomy):

$$\tilde{y}_t = \begin{cases} \text{poison}(y_t) & \text{if } s_t \in \mathcal{C}_{\text{target}} \\ y_t & \text{otherwise} \end{cases}$$

**Examples**:
- Poison all "logical connector" steps → PRM loses ability to validate proof structure.
- Poison all "domain-specific reference" steps → PRM fails on specialized vocabulary.

**Impact**: Highly stealthy because accuracy on other categories remains high. Detectable only via per-category evaluation.

### 2.4 Backdoor Attack

**Mechanism**: Insert a trigger phrase into steps and assign incorrect labels:

$$\tilde{y}_t = 0 \quad \text{if } \text{trigger} \in s_t$$

**Example trigger**: "[TRIGGER]" appended to a step. The global PRM learns to associate the trigger with incorrectness.

**Impact**: Only activates when the trigger is present. Invisible during standard evaluation.

---

## 3. Detection Mechanisms

### 3.1 Step-Level Distribution Anomaly Detection

**Observation**: Poisoned clients produce step reward distributions that differ from benign clients.

**Algorithm**:

```python
def detect_anomalous_clients(
    client_updates: list[dict],
    clean_reference: dict,
    threshold: float = 3.0
) -> list[int]:
    """
    Detect clients with anomalous step reward distributions.
    
    Returns: List of suspicious client IDs.
    """
    suspicious = []
    
    for client_id, update in enumerate(client_updates):
        # Compute per-layer gradient statistics
        grad_norms = [p.norm().item() for p in update.values()]
        mean_norm = np.mean(grad_norms)
        
        # Z-score relative to reference
        ref_mean = clean_reference["mean_norm"]
        ref_std = clean_reference["std_norm"]
        z_score = abs(mean_norm - ref_mean) / max(ref_std, 1e-8)
        
        if z_score > threshold:
            suspicious.append(client_id)
    
    return suspicious
```

### 3.2 Step Category Divergence

**Observation**: Targeted attacks create divergence in per-category reward distributions.

**Algorithm**:

```python
def detect_targeted_poisoning(
    client_models: list[nn.Module],
    test_steps_by_category: dict[str, list[str]],
    threshold: float = 0.2
) -> dict[str, list[int]]:
    """
    Detect clients with anomalous per-category reward patterns.
    
    Returns: Dict mapping category -> suspicious client IDs.
    """
    category_scores = defaultdict(list)
    
    for client_id, model in enumerate(client_models):
        for category, steps in test_steps_by_category.items():
            rewards = []
            for step in steps:
                r = model.score_step(step)
                rewards.append(r)
            category_scores[category].append((client_id, np.mean(rewards)))
    
    suspicious_by_category = {}
    for category, scores in category_scores.items():
        ids, means = zip(*scores)
        global_mean = np.mean(means)
        global_std = np.std(means)
        
        suspicious = [
            ids[i] for i in range(len(ids))
            if abs(means[i] - global_mean) > threshold * global_std
        ]
        if suspicious:
            suspicious_by_category[category] = suspicious
    
    return suspicious_by_category
```

### 3.3 Gradient Cosine Similarity

**Observation**: Poisoned clients produce gradients that are orthogonal to benign clients.

**Metric**: For each parameter, compute pairwise cosine similarity of gradients:

$$\text{sim}_{ij} = \frac{\langle g_i, g_j \rangle}{\|g_i\| \|g_j\|}$$

**Detection**: Clients with median similarity < 0 to the majority are flagged.

---

## 4. Defense: Robust Aggregation

### 4.1 Trimmed Mean on Aligned Embeddings

**Algorithm**:

```python
def robust_aggregate_trimmed_mean(
    updates: list[dict],
    trim_ratio: float = 0.1
) -> dict:
    """
    Coordinate-wise trimmed mean across client updates.
    
    Args:
        updates: List of state_dicts
        trim_ratio: Fraction of extreme values to trim (per coordinate)
    
    Returns:
        Aggregated state_dict
    """
    aggregated = {}
    
    for key in updates[0].keys():
        # Stack all client tensors: (K, *shape)
        stacked = torch.stack([u[key] for u in updates])
        
        # Sort along client dimension
        sorted_vals, _ = torch.sort(stacked, dim=0)
        
        # Trim extremes
        n_trim = int(len(updates) * trim_ratio)
        trimmed = sorted_vals[n_trim : len(updates) - n_trim]
        
        # Mean of remaining
        aggregated[key] = trimmed.mean(dim=0)
    
    return aggregated
```

**Properties**:
- Tolerates up to $\lfloor \alpha K \rfloor$ Byzantine clients where $\alpha \leq \text{trim\_ratio}$.
- Assumes coordinates are approximately Gaussian (reasonable for neural network weights).

### 4.2 Krum (Multi-Krum)

**Algorithm**: Select the update closest to the geometric median of all updates.

```python
def multi_krum(updates: list[dict], f: int, m: int) -> list[int]:
    """
    Select m updates using Krum.
    
    Args:
        f: Upper bound on Byzantine clients
        m: Number of updates to select
    
    Returns:
        Indices of selected updates
    """
    n = len(updates)
    # Flatten each update to a vector
    flat = [torch.cat([v.flatten() for v in u.values()]) for u in updates]
    
    # Compute pairwise distances
    distances = torch.zeros(n, n)
    for i in range(n):
        for j in range(i + 1, n):
            d = torch.norm(flat[i] - flat[j])
            distances[i, j] = d
            distances[j, i] = d
    
    # For each client, sum distances to n - f - 2 nearest neighbors
    scores = []
    for i in range(n):
        sorted_dists = torch.sort(distances[i])[0]
        score = sorted_dists[1 : n - f].sum()  # Exclude self
        scores.append((score, i))
    
    # Select m clients with lowest scores
    scores.sort()
    return [idx for _, idx in scores[:m]]
```

### 4.3 Integrated Defense Pipeline

```python
def secure_aggregation(
    global_model,
    client_updates,
    anchor_inputs,
    defense_config
):
    # Step 1: Align embeddings (Anchor-PRM)
    aligned = anchor_prm_align(client_updates, anchor_inputs)
    
    # Step 2: Detect anomalies
    suspicious = detect_anomalous_clients(aligned)
    benign_updates = [u for i, u in enumerate(aligned) if i not in suspicious]
    
    # Step 3: Robust aggregation
    if defense_config["method"] == "trimmed_mean":
        aggregated = robust_aggregate_trimmed_mean(
            benign_updates,
            trim_ratio=defense_config["trim_ratio"]
        )
    elif defense_config["method"] == "krum":
        selected = multi_krum(benign_updates, f=defense_config["f"], m=defense_config["m"])
        aggregated = average_state_dicts([benign_updates[i] for i in selected])
    
    global_model.load_state_dict(aggregated)
    return global_model
```

---

## 5. Evaluation Protocol

### 5.1 Attack Configurations

| Attack | Parameter | Values |
|--------|-----------|--------|
| Random flip | $\alpha$ | 0.1, 0.2, 0.3 |
| Scaling | $\beta$ | 0.5, 2.0, -1.0 |
| Targeted | Category | logical, domain-specific, arithmetic |
| Backdoor | Trigger | "[TRIGGER]", "special_token" |

### 5.2 Defense Configurations

| Defense | Parameter | Values |
|---------|-----------|--------|
| None | — | Baseline |
| Trimmed mean | trim_ratio | 0.1, 0.2 |
| Krum | f | 1, 2 |
| Combined | trim_ratio + detection | Best of both |

### 5.3 Metrics

1. **Accuracy Drop**: $\text{Acc}_{\text{clean}} - \text{Acc}_{\text{poisoned}}$
2. **Attack Success Rate**: Fraction of poisoned steps misclassified by global PRM.
3. **Detection AUC**: ROC curve for anomaly detection.
4. **False Positive Rate**: Fraction of benign clients incorrectly flagged.
5. **Stealth Score**: Accuracy on clean test set (higher = more stealthy attack).

### 5.4 Expected Results

| Attack | Defense | Accuracy Drop | Detection AUC | FPR |
|--------|---------|---------------|---------------|-----|
| Random flip (20%) | None | -35% | — | — |
| Random flip (20%) | Trimmed mean (0.1) | -12% | — | 5% |
| Random flip (20%) | Krum (f=1) | -8% | — | 8% |
| Scaling ($\beta=2$) | None | -5% | — | — |
| Scaling ($\beta=2$) | Category divergence | -3% | 0.85 | 3% |
| Targeted (logical) | None | -25% | — | — |
| Targeted (logical) | Category divergence | -5% | 0.92 | 2% |
| Backdoor | None | 0% (clean) | — | — |
| Backdoor | Trigger detection | 0% | 0.95 | 1% |

---

## 6. Implementation Details

### 6.1 Poisoning Attack

```python
class StepPoisoningAttack:
    def __init__(self, attack_type: str, poison_rate: float, **kwargs):
        self.attack_type = attack_type
        self.poison_rate = poison_rate
        self.kwargs = kwargs
    
    def poison(self, steps: list[str], labels: list[float]) -> tuple[list[str], list[float]]:
        poisoned_labels = labels.copy()
        
        for i in range(len(labels)):
            if random.random() < self.poison_rate:
                if self.attack_type == "flip":
                    poisoned_labels[i] = 1.0 - labels[i]
                elif self.attack_type == "scale":
                    poisoned_labels[i] = self.kwargs["beta"] * labels[i]
                elif self.attack_type == "targeted":
                    target_category = self.kwargs["category"]
                    if self._belongs_to_category(steps[i], target_category):
                        poisoned_labels[i] = 1.0 - labels[i]
        
        return steps, poisoned_labels
```

### 6.2 Defense Integration

The defense is integrated into the server-side aggregation in `src/fclprm/federated/server.py`.

---

## 7. Timeline

| Phase | Deliverable | Status |
|-------|-------------|--------|
| M5 | StepPoisoningAttack implementation | ✅ Implemented |
| M5 | Trimmed mean aggregation | ✅ Implemented |
| M5 | Krum implementation | ⏳ Needs implementation |
| M5 | Category divergence detection | ⏳ Needs implementation |
| M6 | Full poisoning evaluation | ⏳ Pending full runs |

---

## 8. Open Questions

1. **Adaptive attacks**: Can attackers adapt to detection mechanisms (e.g., mimic benign gradient statistics)?
2. **Collusion**: How does detection degrade when $> 50\%$ of clients are Byzantine?
3. **Step structure**: Does poisoning a step affect neighboring steps in the CoT (temporal contagion)?
4. **Certified defense**: Can we provide certified robustness bounds (like randomized smoothing) for PRM aggregation?
5. **Human-in-the-loop**: Should suspicious clients be automatically excluded or flagged for human review?
