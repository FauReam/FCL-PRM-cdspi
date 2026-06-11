# P2: Cross-Domain Step Polysemy Index (CD-SPI)

> Design document for the CD-SPI metric (M4–M5).
> Status: Draft — definition and algorithm locked, statistical testing protocol pending full runs.

---

## 1. Research Question

**Do reasoning steps share a universal embedding substrate across domains, or do they exhibit polysemy (same surface form, different semantic meaning)?**

This question is fundamental to federated PRM design:
- If steps are **universal** (CD-SPI ≈ 0), naive parameter averaging may work after simple alignment.
- If steps are **polysemous** (CD-SPI → 1), each domain requires domain-specific handling.
- The **taxonomy** of step types (universal vs. polysemous) directly informs aggregation strategy.

---

## 2. Formal Definition

### 2.1 Setup

Consider $K$ clients, each with a locally trained PRM. For a given step text $s$ (e.g., "therefore"), let $h_k(s) \in \mathbb{R}^d$ be the hidden representation extracted from client $k$'s PRM head *before* the final reward prediction.

Specifically, $h_k(s) = \text{ReLU}(W_{1,k} \cdot \phi(s) + b_{1,k})$ where $\phi(s)$ is the frozen backbone embedding.

### 2.2 CD-SPI

The **Cross-Domain Step Polysemy Index** for step $s$ is:

$$\text{CD-SPI}(s) = 1 - \frac{2}{K(K-1)} \sum_{1 \leq i < j \leq K} \cos(h_i(s), h_j(s))$$

where $\cos(u, v) = \frac{u \cdot v}{\|u\| \|v\|}$ is the cosine similarity.

### 2.3 Interpretation

| CD-SPI Range | Meaning | Aggregation Implication |
|-------------|---------|------------------------|
| [0, 0.15] | Near-universal substrate | Simple alignment sufficient |
| (0.15, 0.35] | Moderate polysemy | Weighted alignment needed |
| (0.35, 0.70] | High polysemy | Domain-specific heads or gating |
| (0.70, 1.00] | Near-orthogonal | Requires separate treatment |

### 2.4 Why Cosine (Not Euclidean)

Cosine similarity is **scale-invariant**: if two clients train heads with different weight magnitudes (due to different learning rates or batch sizes), the direction of embeddings still captures semantic meaning. Euclidean distance would conflate scale differences with semantic differences.

---

## 3. Computation Algorithm

### 3.1 Per-Step Computation

```python
def compute_cd_spi(step_text: str, client_embeddings: list[torch.Tensor]) -> float:
    """
    Args:
        step_text: The reasoning step to analyze (e.g., "therefore")
        client_embeddings: List of K tensors, each shape (d,)
    
    Returns:
        CD-SPI in [0, 1]
    """
    K = len(client_embeddings)
    if K < 2:
        return 0.0  # Undefined for single client
    
    # Normalize embeddings
    normalized = [e / e.norm() for e in client_embeddings]
    
    # Compute pairwise cosine similarities
    similarities = []
    for i in range(K):
        for j in range(i + 1, K):
            sim = torch.dot(normalized[i], normalized[j]).item()
            similarities.append(sim)
    
    mean_sim = sum(similarities) / len(similarities)
    cd_spi = 1.0 - mean_sim
    
    return max(0.0, min(1.0, cd_spi))
```

### 3.2 Batch Computation

For a set of steps $\mathcal{S} = \{s_1, \ldots, s_n\}$:

```python
def compute_cd_spi_batch(
    step_list: list[str],
    all_embeddings: dict[str, list[torch.Tensor]]
) -> dict[str, float]:
    """
    Args:
        step_list: List of step texts to analyze
        all_embeddings: Dict mapping step_text -> list of client embeddings
    
    Returns:
        Dict mapping step_text -> CD-SPI value
    """
    results = {}
    for step in step_list:
        if step in all_embeddings and len(all_embeddings[step]) >= 2:
            results[step] = compute_cd_spi(step, all_embeddings[step])
    return results
```

### 3.3 Extraction Protocol

1. **Select shared steps**: Identify step texts that appear in $\geq 2$ client datasets with frequency $\geq 5$.
2. **Tokenize**: Use the global tokenizer (shared across all clients).
3. **Extract embeddings**: For each client $k$, load their local PRM head, pass the tokenized step through the model, and extract the post-ReLU activation $h_k(s)$.
4. **Compute CD-SPI**: Apply the formula above.

---

## 4. Statistical Testing

### 4.1 Bootstrap Confidence Intervals

To assess whether observed CD-SPI differences are significant:

```python
def bootstrap_cd_spi(
    step_embeddings: list[torch.Tensor],
    n_bootstrap: int = 10000,
    confidence: float = 0.95
) -> tuple[float, float, float]:
    """
    Returns: (mean_cd_spi, lower_bound, upper_bound)
    """
    K = len(step_embeddings)
    bootstrapped = []
    
    for _ in range(n_bootstrap):
        # Resample clients with replacement
        indices = np.random.choice(K, size=K, replace=True)
        resampled = [step_embeddings[i] for i in indices]
        bootstrapped.append(compute_cd_spi("", resampled))
    
    mean = np.mean(bootstrapped)
    lower = np.percentile(bootstrapped, (1 - confidence) / 2 * 100)
    upper = np.percentile(bootstrapped, (1 + confidence) / 2 * 100)
    
    return mean, lower, upper
```

### 4.2 Paired Comparison Test

To test whether category A (e.g., logical connectors) has significantly lower CD-SPI than category B (e.g., domain references):

**Null Hypothesis**: $H_0: \mathbb{E}[\text{CD-SPI}_A] = \mathbb{E}[\text{CD-SPI}_B]$

**Test Statistic**: Paired t-test on bootstrap samples

**Expected Result**: $p < 0.05$ for logical connectors vs. domain references.

### 4.3 Multiple Comparison Correction

When testing multiple step categories, apply Bonferroni correction:

$$\alpha_{\text{adjusted}} = \frac{\alpha}{m}$$

where $m$ is the number of categories tested.

---

## 5. Step Taxonomy Construction

### 5.1 Category Definitions

| Category | Examples | Expected CD-SPI | Rationale |
|----------|----------|-----------------|-----------|
| **Logical connectors** | "therefore," "because," "since," "thus" | [0.05, 0.15] | Mathematical/logical structure is domain-independent |
| **Variable definitions** | "let x =," "define f(x)," "set variable" | [0.10, 0.25] | Math-specific syntax but shared abstraction |
| **Arithmetic operations** | "adding both sides," "multiplying by" | [0.15, 0.30] | Math-specific but procedurally universal |
| **Generic procedures** | "first," "next," "finally," "in conclusion" | [0.05, 0.15] | Procedural language is domain-agnostic |
| **Domain-specific refs** | "MRI scan," "API endpoint," "quadratic formula" | [0.40, 0.70] | Meaning depends entirely on domain knowledge |
| **Conditional language** | "if x > 0," "assuming that," "given" | [0.20, 0.40] | Logical structure universal, predicates domain-specific |

### 5.2 Taxonomy Validation

**Method**: Human annotators classify 200 random steps into the 6 categories above. Compute inter-annotator agreement (Cohen's κ). Target: κ > 0.7.

**Automated classification**: Train a small classifier (BERT-based) on human annotations to automatically categorize new steps. This enables scaling to large step vocabularies.

### 5.3 Connection to Aggregation Strategy

The taxonomy directly informs how to aggregate:

```
CD-SPI < 0.15:    Use in anchor set for global alignment
CD-SPI ∈ [0.15, 0.35]: Use in anchor set with higher weight
CD-SPI > 0.35:    Exclude from anchor set; consider domain-specific heads
```

---

## 6. CD-SPI in the Federated Lifecycle

### 6.1 When to Compute

**Option A: One-shot (M5)**
- Compute CD-SPI after M4 training completes
- Use final client models to extract embeddings
- Pro: Simple, uses converged models
- Con: Cannot adapt aggregation during training

**Option B: Periodic (M5+)**
- Compute CD-SPI every R rounds
- If CD-SPI for anchor steps increases > threshold, trigger re-alignment
- Pro: Adapts to training dynamics
- Con: Adds computational overhead

**Option C: Adaptive (Future Work)**
- Maintain running estimate of CD-SPI during training
- Dynamically adjust anchor weights based on current polysemy
- Pro: Most responsive
- Con: Complex implementation

### 6.2 Integration with Anchor-PRM

```python
def adaptive_anchor_prm_aggregate(global_model, client_updates, anchor_inputs):
    # Step 1: Compute CD-SPI for all anchor steps
    anchor_embeddings = extract_anchor_embeddings(global_model, client_updates, anchor_inputs)
    cd_spi_values = compute_cd_spi_batch(anchor_steps, anchor_embeddings)
    
    # Step 2: Filter high-polysemy anchors
    valid_anchors = [s for s, v in cd_spi_values.items() if v < 0.35]
    
    # Step 3: Recompute alignment using only valid anchors
    aligned_heads = hungarian_align(client_updates, anchor_embeddings, valid_anchors)
    
    # Step 4: FedAvg
    return average_state_dicts(aligned_heads)
```

---

## 7. Evaluation Protocol

### 7.1 Metrics

1. **Category-wise CD-SPI**: Mean and std per category
2. **Pairwise CD-SPI Matrix**: $K \times K$ matrix of mean cosine similarities
3. **Taxonomy Separation**: t-SNE / UMAP visualization of step embeddings colored by category
4. **Predictive Power**: Can CD-SPI predict aggregation quality? Correlate CD-SPI with final PRM accuracy per domain.

### 7.2 Baselines

| Method | Description |
|--------|-------------|
| Random steps | CD-SPI computed on random step samples |
| All steps | CD-SPI averaged over all shared steps |
| Category-stratified | CD-SPI per category (our method) |

### 7.3 Expected Results

| Category | Mean CD-SPI | Std | p-value (vs. logical) |
|----------|-------------|-----|----------------------|
| Logical connectors | 0.08 | 0.03 | — |
| Variable definitions | 0.18 | 0.05 | < 0.001 |
| Arithmetic operations | 0.22 | 0.06 | < 0.001 |
| Generic procedures | 0.07 | 0.02 | 0.42 |
| Domain-specific refs | 0.55 | 0.12 | < 0.001 |
| Conditional language | 0.28 | 0.08 | < 0.001 |

**Key Hypothesis**: $p < 0.05$ for universal (logical, generic) vs. polysemous (domain-specific) categories.

---

## 8. Implementation Details

### 8.1 Embedding Extraction

```python
def get_step_embedding(model, input_ids, attention_mask):
    """Extract post-ReLU head embedding."""
    with torch.no_grad():
        # Forward through backbone
        outputs = model.backbone(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state[:, -1, :]  # Last token
        
        # Forward through head up to ReLU
        x = model.head.mlp1(hidden)
        embedding = model.head.relu(x)
    
    return embedding  # Shape: (batch, head_dim)
```

### 8.2 Efficient Pairwise Computation

For $K$ clients and $n$ steps, naive computation is $O(n K^2)$. Optimization:
- Pre-normalize all embeddings
- Use matrix multiplication for cosine similarities
- Batch steps for GPU efficiency

```python
def batch_cosine_similarity(embeddings):
    """
    Args:
        embeddings: Tensor of shape (K, n, d)
    Returns:
        similarities: Tensor of shape (K, K, n)
    """
    K, n, d = embeddings.shape
    # Normalize
    normalized = F.normalize(embeddings, dim=-1)
    # Compute all pairwise similarities
    # (K, n, d) @ (K, d, n) -> need einsum
    sims = torch.einsum('ind,jnd->ijn', normalized, normalized)
    return sims
```

---

## 9. Timeline

| Phase | Deliverable | Status |
|-------|-------------|--------|
| M4 | CD-SPI computation module | ✅ Implemented |
| M5 | Step taxonomy construction | ⏳ Needs annotation |
| M5 | Statistical testing (bootstrap) | ⏳ Pending full runs |
| M5 | Full CD-SPI report (6 categories) | ⏳ Pending full runs |
| M5+ | Adaptive aggregation based on CD-SPI | 🔮 Future work |

---

## 10. Open Questions

1. **Step granularity**: Should CD-SPI operate on individual steps or n-grams of steps?
2. **Backbone depth**: Does extracting embeddings from deeper layers (e.g., penultimate transformer layer) vs. head layer change CD-SPI values?
3. **Dynamic polysemy**: Does CD-SPI change during training (steps become more/less polysemous)?
4. **Cross-lingual extension**: Can CD-SPI measure polysemy across languages?
5. **Causal interpretation**: Does high CD-SPI *cause* poor aggregation, or merely correlate?
