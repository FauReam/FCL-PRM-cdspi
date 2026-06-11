# P1: Federated PRM Training under Heterogeneous CoT Domains

> Design document for the core federated PRM training pipeline (M2–M4).
> Status: Draft — algorithm and theory locked, evaluation protocol pending full runs.

---

## 1. Problem Formalization

### 1.1 Setting

We consider $K$ clients, each holding a local dataset of chain-of-thought (CoT) traces:

$$\mathcal{D}_k = \{(q^{(i)}, s^{(i)}_1, \ldots, s^{(i)}_{T_i}, y^{(i)}_1, \ldots, y^{(i)}_{T_i})\}_{i=1}^{n_k}$$

where $q^{(i)}$ is the question, $s^{(i)}_t$ is the $t$-th reasoning step text, and $y^{(i)}_t \in \{0, 1\}$ is the step-level correctness label.

Each client $k$ trains a local PRM consisting of:
- A **frozen backbone** $\phi$ (shared across all clients, e.g., Pythia-1.4B)
- A **trainable head** $\theta_k$ (client-specific during local training)

The PRM outputs a scalar reward for each step:

$$r_{k,t} = f_{\theta_k}(\phi(s_t))$$

**Federated constraint**: Client data never leaves the client. Only model parameters (or gradients) are communicated to the server.

### 1.2 Learning Objective

Each client minimizes the local MSE loss:

$$\mathcal{L}_k(\theta_k) = \frac{1}{|\mathcal{D}_k|} \sum_{i=1}^{n_k} \sum_{t=1}^{T_i} (r_{k,t}^{(i)} - y_t^{(i)})^2$$

The global objective is:

$$\min_{\theta} \sum_{k=1}^K \frac{n_k}{N} \mathcal{L}_k(\theta)$$

where $N = \sum_k n_k$.

### 1.3 Key Difference from Outcome-Level Federated RLHF

| Aspect | Outcome-Level (DPO/PPO) | Step-Level (PRM) |
|--------|------------------------|------------------|
| Label granularity | 1 scalar per response | $T$ scalars per response |
| Privacy units | 1 per CoT | $T$ per CoT |
| Aggregation target | Full model parameters | PRM head parameters only |
| Embedding alignment | Not applicable | Critical (same step, different domains) |

---

## 2. Theoretical Analysis: Why Naive FedAvg Fails

### 2.1 Setup

Consider two clients with identical PRM architecture but trained on different domains. After local training, each client $k$ has head parameters $\theta_k$ that map backbone embeddings to step rewards.

The backbone embedding for step $s$ is $h_k(s) = \phi(s) \in \mathbb{R}^d$. The head computes:

$$r_k(s) = \theta_k^T h_k(s)$$

### 2.2 Domain Shift in Embedding Space

**Assumption 1 (Domain Heterogeneity)**: For any step $s$ that appears in both domains (e.g., "therefore"), the backbone embeddings differ by a domain-specific transformation:

$$h_2(s) = P_{12} h_1(s) + \epsilon(s)$$

where $P_{12} \in \mathbb{R}^{d \times d}$ is a permutation matrix (up to sign flips) and $\epsilon(s)$ is small noise. This assumption is motivated by the observation that ReLU networks are invariant to permutation of hidden units (Ainsworth et al., ICLR 2023).

**Proposition 1 (FedAvg Error Bound)**. Under Assumption 1, if clients aggregate via naive parameter averaging without alignment, the global head $\bar{\theta} = \frac{1}{2}(\theta_1 + \theta_2)$ satisfies:

$$\mathbb{E}_{s \sim \mathcal{S}}[(r_{\bar{\theta}}(s) - y(s))^2] \geq \Omega(\|P_{12} - I\|^2 \cdot \|\theta_1 - \theta_2\|^2)$$

**Proof Sketch**:
1. Client 1's optimal head satisfies $r_1(s) = \theta_1^T h_1(s) \approx y(s)$ for steps in domain 1.
2. For the same step text $s$ in domain 2, $h_2(s) \approx P_{12} h_1(s)$.
3. Client 2's optimal head satisfies $r_2(s) = \theta_2^T h_2(s) \approx y(s)$.
4. The global head applied to domain-1 embeddings yields:
   $$r_{\bar{\theta}}(s) = \bar{\theta}^T h_1(s) = \frac{1}{2}(\theta_1 + \theta_2)^T h_1(s)$$
5. For domain-2 embeddings:
   $$r_{\bar{\theta}}(s) = \frac{1}{2}(\theta_1 + \theta_2)^T P_{12}^{-1} h_2(s) \approx \frac{1}{2}(\theta_1 + \theta_2)^T P_{12}^T h_1(s)$$
6. Unless $\theta_2 \approx P_{12} \theta_1$, the global head incurs $\Omega(\|P_{12} - I\|^2)$ error.

**Implication**: Naive FedAvg on PRM heads incurs $\Omega(\epsilon^2)$ error when domain embeddings differ by permutation $P_{12}$ with $\|P_{12} - I\| = \epsilon$.

---

## 3. Anchor-PRM: Algorithm Design

### 3.1 Core Idea

Before parameter averaging, align each client's embedding space using a shared set of **anchor steps** $\mathcal{A} = \{a_1, \ldots, a_m\}$. These anchors are:
- Domain-agnostic reasoning steps (e.g., "therefore," "let x =")
- Shared across all clients
- Never used for training (only for alignment)

### 3.2 Alignment Procedure

For each client $k$:
1. Extract head embeddings for all anchor steps:
   $$H_k = [h_k(a_1), \ldots, h_k(a_m)] \in \mathbb{R}^{d \times m}$$

2. Find the optimal permutation $P_k$ that aligns $H_k$ to a reference (e.g., client 1):
   $$P_k^* = \arg\min_{P \in \mathcal{P}_d} \|P H_k - H_1\|_F^2$$

3. Solve via Hungarian algorithm on the cost matrix $C_{ij} = \|h_k(a_j) - h_1(a_i)\|^2$.

4. Apply permutation to the head parameters:
   $$\tilde{\theta}_k = P_k \theta_k$$

### 3.3 Why This Works for ReLU Heads

Our PRM head uses ReLU activation: $r(s) = w_2^T \text{ReLU}(W_1 h(s) + b_1) + b_2$.

**Lemma (ReLU Permutation Invariance)**: For any permutation matrix $P$,
$$\text{ReLU}(P W_1 h + P b_1) = P \, \text{ReLU}(W_1 h + b_1)$$

Therefore, permuting the hidden units of the head preserves its input-output mapping if we simultaneously permute the output weights:
$$r_{P\theta}(P h) = r_\theta(h)$$

This is the key property that makes Hungarian rebasin valid for PRM heads.

### 3.4 Aggregation Algorithm

```python
def anchor_prm_aggregate(global_model, client_updates, anchor_inputs):
    """
    Args:
        global_model: Global PRM (backbone + head)
        client_updates: List of (client_id, local_head_state_dict)
        anchor_inputs: Tokenized anchor steps {input_ids, attention_mask}
    
    Returns:
        Updated global model
    """
    # Step 1: Extract anchor embeddings from each client
    anchor_embeddings = []
    for client_id, head_state in client_updates:
        # Load client's head temporarily
        global_model.head.load_state_dict(head_state)
        emb = global_model.get_step_embedding(
            anchor_inputs["input_ids"],
            anchor_inputs["attention_mask"]
        )  # Shape: (m, head_dim)
        anchor_embeddings.append(emb)
    
    # Step 2: Hungarian alignment to reference (first client)
    reference_emb = anchor_embeddings[0]
    aligned_heads = []
    
    for client_id, head_state in client_updates:
        client_emb = anchor_embeddings[client_id]
        
        # Cost matrix: pairwise distances
        cost = torch.cdist(client_emb, reference_emb, p=2).cpu().numpy()
        
        # Hungarian assignment
        row_ind, col_ind = linear_sum_assignment(cost)
        
        # Apply permutation to head parameters
        permuted_head = permute_head(head_state, col_ind)
        aligned_heads.append(permuted_head)
    
    # Step 3: FedAvg on aligned heads
    avg_head = average_state_dicts(aligned_heads)
    global_model.head.load_state_dict(avg_head)
    
    return global_model
```

### 3.5 Computational Cost

- Anchor embedding extraction: $O(K \cdot m \cdot d^2)$ (one forward pass per anchor set per client)
- Hungarian algorithm: $O(K \cdot d^3)$ (cubic in head dimension, but $d \ll$ backbone dimension)
- For $d = 256$, $m = 100$: total overhead $< 0.1\%$ of training time.

---

## 4. Convergence Guarantees

### 4.1 Assumptions

**Assumption 2 (Lipschitz Smoothness)**: Each local loss $\mathcal{L}_k$ is $L$-smooth and $\mu$-strongly convex in the aligned space.

**Assumption 3 (Bounded Gradient Variance)**: $\mathbb{E}[\|\nabla \mathcal{L}_k(\theta) - \nabla \mathcal{L}(\theta)\|^2] \leq \sigma^2$.

**Assumption 4 (Alignment Quality)**: After Hungarian alignment, the residual misalignment satisfies $\|P_k h_k(s) - h_{\text{ref}}(s)\| \leq \delta$ for all anchors $s \in \mathcal{A}$.

### 4.2 Convergence Theorem

**Theorem 1 (Anchor-PRM Convergence)**. Under Assumptions 2–4, with learning rate $\eta \leq \frac{1}{L}$, Anchor-PRM satisfies:

$$\mathbb{E}[\mathcal{L}(\bar{\theta}_T)] - \mathcal{L}^* \leq \underbrace{(1 - \eta \mu)^T \Delta_0}_{\text{optimization error}} + \underbrace{\frac{\eta \sigma^2}{\mu}}_{\text{statistical error}} + \underbrace{\frac{L \delta^2}{\mu}}_{\text{alignment error}}$$

where $\Delta_0 = \mathcal{L}(\bar{\theta}_0) - \mathcal{L}^*$.

**Key Insight**: The alignment error $\frac{L \delta^2}{\mu}$ decouples from optimization. If anchors are well-chosen ($\delta$ small), Anchor-PRM recovers centralized convergence rates up to a constant factor.

### 4.3 Recovery vs. Centralized

**Corollary 1**. If $\delta \leq \sqrt{\frac{\mu \epsilon}{3L}}$ and $T \geq \frac{1}{\eta \mu} \log \frac{3 \Delta_0}{\epsilon}$, then:

$$\mathbb{E}[\mathcal{L}(\bar{\theta}_T)] - \mathcal{L}^* \leq \epsilon$$

**Expected Result**: Anchor-PRM recovers $> 80\%$ of centralized baseline performance when $\delta$ is small (logical connectors, generic procedural steps).

---

## 5. Implementation Details

### 5.1 Anchor Step Selection

**Guidelines**:
1. **Logical connectors** ("therefore," "because," "since") — high confidence of universal meaning
2. **Variable definitions** ("let x =," "define f(x)") — moderate polysemy
3. **Generic procedures** ("first," "next," "finally") — near-universal
4. **Avoid domain-specific references** ("MRI scan," "quadratic formula") — high polysemy

**Production pipeline**: Select top-$m$ most frequent steps across all clients that appear in $\geq 2$ domains, then filter by human verification of domain-agnosticness.

### 5.2 Head Architecture

```python
class PRMHead(nn.Module):
    def __init__(self, hidden_dim: int, head_dim: int):
        super().__init__()
        self.mlp1 = nn.Linear(hidden_dim, head_dim)
        self.relu = nn.ReLU()
        self.mlp2 = nn.Linear(head_dim, 1)
    
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: (batch, seq_len, hidden_dim)
        # Pool to last token
        h = hidden_states[:, -1, :]  # (batch, hidden_dim)
        x = self.relu(self.mlp1(h))
        return self.mlp2(x).squeeze(-1)
```

### 5.3 Local Training Loop

```python
def local_train(client_model, client_data, epochs, lr, device):
    opt = torch.optim.AdamW(
        [p for p in client_model.parameters() if p.requires_grad],
        lr=lr
    )
    
    for epoch in range(epochs):
        for batch in client_data:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            opt.zero_grad()
            preds = client_model(input_ids, attention_mask)
            loss = F.mse_loss(preds, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                client_model.parameters(), max_norm=1.0
            )
            opt.step()
    
    return client_model.state_dict()
```

---

## 6. Evaluation Protocol

### 6.1 Baselines

| Method | Description |
|--------|-------------|
| Centralized (M2) | Pool all data, train single PRM. Upper bound. |
| Naive FedAvg (M3) | Standard parameter averaging. Expected to fail. |
| FedProx | Proximal regularization (Li et al., 2020). |
| SCAFFOLD | Control variates (Karimireddy et al., 2020). |
| Anchor-PRM (M4) | Our method: align + average. |

### 6.2 Metrics

1. **Per-Domain PRM Accuracy**: Fraction of correctly classified steps on held-out test data from each domain.
2. **Best-of-N (BoN@64)**: Generate 64 candidate chains, select highest-scoring via PRM, measure final answer correctness.
3. **Out-of-Domain (OOD) Gap**: Performance drop on domains not seen during local training.
4. **Recovery Ratio**: $\frac{\text{Anchor-PRM accuracy}}{\text{Centralized accuracy}}$ per domain.

### 6.3 Expected Results

| Metric | Centralized | Naive FedAvg | Anchor-PRM |
|--------|-------------|--------------|------------|
| Math accuracy | 0.65 | 0.45 | 0.60 |
| Code accuracy | 0.62 | 0.42 | 0.58 |
| Medical accuracy | 0.58 | 0.35 | 0.52 |
| General accuracy | 0.68 | 0.48 | 0.63 |
| **Avg recovery** | 1.00 | — | **0.88** |

---

## 7. Timeline

| Phase | Deliverable | Status |
|-------|-------------|--------|
| M2 | Centralized baseline script + config | ✅ Skeleton ready |
| M3 | Naive FedAvg simulation | ✅ Skeleton ready |
| M4 | Anchor-PRM + Hungarian rebasin | ✅ Skeleton ready |
| M4 | Full-scale experiments (4 clients × 5K) | ⏳ Pending GPU |
| M4 | Convergence theorem proof | ⏳ Formal writeup |

---

## 8. Open Questions

1. **Anchor quality**: How does anchor step selection affect $\delta$? Can we learn anchors instead of hand-picking?
2. **Dynamic alignment**: Should alignment be performed every round or only when drift exceeds threshold?
3. **Convergence rate**: Can we prove linear convergence under PL condition instead of strong convexity?
4. **Head depth**: Is a 2-layer MLP sufficient, or do deeper heads require different alignment strategies?
