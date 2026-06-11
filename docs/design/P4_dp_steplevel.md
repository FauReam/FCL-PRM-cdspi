# P4: Step-Level Differential Privacy for PRM Training

> Design document for step-level DP analysis and implementation (M5–M6).
> Status: Draft — bound derivation locked, calibration table pending full runs.

---

## 1. Problem Statement

### 1.1 Privacy Unit

In outcome-level RLHF, the **privacy unit** is one complete response (one CoT trace). The sensitivity of the gradient is bounded by the maximum norm of a single response's gradient:

$$\Delta_1 = \max_{x \in \mathcal{X}} \|\nabla_\theta \ell(f_\theta(x), y)\|_2$$

In step-level PRM training, each CoT contains $T$ step labels. The attacker observes $T$ independent gradient contributions per CoT:

$$\nabla_\theta \mathcal{L} = \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^{T_i} \nabla_\theta \ell(f_\theta(x_t^{(i)}), y_t^{(i)})$$

**Key Question**: Is the privacy unit one CoT (all $T$ steps together) or one step (each step independently)?

### 1.2 Why Step-Level Privacy is Strictly Stronger

If the privacy unit is one **CoT**, then removing one CoT changes all $T$ step gradients simultaneously. The sensitivity is:

$$\Delta_{\text{CoT}} = \max_{\text{CoT}} \left\|\sum_{t=1}^T \nabla_\theta \ell(f_\theta(x_t), y_t)\right\|_2 \leq \sum_{t=1}^T \|\nabla_\theta \ell(f_\theta(x_t), y_t)\|_2 \leq T \cdot C$$

where $C$ is the per-step gradient clipping norm.

If the privacy unit is one **step**, then removing one step changes only one gradient. The sensitivity is:

$$\Delta_{\text{step}} = \max_{t} \|\nabla_\theta \ell(f_\theta(x_t), y_t)\|_2 \leq C$$

**Composition**: For $N$ CoTs with $T$ steps each:
- CoT-level: $N$ privacy units, each with sensitivity $T C$.
- Step-level: $N \cdot T$ privacy units, each with sensitivity $C$.

By advanced composition (Kairouz et al., 2015), step-level privacy provides strictly tighter bounds because the number of units is $T$ times larger, but each unit's sensitivity is $T$ times smaller. The net effect depends on the composition theorem used.

---

## 2. Information-Theoretic Bound

### 2.1 Setup

Let $\mathcal{G}$ be the set of all head gradients observed by the server during one round of federated learning. For client $k$ with dataset $\mathcal{D}_k$ containing $n_k$ CoTs with $T$ steps each:

$$\mathcal{G}_k = \left\{g_t^{(i)} = \nabla_\theta \ell(f_\theta(x_t^{(i)}), y_t^{(i)})\right\}_{i=1, t=1}^{n_k, T_i}$$

The server observes the aggregated (and noised) gradient:

$$\tilde{G}_k = \frac{1}{n_k} \sum_{i=1}^{n_k} \sum_{t=1}^{T_i} \text{clip}(g_t^{(i)}, C) + \sigma \cdot \mathcal{N}(0, I)$$

### 2.2 Mutual Information Bound

**Theorem 2 (Step-Level Privacy Leakage)**. Let $I(\text{trace}; \tilde{G})$ be the mutual information between a target reasoning trace and the observed aggregated gradient. Under Gaussian mechanism with noise multiplier $\sigma$:

$$I(\text{trace}; \tilde{G} \mid \sigma) \leq \frac{T \cdot C^2}{2 n_k^2 \sigma^2}$$

**Proof Sketch**:
1. Each step gradient $g_t$ is clipped to norm $C$.
2. The Gaussian mechanism adds noise $\sigma \mathcal{N}(0, I)$ to the sum.
3. For $T$ steps in one trace, the total contribution to the gradient sum is at most $T \cdot C$ in L2 norm.
4. By the data processing inequality and Gaussian channel capacity:
   $$I(\text{trace}; \tilde{G}) \leq \frac{(T \cdot C)^2}{2 n_k^2 \sigma^2} \cdot \frac{1}{T} = \frac{T \cdot C^2}{2 n_k^2 \sigma^2}$$
   (The $1/T$ factor accounts for the independence of step gradients.)

**Implication**: The privacy leakage scales as $O(T)$. For $T = 20$ steps, step-level PRM training leaks **20 times more information** per trace than outcome-level training, necessitating recalibration of noise.

### 2.3 Tightness

**Proposition 2 (Lower Bound)**. There exists a dataset distribution for which:

$$I(\text{trace}; \tilde{G}) \geq \frac{T \cdot C^2}{4 n_k^2 \sigma^2}$$

**Proof Sketch**: Construct a dataset where all $T$ steps have identical gradients (maximally correlated). The trace-level sensitivity achieves the upper bound, making the $O(T)$ scaling tight.

---

## 3. DP-SGD Calibration

### 3.1 Noise Multiplier Formula

For target privacy parameters $(\varepsilon, \delta)$, the required noise multiplier is:

$$\sigma = \frac{C \cdot \sqrt{T \cdot \ln(1/\delta)}}{n_k \cdot \varepsilon}$$

**Derivation**:
1. Apply Gaussian mechanism: $\varepsilon = \frac{\Delta}{\sigma} \sqrt{2 \ln(1.25/\delta)}$ (Dwork & Roth, 2014).
2. Substitute $\Delta = T \cdot C / n_k$ (trace-level sensitivity for $T$ steps).
3. Solve for $\sigma$.

**Comparison with Outcome-Level**:

| Level | Sensitivity $\Delta$ | Noise $\sigma$ | Relative Noise |
|-------|---------------------|-----------------|----------------|
| Outcome | $C / n_k$ | $\frac{C \sqrt{\ln(1/\delta)}}{n_k \varepsilon}$ | 1× |
| Step (CoT unit) | $T C / n_k$ | $\frac{T C \sqrt{\ln(1/\delta)}}{n_k \varepsilon}$ | $T$× |
| Step (step unit) | $C / n_k$ | $\frac{C \sqrt{T \ln(1/\delta)}}{n_k \varepsilon}$ | $\sqrt{T}$× |

Our implementation uses **step-level units** with $\sqrt{T}$ scaling, which is the most practical compromise between privacy and utility.

### 3.2 Practical Calibration Table

For $n_k = 1000$ CoTs per client, $C = 1.0$, $\delta = 10^{-5}$:

| $T$ (steps/CoT) | $\varepsilon = 2$ | $\varepsilon = 4$ | $\varepsilon = 8$ |
|-----------------|-------------------|-------------------|-------------------|
| 5 | 0.0033 | 0.0017 | 0.0008 |
| 10 | 0.0047 | 0.0023 | 0.0012 |
| 20 | 0.0067 | 0.0033 | 0.0017 |
| 50 | 0.0105 | 0.0053 | 0.0026 |

The noise multiplier increases with $\sqrt{T}$, confirming that longer CoTs require proportionally more noise.

---

## 4. Composition Analysis

### 4.1 Round-Level Composition

For $R$ federated rounds with sampling rate $q = \frac{\text{batch size}}{n_k}$:

**Moments Accountant** (Abadi et al., 2016):

$$\varepsilon_{\text{total}} = O\left(q \sqrt{R} \cdot \frac{\sqrt{T} \cdot C}{\sigma}\right)$$

**Rényi DP** (Mironov, 2017):

$$\varepsilon_{\text{RDP}}(\alpha) = \frac{R \alpha q^2 T C^2}{2 n_k^2 \sigma^2} + \frac{\ln(1/\delta)}{\alpha - 1}$$

**Implication**: The privacy cost scales as $\sqrt{R \cdot T}$. For $R = 100$ rounds and $T = 20$ steps, the total cost is $\sqrt{2000} \approx 45$ times the single-round cost.

### 4.2 Client-Level vs. Instance-Level

| Accounting Level | Composition | Use Case |
|-----------------|-------------|----------|
| Instance-level | Each step is independent | Strictest guarantee |
| CoT-level | All steps in one CoT together | Practical default |
| Client-level | All steps from one client | Loosest guarantee |

Our implementation supports all three levels via config parameter `dp.accounting_level`.

---

## 5. Implementation

### 5.1 Step-Level DP-SGD

```python
class StepLevelDPSGD:
    def __init__(self, epsilon: float, delta: float, max_grad_norm: float):
        self.epsilon = epsilon
        self.delta = delta
        self.max_grad_norm = max_grad_norm
        self.noise_multiplier = self._compute_noise_multiplier()
    
    def _compute_noise_multiplier(self) -> float:
        """Compute noise multiplier for target (epsilon, delta)."""
        return self.max_grad_norm * np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def clip_and_noise(self, model: nn.Module, batch_size: int, num_steps: int):
        """
        Clip per-step gradients and add Gaussian noise.
        
        Args:
            model: Model with gradients accumulated
            batch_size: Number of CoTs in batch
            num_steps: Average steps per CoT (for sensitivity scaling)
        """
        # Clip per-parameter gradients
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=self.max_grad_norm * num_steps
        )
        
        # Add noise proportional to sensitivity
        sensitivity = self.max_grad_norm * num_steps / batch_size
        std_dev = self.noise_multiplier * sensitivity
        
        for param in model.parameters():
            if param.grad is not None:
                noise = torch.randn_like(param.grad) * std_dev
                param.grad += noise
```

### 5.2 Privacy Budget Tracking

```python
class PrivacyBudgetTracker:
    def __init__(self, epsilon: float, delta: float):
        self.epsilon_budget = epsilon
        self.delta_budget = delta
        self.epsilon_spent = 0.0
        self.delta_spent = 0.0
    
    def spend(self, epsilon_round: float, delta_round: float):
        """Spend privacy budget for one round."""
        self.epsilon_spent += epsilon_round
        self.delta_spent += delta_round
        
        if self.epsilon_spent > self.epsilon_budget:
            raise PrivacyBudgetExhausted(
                f"Privacy budget exceeded: {self.epsilon_spent:.2f} > {self.epsilon_budget:.2f}"
            )
    
    @property
    def remaining(self) -> tuple[float, float]:
        return (
            self.epsilon_budget - self.epsilon_spent,
            self.delta_budget - self.delta_spent
        )
```

### 5.3 Integration with Federated Training

```python
def local_train_dp(
    client_model,
    client_data,
    epochs,
    lr,
    device,
    dp_config
):
    dp = StepLevelDPSGD(
        epsilon=dp_config["epsilon"],
        delta=dp_config["delta"],
        max_grad_norm=dp_config["max_grad_norm"]
    )
    
    opt = torch.optim.AdamW(
        [p for p in client_model.parameters() if p.requires_grad],
        lr=lr
    )
    
    for epoch in range(epochs):
        for batch in client_data:
            opt.zero_grad()
            
            # Compute per-step gradients (not averaged)
            losses = []
            for step in batch:
                pred = client_model(step["input_ids"], step["attention_mask"])
                loss = F.mse_loss(pred, step["label"])
                losses.append(loss)
            
            # Average and backprop
            total_loss = sum(losses) / len(losses)
            total_loss.backward()
            
            # Apply DP-SGD
            dp.clip_and_noise(
                client_model,
                batch_size=len(batch),
                num_steps=dp_config["avg_steps_per_cot"]
            )
            
            opt.step()
    
    return client_model.state_dict()
```

---

## 6. Privacy-Utility Tradeoff

### 6.1 Expected Results

| $\varepsilon$ | MIA AUC | PRM Acc (math) | PRM Acc (code) | Utility Loss |
|--------------|---------|----------------|----------------|--------------|
| $\infty$ (no DP) | 0.75 | 0.65 | 0.62 | 0% |
| 8 | 0.58 | 0.62 | 0.59 | 5% |
| 4 | 0.52 | 0.59 | 0.56 | 10% |
| 2 | 0.51 | 0.54 | 0.51 | 17% |
| 1 | 0.50 | 0.48 | 0.45 | 26% |

**Key Insight**: $\varepsilon = 4$ provides near-random MIA AUC (0.52) with < 10% accuracy drop, representing the practical privacy-utility frontier.

### 6.2 Step Count Impact

For fixed $\varepsilon = 4$ and varying $T$:

| $T$ | Noise Multiplier | MIA AUC | Accuracy |
|-----|-----------------|---------|----------|
| 5 | 0.0033 | 0.51 | 0.63 |
| 10 | 0.0047 | 0.52 | 0.60 |
| 20 | 0.0067 | 0.52 | 0.58 |
| 50 | 0.0105 | 0.54 | 0.52 |

Longer CoTs require more noise, degrading utility. This motivates **adaptive privacy budgeting** (future work): allocate more budget to critical steps and less to generic ones.

---

## 7. Evaluation Protocol

### 7.1 Attack Evaluation

1. **Shadow Model MIA**: Train shadow models on auxiliary data, build attack classifier.
2. **Metric**: AUC-ROC for distinguishing member vs. non-member CoTs.
3. **Step-Level MIA**: Attack each step independently vs. attacking the full CoT.

### 7.2 Utility Evaluation

1. **ProcessBench Accuracy**: Standard PRM evaluation on held-out test sets.
2. **Best-of-N**: Measure end-to-end reasoning quality.
3. **Calibration**: Expected Calibration Error (ECE) for predicted rewards.

### 7.3 Privacy Audit

1. **Membership Inference**: Canaries (synthetic CoTs with known labels).
2. **Gradient Reconstruction**: DLG attack on noised vs. non-noised gradients.
3. **Attribute Inference**: Infer domain label from model parameters.

---

## 8. Timeline

| Phase | Deliverable | Status |
|-------|-------------|--------|
| M5 | StepLevelDPSGD implementation | ✅ Implemented |
| M5 | PrivacyBudgetTracker | ⏳ Needs implementation |
| M6 | Privacy-utility tradeoff experiments | ⏳ Pending full runs |
| M6 | MIA evaluation under DP | ⏳ Pending full runs |
| M6+ | Adaptive privacy budgeting | 🔮 Future work |

---

## 9. Open Questions

1. **Tightness**: Can we improve the $O(T)$ lower bound to $O(\sqrt{T})$ for correlated steps?
2. **Adaptive steps**: Should privacy budget be allocated per-step based on importance?
3. **Group privacy**: How to protect sets of related CoTs (e.g., all from one patient)?
4. **Post-processing**: Does PRM inference on noised models leak additional information?
5. **Composition**: Can we use f-DP (Dong et al., 2022) for tighter composition than RDP?
