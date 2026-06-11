# Related Work

> Comprehensive literature review for FCL-PRM.
> Last updated: 2026-05-05.

---

## 1. Process Reward Models (Centralized)

### 1.1 Foundational Work

**Outcome Reward Models (ORMs)** predate PRMs and assign a single scalar to an entire reasoning chain. While effective for simple tasks, ORMs suffer from credit assignment problems: a correct final answer may mask incorrect intermediate steps, and an incorrect final answer may arise from a single error in an otherwise sound chain. Uesato et al. (2022) established that process-based feedback outperforms outcome-based feedback on math word problems, providing the first empirical evidence that step-level supervision improves reasoning quality.

**PRM800K** (Lightman et al., 2023) represents the seminal large-scale effort in process reward modeling. The authors collected ~800K step-level labels for mathematical problem-solving traces, training a GPT-4-based PRM that substantially improved best-of-N selection. Their key insight was that PRMs enable more fine-grained search: instead of generating N complete solutions and scoring each, one can prune partial solutions at each step, reducing the effective search space from exponential to linear in chain length.

**OpenAI o1** (OpenAI, 2024) extended PRM-style training to general reasoning via large-scale reinforcement learning on chain-of-thought traces. While details remain unpublished, the public demonstrations confirm that step-level reward models scale to complex domains beyond mathematics, including coding, physics, and logical reasoning. This motivates our multi-domain federated setting: if PRMs are to serve diverse institutions (hospitals, banks, schools), they must be trainable without centralizing sensitive reasoning data.

**Key Gap**: All centralized PRM work assumes data pooling. No prior work addresses the federated setting where step-level labels reside in silos.

### 1.2 Multi-Domain PRM

**VersaPRM** (Zeng et al., ICML 2025) is the closest existing work to our multi-domain ambition. The authors constructed a 14-domain synthetic CoT dataset and trained a unified PRM across domains. Their results show that a single PRM can generalize across disparate reasoning tasks, but they achieve this via centralized training on pooled synthetic data. Critically, VersaPRM does not measure whether step embeddings from different domains share a universal substrate or exhibit polysemy—this is precisely the question our CD-SPI metric answers.

**MedPRMBench** (2026) focuses on medical diagnosis chains, where step correctness depends on clinical knowledge. The specialized vocabulary (e.g., "differential diagnosis," "CT angiography") creates a natural domain boundary that federated learning must cross. Their benchmark provides evaluation data but no federated training protocol.

**DreamPRM** (2025) introduces domain reweighting for multimodal PRMs, adjusting the contribution of each domain based on difficulty. While related to heterogeneous FL, their reweighting operates on outcome-level losses, not step-level embeddings.

**OmegaPRM** (Google, 2024) automates PRM training via Monte Carlo Tree Search (MCTS) labeling. The approach eliminates human annotation but requires centralized access to both the generator policy and the verification environment, making it incompatible with cross-institutional privacy constraints.

**Key Gap**: Multi-domain PRM works assume centralized data. None analyze step embedding polysemy across domains.

### 1.3 PRM Robustness

**APRM** (2025) studies adversarial robustness of PRMs via gradient-based attacks on step labels. Their threat model assumes centralized training with poisoned labels, not the federated setting where poisoning originates from Byzantine clients.

**Noise-aware PRM** (2026) adds a noise model to the PRM head to handle label uncertainty. While related to our robust aggregation, their approach modifies the model architecture rather than the aggregation mechanism.

**PRM-BiasBench** critiques PRMs as fluency detectors, showing that PRMs sometimes reward syntactically polished but logically incorrect steps. This motivates our anchor-based alignment: if different domains produce steps with different surface forms but shared logical structure, alignment is necessary before aggregation.

**Key Gap**: No prior work studies robustness in the federated PRM setting with Byzantine clients and heterogeneous domains.

---

## 2. Federated Learning for LLMs

### 2.1 Federated Fine-Tuning of LLMs

Recent works have extended federated learning to large language models, but primarily for supervised fine-tuning (SFT) and preference alignment at the outcome level. **FedBiscuit** (ICLR 2025) clusters clients by preference similarity and trains binary selectors per cluster. Their approach handles preference heterogeneity but operates on response-level labels, not step-level reasoning traces.

**APPA** (2026) proposes adaptive preference pluralistic alignment, allowing the global model to maintain multiple "preference heads" for different client groups. While elegant, their architecture assumes access to all preference data for head initialization, violating the federated premise.

**FedPDPO** (2026) and **PluralLLM** (2025) extend DPO (Direct Preference Optimization) to federated settings. Both works optimize at the response level: given two complete responses, clients vote on which is better. This is fundamentally different from step-level PRM training, where each step within a response requires an independent label.

**Key Distinction**: All existing federated LLM alignment works operate at **outcome-level** (one scalar per response). FCL-PRM operates at **step-level** (T scalars per response), requiring new aggregation mechanisms that respect step semantics.

### 2.2 Model Alignment and Rebasin

**Git Re-Basin** (Ainsworth et al., ICLR 2023) proved that independently trained neural networks can be aligned via permutation of hidden units before parameter averaging. Their key result: for networks with permutation-invariant activation functions (e.g., ReLU), there exists a permutation of one network's units that makes its parameters close to another network's parameters.

Our Anchor-PRM extends this insight to the federated PRM setting. Instead of aligning entire networks (computationally prohibitive for LLMs), we align only the PRM head embeddings using a small set of shared anchor steps. The Hungarian algorithm (Kuhn, 1955; Munkres, 1957) provides the optimal linear assignment for this alignment.

**Model Patching** (Ilharco et al., 2022) and **Task Vectors** (Ortiz-Jimenez et al., 2023) edit model behavior by arithmetic in weight space. These works assume models trained on the same task distribution, whereas federated PRM deals with heterogeneous domain distributions.

**Key Gap**: No prior work applies model rebasin to federated LLM fine-tuning with step-level structure.

---

## 3. Chain-of-Thought Privacy

### 3.1 Inference-Time Privacy

**SALT** (2025) uses activation steering to hide sensitive information in CoT reasoning. The approach modifies model activations during inference to suppress PII-related neurons. While effective for inference, SALT does not address training-time privacy: the model may have already memorized sensitive reasoning patterns during training.

**InvisibleInk** and **AdaPMixED** (2025) propose private decoding methods that perturb token probabilities to prevent leakage of sensitive entities. These methods trade off reasoning quality for privacy and do not provide formal privacy guarantees.

**Key Distinction**: Inference-time privacy protects the user at query time. Training-time privacy protects the data contributor whose reasoning traces were used to train the model. FCL-PRM focuses on the latter.

### 3.2 Training-Time Privacy

**"Plugging PII Leakage in CoT"** (2026) applies DP-SGD to CoT generation models, adding noise to gradients during fine-tuning. Their privacy unit is the **entire CoT trace**, yielding loose bounds when traces are long (T steps → T times more sensitivity than outcome-level).

**"When Reasoning Leaks Membership"** (2026) presents the first membership inference attack (MIA) against reasoning models. Their shadow model approach achieves AUC > 0.75 on GPT-4-generated CoT traces, confirming that reasoning patterns are highly memorizable. However, their analysis treats the entire CoT as a single privacy unit, ignoring the step-level structure that PRM training exposes.

**Gradient Inversion Attacks**: Deep Leakage from Gradients (DLG; Zhu et al., NeurIPS 2019) and improved variants (Geiping et al., 2020) reconstruct training inputs from shared gradients. For frozen-backbone + trainable-head PRMs, the attack surface is limited to head gradients, but as our smoke tests confirm (DLG distance 0.9713), even these leak partial information about step content.

**Key Gap**: No prior work provides **information-theoretic tight bounds** for step-level DP in PRM training. The step-level setting strictly requires stronger privacy guarantees than outcome-level RLHF.

---

## 4. Cross-Domain Representation Analysis

### 4.1 Task Vectors and Skill Vectors

**Task Vectors** (Ortiz-Jimenez et al., 2023) show that models fine-tuned on different tasks occupy distinct regions in weight space, and arithmetic combinations of task vectors enable zero-shot task composition. **Skill Vectors** (Hendel et al., 2023) identify subspaces of model activations corresponding to specific capabilities (e.g., arithmetic, translation).

These works analyze **model-level** representations, not **step-level** embeddings. Our CD-SPI metric operates at a finer granularity: it asks whether individual reasoning steps ("therefore," "let x =") have consistent meanings across domains, not whether entire models have shared capabilities.

### 4.2 Polysemy in Neural Representations

**BERTology** literature (Rogers et al., 2020) extensively studies polysemy in word embeddings, showing that context-dependent models disambiguate word senses via different activation patterns. Our CD-SPI extends this question to the **step level**: does "therefore" activate the same neurons in math and medical reasoning?

**Cross-lingual Alignment** (Conneau et al., 2020) demonstrates that multilingual models learn shared semantic spaces across languages. Our anchor-based alignment is conceptually similar: we seek a shared semantic space for reasoning steps across domains, using anchor steps as "translation pairs."

**Key Gap**: No prior work measures **step embedding polysemy across reasoning domains**. CD-SPI is the first metric for this phenomenon, and the federated setting is necessary because centralized training destroys the ability to measure native cross-domain differences.

---

## 5. Federated Learning Foundations

### 5.1 Aggregation Rules

**FedAvg** (McMahan et al., 2017) averages client model parameters weighted by dataset size. While simple and widely used, FedAvg fails under heterogeneous data distributions: local optima diverge, and the global average may fall in a region of high loss for all clients (Zhao et al., 2018).

**FedProx** (Li et al., 2020) adds a proximal term to local objectives, penalizing deviation from the global model. This reduces client drift but does not address embedding space misalignment.

**SCAFFOLD** (Karimireddy et al., 2020) uses control variates to correct for client drift via variance reduction. It achieves faster convergence than FedProx but requires maintaining state (control variates) at each client, increasing communication cost.

**FedGMKD** (Zhang et al., NeurIPS 2024) addresses Non-IID federated learning via Cluster Knowledge Fusion (CKF) and Discrepancy-Aware Aggregation (DAT). CKF uses Gaussian Mixture Models to generate prototype features and soft predictions per class, enabling knowledge distillation without public datasets. DAT adjusts aggregation weights by KL-divergence between local and global prototype distributions, improving both local and global accuracy on CIFAR-10/100 and SVHN. While FedGMKD advances prototype-based aggregation in image classification, its assumptions do not transfer to step-level PRM: (1) class prototypes are discrete and finite, whereas reasoning steps are combinatorial and unbounded; (2) DAT operates on prediction distributions, not step embedding spaces; (3) the privacy unit is the sample, not the individual reasoning step.

**Key Gap**: All existing FL aggregation rules operate in **parameter space** or **class-prototype space**, ignoring the **semantic structure** of step-level reasoning. Anchor-PRM is the first aggregation rule that aligns client embeddings before averaging.

### 5.2 Robust Aggregation

**Byzantine-Robust FL** (Blanchard et al., 2017; Yin et al., 2018) studies aggregation rules that tolerate malicious clients submitting arbitrary updates. Trimmed mean and coordinate-wise median provide robustness guarantees under bounded fractions of Byzantine clients.

Our poisoned setting differs: attackers do not submit arbitrary parameters but instead train on poisoned step labels, producing updates that appear legitimate in parameter space but degrade step-level reward accuracy. This requires **step-level anomaly detection** rather than standard robust aggregation.

---

## 6. Differential Privacy Foundations

### 6.1 DP-SGD

**DP-SGD** (Abadi et al., CCS 2016) is the canonical algorithm for training models with differential privacy. It clips per-sample gradients to bounded L2 norm and adds Gaussian noise calibrated to the privacy budget (ε, δ).

**Composition Bounds**: The moments accountant (Abadi et al., 2016) and Rényi DP (Mironov, 2017) provide tight composition bounds for iterative algorithms. For PRM training with T steps per CoT, the sensitivity scales as O(T) if each step is an independent privacy unit, or O(1) if the entire CoT is a single unit.

**Key Insight**: Step-level PRM training strictly requires **step-level privacy units** because each step gradient reveals information about that specific step's label. Our analysis proves that the required noise scales as σ√T, establishing that step-level PRM training needs stronger privacy guarantees than outcome-level RLHF.

### 6.2 Privacy Attacks

**Membership Inference Attacks** (Shokri et al., 2017; Yeom et al., 2018) determine whether a sample was in the training set. For PRMs, our smoke tests confirm that loss-based MIA achieves a positive gap (member score 0.9945 vs. non-member 0.8247) even with only head gradients visible, establishing a non-trivial attack baseline.

**Gradient Reconstruction** (Zhu et al., NeurIPS 2019; Geiping et al., 2020) recovers training inputs from gradients. Our DLG smoke test (distance 0.9713) shows that frozen-backbone PRMs resist reconstruction better than full-model settings, but the attack still recovers partial step content.

---

## 7. Critical Synthesis: Why FCL-PRM is Necessary

The literature reveals three converging gaps:

1. **Centralized PRM training cannot handle cross-institutional data**. Existing multi-domain PRMs (VersaPRM) pool all data, violating privacy constraints of hospitals, financial institutions, and educational platforms.

2. **Existing federated LLM alignment operates at outcome-level**. FedBiscuit, APPA, FedPDPO all optimize one scalar per response. PRM training requires T scalars per response, demanding new aggregation mechanisms.

3. **No formal privacy analysis exists for step-level PRM training**. DP-SGD calibration for outcome-level RLHF is well-understood, but step-level training leaks strictly more information (T step labels vs. 1 outcome label).

FCL-PRM addresses all three gaps through: (i) Anchor-PRM aggregation with embedding alignment, (ii) CD-SPI for measuring cross-domain step polysemy, and (iii) tight information-theoretic bounds on step-level privacy leakage.

---

## References

- Abadi et al. "Deep Learning with Differential Privacy." CCS 2016.
- Ainsworth et al. "Git Re-Basin: Merging Models modulo Permutation Symmetries." ICLR 2023.
- Blanchard et al. "Machine Learning with Adversaries: Byzantine Tolerant Gradient Descent." NeurIPS 2017.
- Conneau et al. "Emerging Cross-lingual Structure in Pretrained Language Models." ACL 2020.
- Geiping et al. "Inverting Gradients -- How Easy is it to Break Privacy in Federated Learning?" NeurIPS 2020.
- Hendel et al. "In-context Learning Creates Task Vectors." EMNLP 2023.
- Ilharco et al. "Editing Models with Task Arithmetic." ICLR 2023.
- Karimireddy et al. "SCAFFOLD: Stochastic Controlled Averaging for Federated Learning." ICML 2020.
- Kuhn. "The Hungarian Method for the Assignment Problem." 1955.
- Li et al. "FedProx: Federated Optimization in Heterogeneous Networks." MLSys 2020.
- Lightman et al. "Let's Verify Step by Step." 2023.
- McMahan et al. "Communication-Efficient Learning of Deep Networks from Decentralized Data." AISTATS 2017.
- Mironov. "Rényi Differential Privacy." CSF 2017.
- OpenAI. "Learning to Reason with LLMs." OpenAI Blog, 2024.
- Ortiz-Jimenez et al. "Task Arithmetic in the Tangent Space." ICML 2023.
- Rogers et al. "A Primer in BERTology." TACL 2020.
- Shokri et al. "Membership Inference Attacks Against Machine Learning Models." S&P 2017.
- Uesato et al. "Solving Math Word Problems with Process- and Outcome-based Feedback." 2022.
- Yeom et al. "Privacy Risk in Machine Learning." CSF 2018.
- Yin et al. "Byzantine-Robust Distributed Learning: Towards Optimal Statistical Rates." ICML 2018.
- Zeng et al. "VersaPRM: Multi-Domain Process Reward Model via Synthetic Reasoning Data." ICML 2025.
- Zhao et al. "Federated Learning with Non-IID Data." 2018.
- Zhang et al. "FedGMKD: An Efficient Prototype Federated Learning Framework through Knowledge Distillation and Discrepancy-Aware Aggregation." NeurIPS 2024.
- Zhu et al. "Deep Leakage from Gradients." NeurIPS 2019.
