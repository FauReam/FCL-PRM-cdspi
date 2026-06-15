# FCL-PRM-fullft

**Federated Continual Process Reward Model — CD-SPI Diagnostic Framework**  
联邦持续过程奖励模型 · CD-SPI 发散结构诊断框架

跨机构联邦学习协同训练 step-level PRM。**核心贡献：CD-SPI 诊断框架，区分联邦聚合中的噪声型与信号型散度。**

---

## 核心命题

本文提出 **CD-SPI（Client Divergence Signal-Noise Partition Index）诊断框架**——一个结构化的两阶段统计诊断协议，由排列检验（permutation test）和主成分解释方差比分析（PCA EVR）组成。

CD-SPI **不是新的相似度度量**（余弦相似度在 FL 中已被 CFL 2020, FedCDD 2025 广泛使用），而是将 FL 中已有的余弦相似度分析方法**系统化为可区分噪声型与信号型发散的统计诊断协议**，首次在联邦 PRM 微调场景下探究**容量-发散类型-聚合性能**之间的三元实证联系。

### 核心断言（H1-H4）

在 dense backbone (Pythia-1.4B/2.8B) + FedAvg 受控基线设置下：

| 假说 | 描述 | 验证方式 |
|:---:|---|:---:|
| **H1** | 低容量配置（head-only / LoRA r<256）中，跨客户端参数差异不可与噪声区分，且与 FedAvg 聚合性能无关 | CD-SPI 双重统计标准（排列检验 p>0.05 + PCA EVR<0.4） |
| **H2** | 全参数 FT 产生的参数差异呈现统计显著的结构化模式，且与下游聚合性能正相关 | 排列检验 p<0.05 + EVR>0.6 + 性能相关性 |
| **H3** | 存在容量转变阈值（噪声主导→语义结构主导），通过 LoRA rank 精细网格扫描定位 | LoRA r=8/64/128/256+ 精细扫描 |
| **H4** | 散度结构化程度与聚合性能之间存在因果关系 | CD-SPI 正则化干预 + 独立聚合指标交叉验证 |

### 证伪条件（F1-F5）

| 条件 | 描述 | 触发后果 |
|:---:|---|:---:|
| **F1** | head 维度增至 1024 后 CD-SPI 显示结构化模式 | "head-only 必然噪声"假设受容量限制 |
| **F2** | LoRA(r=64) 同时达 full FT 95% 性能且 CD-SPI 为噪声 | "结构化发散是聚合必要条件"被证伪 |
| **F3** | head-only 与 full FT 仅定量差异（程度不同）而非定性差异（噪声vs结构） | 二分法不成立，退化为连续谱叙事 |
| **F4** | 独立指标（CKA/JS 散度）与 CD-SPI 结论不一致 | CD-SPI 引入系统性偏差 |
| **F5** | 集中式基线中 head-only 与 full FT 性能差距 ≤ 2% | 非联邦特异问题，联邦特异性 claim 不成立 |

### 方法论要点

- **对称化测量**：CD-SPI 从 backbone 倒数第二层 hidden state 统一提取，确保 head-only 和 full FT 的测量空间一致
- **三层递进排列检验**：(a) 客户端标签置换 (b) 容量匹配独立训练基线 (c) 随机特征映射替换
- **交叉验证**：CD-SPI 之外强制加入 CKA（Centered Kernel Alignment）和函数空间 JS 散度
- **架构消融**：三种激活函数（ReLU, GELU, Identity）下验证 CD-SPI 排序一致性
- **效应量为主**：排列检验 p 值辅助展示，主要结论依靠 Cohen's d 效应量

---

## 实验规划（Phase 0 → 1 → 2）

### Phase 0 — 关键路径实验（2 周）

| # | 实验 | 目的 |
|:---:|---|:---:|
| 0-1 | M2 集中式基线 + 对称 CD-SPI | 在 VersaPRM 上验证天花板，建立对称测量基线 |
| 0-2 | 容量连续谱 M3 | LoRA r=8/64/128/256 + partial FT + head-only + full FT |
| 0-3 | 训练前 CD-SPI 基线 | 第 0 轮测量控制初始化混淆 |

### Phase 1 — 控制实验（3 周）

| # | 实验 | 目的 |
|:---:|---|:---:|
| 1-1 | **架构消融** | 3 种激活函数（ReLU/GELU/Identity）验证排序一致性 |
| 1-2 | **三层排列检验** | 增强 null distribution，排除随机性解释 |
| 1-3 | **CKA 交叉验证** | 独立指标重申 CD-SPI 结论 |
| 1-4 | **合成数据校准** | 已知发散结构数据验证精确率+召回率 |

### Phase 2 — 深度验证 + 理论（3 周）

| # | 实验 | 目的 |
|:---:|---|:---:|
| 2-1 | 散度指向性区分 | CD-SPI vs Wasserstein 距离 |
| 2-2 | 独立聚合指标 | FedDYN drift, SCAFFOLD 控制变分 |
| 2-3 | 反直觉条件测试 | 极端异质性下低容量结构化 > 高容量噪声？ |
| 2-4 | CD-SPI 理论形式化 | 有界性定理 + 尺度不变性 + FedAvg 收敛率联系 |
| 2-5 | OOD 鲁棒性 + 异质性消融 | 跨域 OOD + 标签扰动 + Dirichlet/label_shift/mixed |

---

## 实验执行命令

```bash
# Phase 0-1: 集中式基线
python scripts/train_centralized_prm.py --config configs/m2_centralized_full_1.4b.yaml

# Phase 0-2: 容量连续谱 M3
python scripts/run_federated.py --config configs/m3_fedavg_lora_r8_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_lora_r64_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_lora_r128_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_lora_r256_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_partialft_last2_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_partialft_last4_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_partialft_last8_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_partialft_mlp_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_head_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_full_1.4b.yaml

# Phase 1-1: 架构消融（激活函数）
python scripts/run_federated.py --config configs/m3_fedavg_head_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_head_1.4b_gelu.yaml
python scripts/run_federated.py --config configs/m3_fedavg_head_1.4b_identity.yaml

# Phase 2-5: OOD + 异质性
python scripts/run_federated.py --config configs/m3_fedavg_full_1.4b.yaml --ood
```

---

## 硬件

NVIDIA GB10（121GB 统一内存，ARM64）。可运行 Pythia-1.4B 全参数 FT（~21GB）至 Pythia-2.8B（~40GB）。

---

## 专家组诊断

> **2026-06-11 两次 Adversarial Panel**：

| 轮次 | 平均评分 | 关键发现 |
|:---:|:---:|:---|
| 第 1 轮（6 专家） | 4.25/10 | 缺少 LoRA 基线、CD-SPI 无统计完备性、99.7% 饱和 |
| 第 2 轮（4 专家×2 轮迭代） | 3.50/10 | CD-SPI 非独立方法贡献、「容量叙事」降级为实验变量、三项矛盾两项解决 |

详见完整报告：👉 **[EXPERT_PANEL_REPORT.md](./EXPERT_PANEL_REPORT.md)** / **[EXPERT_PANEL_REPORT_V2.html](./EXPERT_PANEL_REPORT_V2.html)**

---

## 关键路径

- `scripts/run_federated.py` — 联邦模拟主入口
- `scripts/train_centralized_prm.py` — 集中式训练
- `src/fclprm/models/base_wrapper.py` — StepRewardModel（LoRA/partial-FT/AttnRes + 对称化嵌入）
- `src/fclprm/metrics/cd_spi.py` — CD-SPI 核心（余弦相似度 + PCA EVR）
- `src/fclprm/metrics/cd_spi_stats.py` — 排列检验、噪声注入、函数空间散度
- `src/fclprm/metrics/cka.py` — CKA 独立交叉验证
- `src/fclprm/federated/simulator.py` — 联邦模拟调度器
- `configs/` — 实验配置（YAML）

---

## 许可证

MIT（待定）
