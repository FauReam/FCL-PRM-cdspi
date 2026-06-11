# FCL-PRM：联邦全参数 PRM 微调 —— 项目框架

> 创建日期：2026-06-11
> 旧方向（Anchor-PRM / permutation rebasin）在 RTX 4070 上完成 M2-M4 实验，核心创新经诊断无效。当前转向联邦全参数 PRM 微调路线，利用本设备（NVIDIA GB10，121GB 统一内存）的硬件优势。

---

## 一、核心叙事

> **首次系统性研究联邦场景下 step-level PRM 的全参数微调，证明 head-only 训练存在根本性容量瓶颈，而全参数微调在统一内存架构上可以首次实现 centralized-equivalent 性能。**

### 关键 Insight

M2 centralized 实验中，head-only PRM 已达 99.7% val_accuracy——任务太简单了，256-dim head 已足够好。这意味着：

1. **联邦场景下 head-only 没有足够的参数容量来产生有意义的跨客户端差异** → FedAvg 没有劣化空间，对齐也没有改善空间
2. **全参数微调给模型足够的容量来学习领域特异性** → 跨客户端 divergence 上升，聚合才有意义
3. **当前硬件（GB10 121GB 统一内存）是第一个能让 Pythia-2.8B 全参数联邦训练跑完的消费级设备**

---

## 二、模型选择

| 角色 | 模型 | 训练方式 | 峰值内存 | 预估时间 |
|------|------|---------|---------|---------|
| **🔴 主实验** | **Pythia-2.8B** | **全参数 FT** | ~40 GB | ~7-8 天 |
| **🔵 辅助 1** | **Pythia-1.4B** | **全参数 FT** | ~21 GB | ~3-4 天 |
| **🟢 辅助 2** | **LLaMA-3.1-8B** | **head-only** | ~16 GB | ~1 天 |

**选型逻辑：**
- Pythia-2.8B 全参数 FT 在 GB10 上内存充裕（40GB/121GB），训练时间可控（~1 周）
- Pythia-1.4B 同架构小规模，形成 1.4B→2.8B scaling 趋势线
- LLaMA-3.1-8B head-only 验证跨架构泛化（LoRA 支持待扩展）

---

## 三、实验规划

### 阶段 1：集中式锚点（新 M2）

在 centralized 数据上建立全参数微调的上限：

| 实验 | 配置 | 目的 |
|------|------|------|
| M2-full-2.8B | `m2_centralized_full_2.8b.yaml` | **主锚点**：全参数 FT 的 centralized upper bound |
| M2-full-1.4B | `m2_centralized_full_1.4b.yaml` | 规模对比，形成 1.4B→2.8B 趋势线 |
| M3-fedavg-head-LLaMA-8B | `m3_fedavg_head_llama_8b.yaml` | head-only 限制是否跨架构成立？ |

### 阶段 2：联邦全参数微调（新 M3）

核心对比实验：

| 实验 | 配置 | 核心问题 |
|------|------|---------|
| M3-fedavg-full-2.8B | `m3_fedavg_full_2.8b.yaml` | 全参数 FedAvg 是否接近 centralized 水平？ |
| M3-fedavg-full-1.4B | `m3_fedavg_full_1.4b.yaml` | 小模型下趋势是否一致？ |

### 阶段 3：扩展与诊断

基于阶段 1-2 的结果：
- CD-SPI 对比（full vs head）：全参数训练下跨客户端差异是否显著增大？
- Partial FT 扫描：最后 N 层 vs full FT 的 Pareto frontier
- 通信效率：梯度压缩对全参数联邦的影响

### 附加实验：AttnRes backbone（可选，2026-06 新增）

**引用**：[Attention Residuals](https://arxiv.org/abs/2603.15031) (Kimi Team, arXiv Mar 2026)

AttnRes 将标准残差连接的固定累加替换为深度维度的 softmax 注意力，在 LLM 预训练中表现出 1.25× compute advantage。它与本项目核心论点完全正交——如果有效，可作为论文的额外卖点；如果无效，不影响主线结论。

| 实验 | 配置 | 目的 |
|------|------|------|
| M2-AttnRes-1.4B | `m2_centralized_full_1.4b_attnres.yaml` | AttnRes backbone 在 centralized PRM 上的效果 |
| M3-AttnRes-1.4B | `m3_fedavg_full_1.4b_attnres.yaml` | AttnRes 是否能减少联邦客户端漂移 |

---

## 四、代码架构

```
FCL-PRM-fullft/
├── PROJECT_FRAMEWORK.md     ← 本文
├── CLAUDE.md                ← 项目上下文
├── README.md                ← 简短介绍
│
├── configs/                 ← 实验配置（YAML）
│   ├── m2_centralized_full_1.4b.yaml              ← M2 辅助锚点
│   ├── m2_centralized_full_2.8b.yaml              ← M2 主锚点
│   ├── m2_centralized_full_1.4b_attnres.yaml      ← M2 + AttnRes（可选）
│   ├── m3_fedavg_full_1.4b.yaml                   ← M3 辅助
│   ├── m3_fedavg_full_1.4b_attnres.yaml           ← M3 + AttnRes（可选）
│   ├── m3_fedavg_full_2.8b.yaml                   ← M3 主实验
│   └── m3_fedavg_head_llama_8b.yaml               ← 跨架构验证
│
├── src/fclprm/
│   ├── models/
│   │   ├── base_wrapper.py    ← StepRewardModel（核心模型包装器）
│   │   ├── prm_head.py        ← PRMHead
│   │   ├── attnres_backbone.py ← Block AttnRes backbone（可选模块）
│   │   └── checkpoint.py      ← 检查点管理
│   ├── metrics/
│   │   └── cd_spi.py          ← 跨客户端散度度量
│   ├── federated/
│   │   ├── client.py          ← FederatedClient
│   │   ├── server.py          ← FederatedServer
│   │   ├── simulator.py       ← 单机多客户端调度
│   │   ├── aggregators.py     ← 聚合策略（FedAvg / Anchor-PRM / 鲁棒）
│   │   └── dp.py              ← 差分隐私
│   ├── data/                  ← 数据加载（PRM800K, VersaPRM, Medical）
│   └── utils/                 ← 工具函数
│
├── scripts/
│   ├── run_federated.py          ← 联邦模拟主入口
│   ├── train_centralized_prm.py  ← 集中式训练
│   └── prepare_versaprm.py       ← 数据准备
│
├── experiments/                  ← 实验输出
├── tests/
└── docs/
```

---

## 五、硬件护城河

本设备（NVIDIA GB10）是唯一能在消费级 GPU 上运行以下实验的设备：

| 实验配置 | GB10 (121GB) | 4090 (24GB) | A100 (80GB) |
|---------|-------------|-------------|-------------|
| Pythia-1.4B full FT | ✅ 21GB | ⚠️ 临界 | ✅ |
| Pythia-1.4B + AttnRes | ✅ 21GB + ~2MB | ⚠️ 临界 | ✅ |
| Pythia-2.8B full FT | ✅ 40GB | ❌ | ✅ |
| Pythia-2.8B full FT 联邦 | ✅ 串行 40GB | ❌ | ⚠️ 需共享 |
| LLaMA-3.1-8B head-only | ✅ 16GB | ✅ | ✅ |

**论文地位**："To our knowledge, this is the first study to demonstrate full-parameter federated fine-tuning of step-level reward models on a consumer-grade GPU, enabled by the unified memory architecture of NVIDIA Grace Blackwell."

---

## 六、关键假设（需阶段 1 验证）

| 假设 | 验证实验 | 如不成立的影响 |
|------|---------|--------------|
| **H1: 全参数 FT 在 centralized 上显著优于 head-only** | M2-full vs 旧 M2 (99.7%) | 核心假设错误，项目终止 |
| **H2: 联邦全参数 FT 比 head-only 更接近 centralized** | M3-full vs M2-full | 论文改为"挑战分析" |
| **H3: 全参数 FT 产生更大跨客户端差异 (CD-SPI)** | full vs head CD-SPI 对比 | 全参数不需要特殊聚合 |
| **H4: 趋势在 1.4B→2.8B 上一致** | 两组实验对比 | 规模不可扩展 |
| H5 (可选): AttnRes 在 centralized PRM 上优于标准残差 | M2-full vs M2-AttnRes | 不影响主线结论 |
| H6 (可选): AttnRes 降低联邦客户端漂移 | M3 vs M3-AttnRes (CD-SPI) | 不影响主线结论 |
