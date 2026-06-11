# FCL-PRM：联邦全参数 PRM 微调 —— 项目框架

> 创建日期：2026-06-11
> 背景：旧方向（Anchor-PRM / permutation rebasin）在 RTX 4070 上完成 M2-M4 实验，核心创新经诊断无效。当前转向联邦全参数 PRM 微调路线，利用本设备（NVIDIA GB10，121GB 统一内存）的硬件优势。

---

## 一、核心叙事

> **首次系统性研究联邦场景下 step-level PRM 的全参数微调，证明 head-only 训练存在根本性容量瓶颈，而全参数微调在统一内存架构上可以首次实现 centralized-equivalent 性能。**

### 关键 Insight

M2 centralized 实验中，head-only PRM 已达 99.7% val_accuracy——任务太简单了，256-dim head 已足够好。这意味着：

1. **联邦场景下 head-only 没有足够的参数容量来产生有意义的跨客户端差异** → FedAvg 没有劣化空间，对齐也没有改善空间
2. **全参数微调给模型足够的容量来学习领域特异性** → 跨客户端 divergence 上升，聚合/对齐才有意义
3. **当前硬件（GB10 121GB 统一内存）是第一个能让 Pythia-1.4B 全参数联邦训练跑完的消费级设备**

---

## 二、实验规划

三个递进阶段，每个阶段可独立停止并发表。

### 阶段 1：能力验证（2-3 周）

**目标：验证"全参数 FT 显著优于 head-only"**

| 实验 | 模型 | 训练方式 | 说明 |
|------|------|---------|------|
| M2-centralized-head | Pythia-1.4B | head-only | ✅ 已有，acc=99.7% |
| M2-centralized-full | Pythia-1.4B | **全参数 FT** | 新基线，预期 99.9%+ |
| M3-fedavg-head | Pythia-1.4B | head-only FedAvg | ✅ 已有（旧设备），预期掉点 |
| M3-fedavg-full | Pythia-1.4B | **全参数 FedAvg** | 🔥 核心实验，预期接近 centralized |
| M3-fedavg-lora | Pythia-1.4B | **LoRA (rank=32)** | 轻量对比，3 天出结果 |

### 阶段 2：扩展诊断（2-3 周）

| 实验 | 问题 |
|------|------|
| full vs head CD-SPI 对比 | 全参数训练下 CD-SPI 是否显著更高？ |
| 模型规模扫描（1.4B→2.8B→6.9B） | 更大模型是否带来更大跨客户端差异？ |
| 通信效率实验 | Top-k 梯度压缩 vs full state dict 传输 |
| Partial FT 扫描 | 最后 N 层 FT vs LoRA R=rank vs Adapter 的 Pareto frontier |

### 阶段 3：创新方法（4 周）

基于阶段 1-2 结果选择方向：

- **方向 A**（如果 full FT FedAvg 因 client drift 掉点）：客户端感知的模型合并——用 CD-SPI 度量每个客户端偏移，动态决定聚合权重
- **方向 B**（如果 full FT FedAvg 追平 centralized）：高效联邦全参数 PRM 训练协议——梯度压缩 + 差异化通信
- **方向 C**（并行）：CD-SPI 作为联邦学习中模型差异的诊断工具的系统验证

---

## 三、代码架构

```
FCL-PRM/
├── PROJECT_FRAMEWORK.md        ← 本文项目框架
├── CLAUDE.md                   ← Claude Code 项目上下文
├── UPLOAD_AND_ANALYSIS.md      ← 旧设备数据上传与 v1 失败分析计划
│
├── configs/                    ← 实验配置文件（YAML）
│   ├── m2_centralized.yaml     ← M2 centralized head-only 基线
│   ├── m3_fedavg_full.yaml     ← M3 FedAvg 全参数微调
│   ├── m3_fedavg_lora.yaml     ← M3 FedAvg LoRA
│   └── smoke/                  ← 冒烟测试配置
│
├── src/fclprm/                 ← 核心源码
│   ├── models/
│   │   ├── base_wrapper.py     ← StepRewardModel: 冻结/解冻 backbone 可选
│   │   ├── prm_head.py         ← PRMHead: Linear+ReLU+Linear
│   │   └── lora_wrapper.py     ← LoRA 封装（阶段 1 新增）
│   ├── federated/
│   │   ├── client.py           ← FederatedClient: 本地训练 + 嵌入提取
│   │   ├── server.py           ← FederatedServer: 聚合调度
│   │   ├── simulator.py        ← FederatedSimulator: 单机多客户端调度
│   │   └── aggregators.py      ← 聚合策略（fedavg + 未来扩展）
│   ├── metrics/
│   │   └── cd_spi.py           ← CD-SPI 计算（跨客户端 divergence）
│   ├── data/                   ← 数据加载
│   └── utils/                  ← 工具函数
│
├── scripts/                    ← 运行入口
│   ├── run_federated.py        ← 联邦模拟主入口
│   ├── train_centralized_prm.py ← 集中式训练
│   └── eval_federated.py       ← 联邦评估
│
├── experiments/                ← 实验输出（不提交 .pt 检查点）
│   ├── M2_centralized_prm/     ← M2 结果（含 JSONL 基线）
│   ├── M3_naive_fedavg_prm/    ← M3 head-only 参考结果
│   └── 01_full_ft/             ← 阶段 1 新实验输出目录
│
├── tests/                      ← 单元测试
├── docs/                       ← 设计文档
└── venv/                       ← Python 虚拟环境
```

---

## 四、硬件护城河

本设备（NVIDIA GB10）是唯一能完整运行以下实验的消费级 GPU：

| 实验配置 | GB10 (121GB) | 4070 (12GB) |
|---------|-------------|-------------|
| Pythia-1.4B 全参数微调 | ✅ ~21GB | ❌ |
| Pythia-6.9B 全参数微调 | ✅ ~63GB | ❌ |
| Pythia-6.9B LoRA | ✅ ~16GB | ⚠️ 临界 |
| LLaMA-3-8B LoRA | ✅ ~18GB | ❌ |
| 4 客户端全参数 FedAvg | ✅ 逐客户端串行 | ❌ 单客户端都装不下 |

**论文第一句话草稿**："We present the first systematic study of full-parameter fine-tuning for federated step-level reward models, enabled by the unified memory architecture of NVIDIA Grace Blackwell."

---

## 五、关键假设（需阶段 1 验证）

| 假设 | 验证实验 | 如不成立的影响 |
|------|---------|--------------|
| H1: 全参数 FT 在 centralized 上显著优于 head-only | M2-centralized-full vs M2-centralized-head | 项目核心假设错误 → 回到 v1 问题 |
| H2: 联邦全参数 FT 比 head-only 更接近 centralized 水平 | M3-fedavg-full vs M2-centralized-full | 论文 title 改为"联邦全参数 PRM 的挑战" |
| H3: 联邦全参数 FT 会产生更大 CD-SPI（即跨客户端分歧更大）| full vs head CD-SPI 曲线 | 全参数 FT 不需要特殊聚合策略 |
| H4: 消费者级 GPU（GB10）可完成完整实验 | 冒烟测试跑通 | 迁移到云 GPU / A100 |

---

## 六、未来方向（阶段 2+）

- **模型规模扩展**：Pythia-2.8B → 6.9B，验证趋势是否持续
- **通信压缩**：Top-k gradient sparsification / QSGD / 1-bit SGD
- **异构客户端**：不同客户端不同 batch_size / local_epochs
- **个性化联邦 PRM**：每个客户端保留部分私有层
- **DP-SGD 集成**：复用 src/fclprm/attacks/ 代码，评估差分隐私下全参数微调的 utility 损失
