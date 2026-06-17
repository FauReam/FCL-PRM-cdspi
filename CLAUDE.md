# FCL-PRM — Claude Code 项目上下文

> **新会话必读**：[SESSION_NOTES.md](SESSION_NOTES.md) — 硬件限制、已知 Bug、数据统计、性能权衡、配置速查

## 项目简介
Federated Continual Process Reward Model (联邦持续过程奖励模型)。
跨机构联邦学习协同训练 step-level PRM。**当前方向：CD-SPI 发散结构诊断框架**（详见 PROJECT_FRAMEWORK.md）：

- **核心贡献**：CD-SPI 诊断框架，区分联邦聚合中的噪声型与信号型发散
- **容量作为实验变量**：head-only / LoRA / partial FT / full FT 仅是操控发散类型的手段
- 容量叙事（"full FT 更好"）已根据两次 expert panel 结论——**全体专家一致否决作为核心贡献**
- 设备：NVIDIA GB10，121GB 统一内存，ARM64 CPU

## Scope boundary（明确声明）
本文因果推断限于 **dense backbone (Pythia 系列) + FedAvg 聚合 + 标准异质性设定**。
不主张 full FT 是联邦 PRM 的唯一路径（必要性主张）。CD-SPI 作为诊断工具存在测量不对称性、
架构敏感性等已知局限。结论外推至 MoE/SCAFFOLD 等替代路径需独立验证。

## 模型选择
- **主实验**: Pythia-1.4B 容量连续谱 (head-only / LoRA r=8/64/128/256 / partial-FT / full-FT)
- **扩展**: Pythia-2.8B full FT（验证 scaling 趋势）
- **架构消融**: 三种 head 激活函数 (ReLU/GELU/Identity) 验证 CD-SPI 排序一致性

## 实验阶段
Phase 0（关键路径，2 周）→ Phase 1（控制实验，3 周）→ Phase 2（深度验证，3 周）

详见 PROJECT_FRAMEWORK.md 完整实验规划。

## 常用命令
```bash
# Phase 0-1: 集中式基线 + 对称 CD-SPI
python scripts/train_centralized_prm.py --config configs/m2_centralized_full_1.4b.yaml

# Phase 0-2: 容量连续谱 M3（核心实验）
python scripts/run_federated.py --config configs/m3_fedavg_head_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_lora_r8_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_lora_r64_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_lora_r256_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_partialft_last2_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_partialft_last4_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_partialft_last8_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_partialft_mlp_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_full_1.4b.yaml

# Phase 1-1: 架构消融（激活函数，默认ReLU已覆盖，GELU/Identity为消融）
python scripts/run_federated.py --config configs/m3_fedavg_head_1.4b_gelu.yaml
python scripts/run_federated.py --config configs/m3_fedavg_head_1.4b_identity.yaml

# Phase 1-3: CKA交叉验证（用已有config加symmetrical_cd_spi = true自动启用）
# Phase 2-4: OOD + 异质性（--ood和--label-noise为CLI覆盖）
python scripts/run_federated.py --config configs/m3_fedavg_full_1.4b.yaml --ood
python scripts/run_federated.py --config configs/m3_fedavg_full_1.4b.yaml --label-noise
# 覆盖轮数：
python scripts/run_federated.py --config configs/smoke_versaprm.yaml --rounds 5
```

## 关键路径
- 联邦模拟入口：`scripts/run_federated.py`
- 集中式训练：`scripts/train_centralized_prm.py`
- 客户端训练：`src/fclprm/federated/client.py`
- 服务器聚合：`src/fclprm/federated/server.py` / `aggregators.py`
- 模拟调度：`src/fclprm/federated/simulator.py`
- 模型定义：`src/fclprm/models/base_wrapper.py`（LoRA/partial-FT/AttnRes + 对称化嵌入）
- **对称化 CD-SPI**：`src/fclprm/models/base_wrapper.py` 中 `get_backbone_embedding()`
- 配置目录：`configs/`
- 度量：
  - `src/fclprm/metrics/cd_spi.py` — CD-SPI 核心（余弦相似度 + PCA EVR）
  - `src/fclprm/metrics/cd_spi_stats.py` — 排列检验、噪声注入、函数空间散度
  - `src/fclprm/metrics/cka.py` — CKA 独立交叉验证
  - `src/fclprm/metrics/ood_eval.py` — OOD cross-domain, label perturbation
- 数据分区：
  - `src/fclprm/data/heterogeneity.py` — Dirichlet, label shift, mixed patterns

## 硬件信息
- GPU: NVIDIA GB10 (Blackwell, 计算能力 12.1, CUDA 13.0)
- 内存: 121 GB CPU/GPU 统一内存架构
- CPU: ARM64 (Cortex-X925 + A725)
- 全参数 FT（BF16+Adam）内存：Pythia-1.4B ~21GB, Pythia-2.8B ~40GB

## 已知陷阱
1. **全参数 FT 内存**：用 BF16 加载 backbone，batch_size=4 起步
2. **检查点体积**：全参数 FT 约 5.6 GB（2.8B）/ 2.8 GB（1.4B），注意磁盘空间
3. **客户端串行训练**：4 客户端全参数串行，2.8B 约 1-2 天/25 rounds
4. **设备不匹配**：`_eval_per_domain` 需在返回前调 `.cpu()`
5. **检查点恢复**：崩溃发生在聚合前则不生成该轮检查点
6. **Opacus DP-SGD**：包覆后 `model._module` 才是原始模型
7. **[对称化测量 - 重要]** CD-SPI 必须从 backbone 倒数第二层 hidden state 统一提取（`get_backbone_embedding`），不能用 `get_head_embedding`（不同容量配置的嵌入空间不对等）
8. **[AttnRes] 仅支持 GPTNeoX（Pythia）架构**：`AttnResBackboneModel` 当前仅支持 GPTNeoX-based 模型
9. **[AttnRes] 零初始化必须**：伪查询向量必须初始化为 0（`zero_init=true`），不得更改
10. **[AttnRes] torch.compile 时机**：`run_federated.py` 中 `torch.compile` 在 `StepRewardModel` 构造之后
11. **[AttnRes] checkpoint 兼容性**：AttnRes 模型的 state_dict key 与标准残差模型不兼容

## 工作约定
- 提交信息用英文，遵循 `fix(scope): description` 格式
- 主分支：`main`（远程 `origin` 为 `https://github.com/FauReam/FCL-PRM-cdspi.git`）
- **训练代码规范**：详见 [TRAINING_CONVENTIONS.md](TRAINING_CONVENTIONS.md) — 进度条、检查点、错误栈、运行日志、终端断开存活、修改日志 7 条强制要求
- **Claude Code 启动训练规则**：任何 >1h 的训练必须用 `nohup` + `&` 启动，只返回 PID，日志写入项目 `experiments/` 目录，禁止流式输出到 Claude 对话。详见 [TRAINING_CONVENTIONS.md §0.5](TRAINING_CONVENTIONS.md#05-claude-code-启动训练程序规则强制性)
- **终端命令格式**：当用户索要运行命令时，必须以换行续行的代码块给出。每行不超过 50 字符，用 `\` 折行。禁止单行超长命令。
- **修改日志**：效率/算法类代码变更必须同时写入 [docs/CHANGELOG.md](docs/CHANGELOG.md)，commit message 中引用条目。详见 [TRAINING_CONVENTIONS.md §7](TRAINING_CONVENTIONS.md#7-修改日志-modification-log--效率与算法变更必须记录)

### 进度条（强制性）
- tqdm 覆盖 rounds / clients / batches，每层有 desc / total / postfix
- 每 N 个 batch 通过 `log_interval` 刷新 postfix

### 自动保存（强制性）
- `save_every` 控制频率，命名 `model_m{M}_r{round}_c{client}.pt`
- 保存前 `.cpu()`，保存后 `del`+`gc.collect()`+`torch.cuda.empty_cache()`
- 最终轮结束后强制保存
