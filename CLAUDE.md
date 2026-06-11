# FCL-PRM — Claude Code 项目上下文

## 项目简介
Federated Continual Process Reward Model (联邦持续过程奖励模型)。
跨机构联邦学习协同训练 step-level PRM。

**当前方向：联邦全参数微调**（详见 PROJECT_FRAMEWORK.md）：
- 旧方向（Anchor-PRM / permutation rebasin）在 RTX 4070 上得出阴性结论，已停止
- 核心假设：head-only PRM 容量不足是联邦 PRM 的根本瓶颈，
  全参数微调在本设备（NVIDIA GB10，121GB 统一内存）上可首次实现

## 常用命令
```bash
# 训练模式：
#   freeze_backbone: true  → head-only（已有基线，使用现有 configs）
#   freeze_backbone: false → 全参数微调（新实验，需要 LoRA 或更小 BS）

# M3 全参数 FedAvg（核心实验）
python scripts/run_federated.py --config configs/m3_fedavg_full.yaml

# M3 LoRA FedAvg（轻量对比）
python scripts/run_federated.py --config configs/m3_fedavg_lora.yaml

# M2 centralized head-only（已有基线，再确认用）
python scripts/train_centralized_prm.py --config configs/m2_pythia_1b.yaml

# M3 head-only FedAvg（旧基线，对比用）
python scripts/run_federated.py --config configs/m3_naive_fedavg.yaml
```

## 关键路径
- 联邦模拟入口：`scripts/run_federated.py`
- 客户端训练：`src/fclprm/federated/client.py`
- 服务器聚合：`src/fclprm/federated/server.py` / `aggregators.py`
- 模拟调度：`src/fclprm/federated/simulator.py`
- 模型定义：`src/fclprm/models/base_wrapper.py`（`freeze_backbone` 参数控制冻结/全参数）
- 配置目录：`configs/`

## 已知陷阱
1. **全参数 FT 内存**：Pythia-1.4B 全参数 + Adam 峰值约 21 GB，batch_size=4 起步，用 BF16
2. **LoRA 聚合**当前用 naive FedAvg 聚合 adapter 权重，无特殊处理。如需个性化需改聚合器
3. **检查点体积**：全参数 FT 的检查点约为 head-only 的 50x（从 ~1MB 到 ~2.8GB），注意磁盘空间
4. **客户端串行训练 CPU 开销**：4 个客户端逐串行全参数训练总时间约 4 天（Pythia-1.4B, 25 rounds）
5. **设备不匹配**（旧陷阱保留）：`_eval_per_domain` 会把全局模型移到 GPU，必须在返回前调 `.cpu()`
6. **检查点恢复**（旧陷阱保留）：`save_every` 控制保存频率；崩溃发生在聚合前则该轮检查点不生成
7. **CD-SPI 需用 head embedding**（旧陷阱保留）：`_eval_cd_spi` 必须用 `get_head_embedding` 而非 `get_step_embedding`
8. **Opacus 包裹模型**（旧陷阱保留）：DP-SGD 开启时 `model._module` 才是原始模型

## 硬件信息
- GPU: NVIDIA GB10 (Blackwell, 计算能力 12.1, CUDA 13.0)
- 内存: 121 GB CPU/GPU 统一内存架构
- CPU: ARM64 (Cortex-X925 + A725)
- 本设备能运行实验：Pythia-1.4B→6.9B 全参数 FT, LLaMA-3-8B LoRA
- 本设备不能运行：70B 级全参数 FT（OOM），如需则迁移到云 GPU

## 工作约定
- 提交信息用英文，遵循 `fix(scope): description` 格式
- 主分支：`main`（远程 `origin` 为 `https://github.com/FauReam/FCL-PRM.git`）
- 论文 PDF 不提交到 GitHub（已忽略 `paper/`）

### 进度条（强制性）
- **所有训练脚本必须显示实时进度条**：tqdm 覆盖 rounds、clients、batches，每层都要有 desc、total、postfix（含 loss/lr/steps/s 等关键指标）
- 日志除进度条外还需输出轮次级摘要：avg_loss, per-domain MSE, round 耗时, 吞吐
- 每 N 个 batch 通过 `log_interval` 刷新 tqdm postfix

### 自动保存（强制性）
- **每 N 轮自动保存 checkpoint**：`save_every` 控制频率
- Checkpoint 命名：`model_m{Milestone}_r{round_num}_c{client_id}.pt`
- 保存前所有 tensor 先 `.cpu()`，保存后 `del` + `gc.collect()` + `torch.cuda.empty_cache()`
- 崩溃恢复：`_find_latest_checkpoint` 自动扫描并 resume
- 最终轮结束后无论 `save_every` 是否命中都必须保存最终模型
