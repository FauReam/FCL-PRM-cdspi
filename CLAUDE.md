# FCL-PRM — Claude Code 项目上下文

## 项目简介
Federated Continual Process Reward Model (联邦持续过程奖励模型)。
跨机构联邦学习协同训练 step-level PRM。

**当前方向：联邦全参数微调**（详见 PROJECT_FRAMEWORK.md）：
- 旧方向（Anchor-PRM / permutation rebasin）在 RTX 4070 上得出阴性结论，已停止
- 核心假设：head-only PRM 容量不足是联邦 PRM 的根本瓶颈，
  全参数微调在本设备（NVIDIA GB10，121GB 统一内存）上可首次实现

## 模型选择
- **主实验**: Pythia-2.8B 全参数 FT (~40 GB peak)
- **辅助 1**: Pythia-1.4B 全参数 FT (~21 GB peak, scale trendline)
- **辅助 2**: LLaMA-3.1-8B head-only (~16 GB, cross-architecture validation)

## 常用命令
```bash
# 阶段 1：集中式锚点
python scripts/train_centralized_prm.py --config configs/m2_centralized_full_2.8b.yaml
python scripts/train_centralized_prm.py --config configs/m2_centralized_full_1.4b.yaml

# 阶段 2：联邦全参数微调
python scripts/run_federated.py --config configs/m3_fedavg_full_2.8b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_full_1.4b.yaml
python scripts/run_federated.py --config configs/m3_fedavg_head_llama_8b.yaml

# 附加实验：AttnRes backbone（可选，不影响主线）
python scripts/train_centralized_prm.py --config configs/m2_centralized_full_1.4b_attnres.yaml
python scripts/run_federated.py --config configs/m3_fedavg_full_1.4b_attnres.yaml
```

## 关键路径
- 联邦模拟入口：`scripts/run_federated.py`
- 集中式训练：`scripts/train_centralized_prm.py`
- 客户端训练：`src/fclprm/federated/client.py`
- 服务器聚合：`src/fclprm/federated/server.py` / `aggregators.py`
- 模拟调度：`src/fclprm/federated/simulator.py`
- 模型定义：`src/fclprm/models/base_wrapper.py`
- **AttnRes backbone**：`src/fclprm/models/attnres_backbone.py`
- 配置目录：`configs/`

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
7. **[AttnRes] 仅支持 GPTNeoX（Pythia）架构**：`AttnResBackboneModel` 当前仅支持 GPTNeoX-based 模型（`GPTNeoXForCausalLM`、`GPTNeoXModel`）。LLaMA 等其他架构需扩展 `SUPPORTED_ARCHS`。
8. **[AttnRes] 零初始化必须**：伪查询向量必须初始化为 0（`zero_init=true`），否则训练初期会因非均匀注意力权重导致训练不稳定。不得更改此默认值。
9. **[AttnRes] torch.compile 时机**：`run_federated.py` 中 `torch.compile` 在 `StepRewardModel` 构造之后（而非 backbone 上单独调用），确保 AttnRes 算子也被编译。
10. **[AttnRes] checkpoint 兼容性**：AttnRes 模型有额外的 `pseudo_queries` 和 `key_norm` 参数。标准残差模型的 checkpoint 无法加载到 AttnRes 模型上（state_dict key 不匹配）。

## 工作约定
- 提交信息用英文，遵循 `fix(scope): description` 格式
- 主分支：`main`（远程 `origin` 为 `https://github.com/FauReam/FCL-PRM-fullft.git`）

### 进度条（强制性）
- tqdm 覆盖 rounds / clients / batches，每层有 desc / total / postfix
- 每 N 个 batch 通过 `log_interval` 刷新 postfix

### 自动保存（强制性）
- `save_every` 控制频率，命名 `model_m{M}_r{round}_c{client}.pt`
- 保存前 `.cpu()`，保存后 `del`+`gc.collect()`+`torch.cuda.empty_cache()`
- 最终轮结束后强制保存
