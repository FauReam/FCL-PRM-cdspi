# 会话经验笔记

> 从实际调试中积累的关键信息。新 session 读此文件可跳过踩坑。

> **⚠️ F0 预案激活中**：CD-SPI sym 观测值 0.0011–0.0016（全端点一致）。若全容量谱复现 → Plan N（Null Result 叙事，详见 PROJECT_FRAMEWORK.md §六-B 和 EXPERIMENT_PLAN.md §Plan N）。不要当失败处理——这是更强的结果。

## 硬件

| 项目 | 值 |
|------|-----|
| 机器 | NVIDIA DGX Spark (GB10) |
| GPU | Blackwell, compute 12.1, CUDA 13.0 |
| 内存 | 121 GB 统一内存 (CPU+GPU 共享) |
| CPU | ARM64 (Cortex-X925 + A725) |
| torch.compile | **不可用** (Triton 不兼容 ARM64) |
| 实测速度 | Pythia-1.4B frozen backbone, head-only, batch=128, max_len=256 → **~8s/batch** |

## 关键 Bug 及修复

每次都是真实踩过的，不要复现：

1. **`torch.cuda.synchronize()` 每 batch 调用** — `client.py:168`，强制 GPU 同步，删掉后提速 ~2x
2. **`loss.item()` 调 3 次** — 每次触发 CPU-GPU 传输，合并为 1 次
3. **frozen backbone 走完整 autograd** — `base_wrapper.py` forward 中加 `torch.no_grad()`，省 ~30% 时间
4. **bf16 backbone output → fp32 head crash** — `mat1 and mat2 must have the same dtype`，forward 中 `last_hidden.float()` 解决
5. **YAML `1e-4` 被解析为字符串 `"1e-4"`** — PyYAML 1.1 含混，必须写成 `1.0e-4` 或 `0.0001`。已在 `client.py` 入口加 `float()` 防御
6. **stale checkpoint 导致空跑** — 失败运行残留 `r25` checkpoint，恢复逻辑读到 `start_round=25` 直接跳过全部训练。已加 `>= num_rounds` 检测并自动删除
7. **CD-SPI 静默返回空** — anchor steps 只在 `aggregation: anchor_prm` 时生成。已改为 `needs_anchor_steps = anchor_prm or compute_cd_spi or compute_sym_cd_spi`

## 数据

### VersaPRM

```
84,098 CoT 样本 → 669,218 个 step
平均 question+step: 217 tokens, 中位数 185
max_length 分布:
  192: 52.9%    256: 73.5%    384: 91.3%    512: 97.1%
```

- **净化数据**: `data/versaprm/versa_prm.jsonl`（原始备份 `versa_prm_orig.jsonl`）
- **净化方式**: tokenize 后丢弃 >384 token 的 step（5.8% 丢弃），不截断
- **训练用 max_length=384**：覆盖 91.3%，不引入截断过拟合

### 性能权衡

| 改法 | 效果 | 风险 | 决策 |
|------|------|------|------|
| max_length 256→128 | ~3x 加速 | **截断 78% 数据** | ❌ 已否决 |
| max_length 256→384 | ~1.3x 变慢 | 丢弃 8.7%，无截断 | ✅ 当前方案 |
| batch_size 4→128 | ~32x 加速 | 仅 head-only/LoRA 安全 | ✅ 已改 |
| batch_size 4 (保留) | 安全 | 慢 | partial-FT / full-FT 保留 |

## 配置速查

### Phase 1 配置（全部 ready）

| 配置 | 类型 | batch | CD-SPI |
|------|------|-------|--------|
| `m3_fedavg_head_1.4b.yaml` | head-only ReLU | 128 | ✅ |
| `m3_fedavg_head_1.4b_gelu.yaml` | head-only GELU | 128 | ✅ |
| `m3_fedavg_head_1.4b_identity.yaml` | head-only Identity | 128 | ✅ |
| `m3_fedavg_lora_r{8,64,128,256}_1.4b.yaml` | LoRA | 128 | ✅ |
| `m3_fedavg_partialft_1.4b.yaml` | last 2 layers | 4 | ✅ |
| `m3_fedavg_partialft_last{4,8}_1.4b.yaml` | more layers | 4 | ✅ |
| `m3_fedavg_partialft_mlp_1.4b.yaml` | MLP-only | 4 | ✅ |
| `m3_fedavg_partialft_attn_1.4b.yaml` | Attn-only | 4 | ✅ |
| `m3_fedavg_full_1.4b.yaml` | full FT | 4 | ✅ |
| `m2_centralized_full_1.4b.yaml` | centralized anchor | 4 | - |

所有 federated 配置统一：samples_per_client=5000, max_length=384, rounds=25, eval_every=1, CD-SPI+symmetrical ON

### Smoke test

| 配置 | batch | rounds | 数据 | 时间 |
|------|-------|--------|------|------|
| `smoke_versaprm.yaml` | 128 | 2 | 200 CoTs/域 | 34 min |

## 启动命令格式

项目约定：终端命令用 `\` 折行，每行 ≤50 字符。

```bash
cd /home/jiayu/FCL-PRM && source venv/bin/activate

# 试跑 (2 rounds)
python scripts/run_federated.py \
  --config configs/m3_fedavg_head_1.4b_gelu.yaml \
  --rounds 2

# 完整跑 (25 rounds, 后台)
nohup python scripts/run_federated.py \
  --config configs/m3_fedavg_head_1.4b_gelu.yaml &
tail -f nohup.out
```

## 时间预估 (head-only, batch=128, max_len=384)

| Client | Steps | Batches | ~12s/batch |
|--------|-------|---------|------------|
| math | ~50K | ~390 | ~1.3 h |
| code | ~74K | ~581 | ~1.9 h |
| medical | ~31K | ~240 | ~0.8 h |
| general | ~47K | ~367 | ~1.2 h |
| **每 round** | | | **~5.2 h** |
| **2 rounds** | | | **~10 h** |

## 项目规范文件

| 文件 | 内容 |
|------|------|
| [CLAUDE.md](CLAUDE.md) | 项目上下文、常用命令、工作约定 |
| [TRAINING_CONVENTIONS.md](TRAINING_CONVENTIONS.md) | 训练代码规范 5 条 (含终端存活) |
| [PROJECT_FRAMEWORK.md](PROJECT_FRAMEWORK.md) | CD-SPI 框架、实验规划、投稿策略 |
