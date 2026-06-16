# FCL-PRM-cdspi 工作清单

> 最后更新: 2026-06-16 | 硬件: GB10 (48SM, 92W, 121GB)

## ✅ 已完成

- [x] 效率修复 (4 bugs + 3 bonus)
  - `base_wrapper.py`: autocast 始终包裹 backbone forward
  - `simulator.py`: 持久化 eval model，消除磁盘 I/O 和 GPU-CPU 搬运
  - `simulator.py`: `torch.no_grad()` → `torch.inference_mode()`
  - `simulator.py`: eval DataLoader `pin_memory=True`
  - `client.py`: 去掉冗余 `.clone()`
  - `run_federated.py`: `cudnn.benchmark=True`
  - 全部 config: `num_workers: 0→4`
- [x] Benchmark 验证 (10.5 samp/s — 纯计算瓶颈，软件已无优化空间)
- [x] `m3_fedavg_head_1.4b_gelu` (2 rounds, killed — null baseline 数据已保存)

## 📋 实验矩阵 (每项 2-round 验证 → 确认无误后 25-round 完整跑)

### Phase 0: 集中式基线 (sanity check)
| # | Config | 类型 | 2-round 预估 | 状态 |
|---|--------|------|-------------|------|
| 1 | `m2_centralized_full_1.4b` | Centralized full-FT | ~3h | ⬜ |
| 2 | `m2_hard_centralized_1.4b` | Centralized hard | ~3h | ⬜ |

### Phase 1: 容量连续谱 (核心实验)
| # | Config | 容量 | 2-round 预估 | 状态 |
|---|--------|------|-------------|------|
| 3 | `m3_fedavg_head_1.4b` | Head-only (ReLU) | ~12h | ⬜ |
| 4 | `m3_fedavg_lora_r8_1.4b` | LoRA r=8 | ~13h | ⬜ |
| 5 | `m3_fedavg_lora_r64_1.4b` | LoRA r=64 | ~13h | ⬜ |
| 6 | `m3_fedavg_lora_r128_1.4b` | LoRA r=128 | ~13h | ⬜ |
| 7 | `m3_fedavg_lora_r256_1.4b` | LoRA r=256 | ~14h | ⬜ |
| 8 | `m3_fedavg_partialft_last2_1.4b` | Partial last-2 | ~14h | ⬜ |
| 9 | `m3_fedavg_partialft_last4_1.4b` | Partial last-4 | ~15h | ⬜ |
| 10 | `m3_fedavg_partialft_last8_1.4b` | Partial last-8 | ~17h | ⬜ |
| 11 | `m3_fedavg_partialft_mlp_1.4b` | Partial MLP | ~15h | ⬜ |
| 12 | `m3_fedavg_partialft_attn_1.4b` | Partial Attn | ~15h | ⬜ |
| 13 | `m3_fedavg_full_1.4b` | **Full-FT** 🔑 | ~20h | ⬜ |

### Phase 2: 架构消融 + 异质性
| # | Config | 变量 | 2-round 预估 | 状态 |
|---|--------|------|-------------|------|
| 14 | `m3_fedavg_head_1.4b_identity` | Identity head | ~12h | ⬜ |
| 15 | `m3_hard_fedavg_1.4b` | Hard heterogeneity | ~20h | ⬜ |
| 16 | `m3_fedavg_partialft_1.4b` | Partial-FT legacy | ~15h | ⬜ |
| 17 | `smoke_versaprm` | Smoke test (200 samp) | ~10min | ⬜ |

### 总工作量估算
```
Phase 0:     2 × 3h  = 6h
Phase 1:    11 × 14h = 154h
Phase 2:     3 × 15h + 10min = 45h
────────────────────────────────────
2-round 验证: ~205h (~8.5 天)
25-round 完整: ~205 × 12.5 ≈ 2560h (~107 天)
```

**策略**: 2-round 验证确认每个 config 正确 → 只在关键 config 上跑 25-round。
关键路径: full-FT (25r) + head-only (25r) + LoRA r=64 (25r) ≈ 150h (6 天)。

## 🔧 已知局限
- GB10 raw BF16 ≈ 14 TFLOPS (受限于 92W 功耗墙)
- ARM64 无 Triton → 无 torch.compile
- 纯计算瓶颈，代码已无优化空间
