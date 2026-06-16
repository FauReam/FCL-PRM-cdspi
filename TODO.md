# FCL-PRM-cdspi 工作清单

> 更新: 2026-06-16 19:50 | 5 configs × 5 rounds 批量启动中

## 🔄 进行中: 5-round 趋势扫描

| # | Config | Rounds | 预估 | 状态 |
|---|--------|--------|------|------|
| 1 | `m3_fedavg_head_1.4b` | 5 | ~42h | 🔄 |
| 2 | `m3_fedavg_full_1.4b` | 5 | ~60h | ⬜ |
| 3 | `m3_fedavg_lora_r8_1.4b` | 5 | ~45h | ⬜ |
| 4 | `m3_fedavg_lora_r256_1.4b` | 5 | ~45h | ⬜ |
| 5 | `m2_centralized_full_1.4b` | 5 | ~8h | ⬜ |

**预计总耗时: ~200h (~8.3 天)** — 关闭终端继续运行

```
head-only ──→ full-FT ──→ lora_r8 ──→ lora_r256 ──→ centralized
  ~42h         ~60h        ~45h        ~45h           ~8h
```

## 🔍 每轮 Round 5 检验点

| Config | 检查指标 | 阈值 |
|--------|----------|------|
| head-only | CD-SPI sym | < 0.05 (应为 noise) |
| full-FT | CD-SPI sym | ≥ 0.05 (需见上升趋势) |
| lora_r8 | CD-SPI sym | < 0.05 (应与 head-only 一致) |
| lora_r256 | CD-SPI sym | 观察是否开始上升 |
| centralized | CD-SPI sym | < 0.05 (sanity check) |

## 📊 监控命令
```bash
# 查看批次进度
tail -f experiments/batch_*.log

# 查看当前 config 的实时输出
tail -f experiments/m3_fedavg_head_1.4b_5r/run_*.log

# 快速 CD-SPI 摘要
grep "CD-SPI sym\|PCA EVR\|CKA" experiments/*_5r/run_*.log
```

## ✅ 已完成
- 效率修复 (7项)
- 专家 Panel 2 轮 (4.9/10 → PIVOT)
- 裁撤 12 冗余 config
- Benchmark (10.5 samp/s)
- 检验点方案 (CHECKPOINT_PLAN.md)
