# CD-SPI 检验点方案

> 2026-06-16 | 目标: 在最小计算预算下，用预注册检验点控制论文质量风险。
> 原则: **每个检验点可失败** — 失败即停，不浪费计算。

---

## 总体流水线

```
Gate 0 ──→ Gate 1 ──→ Gate 2 ──→ Gate 3 ──→ Gate 4
(即刻)     (16h)      (+22h)     (+55h)     完成
  │          │           │           │
  └─失败→    └─失败→     └─失败→     └─失败→ KILL + Workshop短文
  修代码     修H1假设    修H2路径    方向错误
```

---

## Gate 0: 代码就绪 (即刻)

| 检查项 | 通过标准 |
|--------|----------|
| `cudnn.benchmark=True` | run_federated.py 第238行已设置 |
| Eval 使用持久化 BF16 模型 | simulator.py `_eval_model` 非空，不触发 HF 重载 |
| `num_workers=4` | 所有 5 个目标 config 已设 |
| 语法检查 | `python -c "import ast; ..."` 5 文件全通过 |

**通过 →** 启动 Gate 1。
**失败 →** 不启动任何实验，先修代码。

---

## Gate 1: H1 确认 — head-only 10-round (~16h 首检, ~80h 完整)

**Config:** `m3_fedavg_head_1.4b`

### 检验点 1a: 2-round 快速筛查 (Round 2 结束时)

| 指标 | 预期 | 实际 | 判定 |
|------|------|------|------|
| CD-SPI sym | < 0.05 | | |
| PCA EVR | < 0.3 | | |
| Permutation test p | > 0.10 | | |
| CKA | > 0.90 | | |
| Loss 下降趋势 | Round 2 < Round 1 | | |

**3 项及以上通过 →** 继续跑完 10 round。
**< 3 项通过 →** H1 假设有误，停止并分析原因。

### 检验点 1b: 10-round 完整 (Round 10 结束时)

| 指标 | 严格阈值 (ICLR级) | 宽松阈值 (Workshop级) |
|------|-------------------|----------------------|
| CD-SPI sym (最终) | < 0.01 | < 0.05 |
| PCA EVR (最终) | < 0.2, interp="zero_variance" | < 0.4 |
| Permutation test p | > 0.20 | > 0.05 |
| Cohen's d vs null | < 0.2 | < 0.5 |
| CD-SPI 随 round 趋势 | 持平或下降 (无积累) | 不明显上升 |
| Round-0 vs Round-10 CD-SPI | 差异 < 0.02 | 差异 < 0.05 |

**严格阈值全部通过 →** H1 强力确认，论文 H1 部分对标 ICLR。
**宽松阈值通过 →** H1 可发表 (Workshop/TMLR)。
**任一不通过 →** 冻结 backbone 竟然产生结构化发散 → 重新审视 CD-SPI 测量协议。**暂停后续实验。**

---

## Gate 2: H2 筛查 — full-FT 2-round (~22h)

**Config:** `m3_fedavg_full_1.4b`

> ⚠️ **这是最早杀门**。如果 full-FT 2-round 就和 head-only 无差异，不必跑 10-round。

| 指标 | 通过 (继续10r) | 警告 (再跑2r) | 杀死 (<2r即停) |
|------|---------------|---------------|----------------|
| CD-SPI sym | ≥ 0.05 | 0.02-0.05 | < 0.02 |
| vs head-only Round2 | CD-SPI 高出 ≥ 3× | 高出 1.5-3× | 高出 < 1.5× |
| PCA EVR | ≥ 0.3 | 0.2-0.3 | < 0.2 |
| Permutation test p | < 0.10 | 0.10-0.20 | > 0.20 |
| Loss vs head-only | 不低于 head-only | — | 明显更差 |

**判定:**
- **≥ 3 项"通过"** → 直接启动 Gate 3 (10-round full-FT)
- **≥ 3 项"警告"** → 再跑 2 round。仍警告 → 降级为 "边界结果"，LoRA r=256 成为主力 H2
- **≥ 2 项"杀死"** → **KILL 方向**，转 Workshop 短文

---

## Gate 3: H2 确认 — full-FT 10-round (+88h from Gate 2 通过)

**Config:** `m3_fedavg_full_1.4b` (继续跑满 10 round)

### 检验点 3a: Round 5 中期杀门

| 指标 | 阈值 | 不满足 → |
|------|------|----------|
| CD-SPI sym | **≥ 0.10** | KILL |
| PCA EVR | **≥ 0.4** | KILL |
| Permutation test p | **≤ 0.10** | KILL |
| CD-SPI 趋势 (r2→r5) | 上升或持平 | 下降 → KILL |

**这是不可协商的硬杀门。** 任一不满足即终止，不等 Round 10。

### 检验点 3b: Round 10 终检

| 指标 | ICLR级 | TMLR级 | Workshop级 |
|------|--------|--------|------------|
| CD-SPI sym | ≥ 0.15 | ≥ 0.10 | ≥ 0.08 |
| PCA EVR | ≥ 0.6 | ≥ 0.4 | ≥ 0.3 |
| Permutation test p | < 0.01 | < 0.05 | < 0.10 |
| Cohen's d | > 2.0 | > 1.0 | > 0.8 |
| CKA (独立验证) | < 0.5 | < 0.7 | < 0.9 |
| CD-SPI 单调递增? | 清晰 | 尚可 | 不要求 |

**ICLR 级全部通过 →** 论文对标 ICLR main track。
**TMLR 级通过 →** 对标 TMLR / FL@ICLR Workshop。
**仅 Workshop 级 →** 对标 Workshop 短文。
**不达 Workshop 级 →** 方向只能产出负结果论文。

---

## Gate 4: 容量连续谱填充 (+170h, 并行化可压缩)

当 Gate 1 (H1) 和 Gate 3 (H2) 均通过后，补充中间点：

| Config | 目的 | 10-round | 检查项 |
|--------|------|----------|--------|
| `lora_r8` | H1 复现: 低秩 LoRA 也是 noise? | ~85h | p > 0.05, EVR < 0.4 |
| `lora_r256` | H2 边界: 高秩 LoRA 过门槛了吗? | ~85h | 两种结果均可发表 |
| `centralized` | 测量管线 sanity | ~3h | CD-SPI < 0.05 |

**Gate 4 不是杀门 —** H1+H2 的二元对比已足够发表，LoRA 端点是有利加分项。

---

## 检验点速查卡

```
检验点          耗时     决策
────────────────────────────────────────
Gate 0          即刻     修/启
Gate 1a (H1 2r)  ~16h    继续80h / 停
Gate 1b (H1 10r) ~80h    H1锁定 / 暂停
Gate 2 (FT 2r)   ~22h    继续110h / 警告 / KILL
Gate 3a (FT 5r)  ~55h    KILL阈值 / 继续
Gate 3b (FT 10r) ~110h   定级: ICLR/TMLR/Workshop
Gate 4 (LoRA)    ~170h   加分 / 跳过
────────────────────────────────────────
总计 (H1+FT)     ~190h (~8天)
总计 (全部5项)   ~363h (~15天)
```

## 并行策略 (缩短 wall-clock)

Gate 1 (head-only 80h) 和 Gate 2/3 (full-FT 132h) **不能并行**（H1 确认是 H2 的前提）。但：

- **理论与实验并行**：Gate 1 跑着的同时起草 Lemma 1+2+3
- **Gate 4 可与写作并行**：H1+H2 结果到手后，写作和 LoRA 跑着同时进行
- **集中式 sanity (3h)** 随时可插空跑

## 失败协议

如果 Gate 3a (Round 5 CD-SPI < 0.10) 触发 KILL：

1. 保留 head-only H1 数据 (已有)
2. 保留 full-FT 1-5 round 数据
3. 写 4 页 Workshop 短文：**"Head-only federated PRM produces zero structured cross-client divergence: a CD-SPI null baseline"**
4. 归档代码，转移研究方向
5. 预计总浪费计算: ~135h (~5.6 天)，在可接受范围
