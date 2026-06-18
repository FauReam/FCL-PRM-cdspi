# FCL-PRM-cdspi · 实验计划

> **更新**：2026-06-16
> **硬件**：DGX Spark GB10 (121GB 统一内存, ARM64)
> **目标**：ICLR 2027（预计 2026 年 10 月截稿）

---

## 核心假设（待验证）

| 假说 | 内容 | 验证方式 |
|:---:|:---|:---|
| **H1** | 低容量配置 (head-only) 的跨客户端差异不可与噪声区分 | 排列检验 p > 0.05 + EVR < 0.4 |
| **H2** | 全参数 FT 呈现统计显著的结构化模式 | 排列检验 p < 0.05 + EVR > 0.6 |
| **H3** | 存在容量转变阈值 (LoRA rank 精细网格) | EVR vs log(rank) 分段线性回归 |
| **H4** | 发散结构化程度与聚合性能有因果关系 | CD-SPI 正则化干预实验 (Phase 2) |

---

## Phase 0-Verify：二元极值验证（优先执行，~24h 出结论）

### 目的

在全量 Phase 0 扫 9 个 config 之前，先用**容量谱两端**的极值点快速验证核心命题：

> 低容量 = CD-SPI 噪声 (EVR ≈ 0)，高容量 = CD-SPI 结构化信号 (EVR > 0)

如果这个二元对比都看不到信号差异，那全量扫 9 个 config 也是浪费时间。

### 实验设计

| 角色 | 配置 | batch | rounds | 预计耗时 | 要验证的命题 |
|:---|:---|:---:|:---:|:---:|:---|
| 🔵 低容量锚点 | `m3_fedavg_head_1.4b_gelu` (GELU) | 128 | ✅ **已完成 R1-2** | 0h | H1: CD-SPI≈0, 噪声 |
| 🔵 低容量复证 | `m3_fedavg_head_1.4b` (ReLU) | 128 | 5 | ~11h | H1 跨激活函数一致性 |
| 🔴 高容量探针 | `m3_fedavg_lora_r256_1.4b` | 128 | 5 | ~12.5h | H2: CD-SPI>0, 结构信号 |

> **为什么选 LoRA r=256 而不是 full FT？** LoRA r=256 是 batch=128 组里容量最高的配置，12.5h 出 5 轮数据，符合"一天内验证"原则。full FT 跑 5 轮需要 6 天，留给 Phase 0 主阶段。

### 判断标准

| 结果 | GELU CD-SPI sym | LoRA r=256 CD-SPI sym | 判定 | 行动 |
|:---|:---:|:---:|:---|:---|
| 🟢 阳性 | < 0.01, EVR ≈ 0 | > 0.01, EVR > 0.3 | 容量-发散因果链成立 | 进入 Phase 0 全量扫 |
| 🟡 弱阳性 | < 0.01, EVR ≈ 0 | 0.005-0.01, EVR 0.1-0.3 | 趋势存在但弱 | 增加 LoRA r=64 对比后再决定 |
| 🔴 阴性 | < 0.01 | < 0.01, EVR ≈ 0 | 容量不影响发散类型 | **触发 Plan B** |

### 已有数据：GELU Round 1 完整诊断

```
CD-SPI (symmetrical backbone):  0.0011  ← 几乎为零！
PCA EVR:                        0.0000  ← 零方差，纯噪声
CKA:                            1.0000  ← 跨客户端完美一致
Function divergence (cos):      0.0581  ← 函数空间差异也极小
Function divergence (JS):       0.1387
```

> **解读**：frozen backbone + head-only 训练后，4 个客户端从 backbone 倒数第二层提取的 hidden state **完全一致**——无论 client 是 math/code/medical/general，Pythia-1.4B 对同一步骤的表示没有任何差异。这正是 H1 预测的"噪声型发散"（其实是零发散）。联邦聚合在这种情况下不产生有意义的信息，FedAvg 只是在平均完全相同的参数更新。

### 执行命令

```bash
cd /home/jiayu/FCL-PRM && source venv/bin/activate

# Step 1: head-only ReLU 5r（低容量复证）
nohup python scripts/run_federated.py \
  --config configs/m3_fedavg_head_1.4b.yaml \
  --rounds 5 &

# Step 2: LoRA r=256 5r（高容量探针）
nohup python scripts/run_federated.py \
  --config configs/m3_fedavg_lora_r256_1.4b.yaml \
  --rounds 5 &
```

### 产出

- `reports/VERIFICATION_REPORT.html` — 含 GELU 已有数据 + ReLU + LoRA r=256 对比
- 上传到 GitHub，作为 Phase 0 的 go/no-go 决策依据

---

## 阶段总览

```
Phase 0: 全量 5 轮筛选 ─── 所有 config 一视同仁
         输出：9 点 CD-SPI 热力图 + MSE 散点图
         ↓
决策点:  看 CD-SPI 在容量谱上是否单调
         ├─ 是 → 选 3 个关键点续跑 Phase 1
         └─ 否 → 触发 Plan B，不续跑
         ↓
Phase 1: 关键点续跑 25 轮 ─── 深度统计验证
         输出：论文核心图表
         ↓
Phase 2: 补充实验 ─── 消融 + 交叉验证 + OOD
```

---

## Phase 0：全量 5 轮筛选（预计 ~13 天）

### 目标

拿到所有配置在第 5 轮的 CD-SPI + per-domain MSE + 排列检验 p 值。
一张散点图（x=容量等级, y=CD-SPI 结构化程度）决定论文生死。

### 配置清单

#### 第一梯队：高吞吐组 (batch=128, 单 config ≤ 0.5 天)

| # | 配置 | 容量 | batch | 单轮 | 5 轮总计 | 要验证的命题 |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 1 | `m3_fedavg_head_1.4b` (ReLU) | 最低 | 128 | ~2.2h | **~11h** | H1 主证据 |
| 2 | `m3_fedavg_head_1.4b_gelu` (GELU) | 最低 | 128 | ~2.2h | ✅ **已完成 10r** | 架构消融：激活函数无关 |
| 3 | `m3_fedavg_head_1.4b_identity` | 最低 | 128 | ~2.2h | **~11h** | 架构消融：线性 head 对照 |
| 4 | `m3_fedavg_lora_r8_1.4b` | 低 | 128 | ~2.4h | **~12h** | 容量连续谱起点 |
| 5 | `m3_fedavg_lora_r64_1.4b` | 中 | 128 | ~2.4h | **~12h** | 终止锚点：若接近 full FT → F2 触发 |
| 6 | `m3_fedavg_lora_r256_1.4b` | 中高 | 128 | ~2.5h | **~12.5h** | 容量转变阈值探测 |

**第一梯队合计**：~3.5 天（含 GELU 10r 已完成）

#### 第二梯队：低吞吐组 (batch=4, 单 config 3-6 天)

| # | 配置 | 容量 | batch | 单轮 | 5 轮总计 | 要验证的命题 |
|:---:|:---|:---:|:---:|:---:|:---:|:---|
| 7 | `m3_fedavg_partialft_1.4b` (last2) | 中高 | 4 | ~15h | **~3.1 天** | 容量谱桥接：LoRA → full FT 之间 |
| 8 | `m3_fedavg_full_1.4b` | 最高 | 4 | ~29h | **~6.0 天** | H2 主证据：容量谱最右端 |
| 9 | `m2_centralized_full_1.4b` (1 epoch) | 天花板 | 4 | — | **~2-3 天** | F5 验证：集中式上下限 |

**第二梯队合计**：~11-12 天

> **M2 集中式说明**：只跑 1 epoch，数据量降到 20K CoTs（而非全量 84K），目的仅验证 full FT 在集中式场景的上限是否足够拉开 head-only。不跑满是因为 3 epochs 需 ~17 天，不符合"单 config 一天"原则。

### 决策点（Phase 0 完成后，半天分析）

在全部 9 个配置的 5 轮数据上画出：

```
        CD-SPI p-value (排列检验)
        ↑
  0.05  ┼ - - - - - - - - - - - 噪声线
        │  head ●
        │  LoRA r=8 ●
        │  LoRA r=64 ●
        │                     LoRA r=256 ●
        │                          partial ●
        │                              full FT ●
        └──────────────────────────────→ 容量等级
```

**判断矩阵**：

| 观测结果 | 判定 | 行动 |
|:---|:---|:---|
| CD-SPI 沿容量单调上升，full FT p < 0.05 | 🟢 H1/H2 初步成立 | 续跑 Phase 1 |
| LoRA r=64 已达 full FT 95%+ | 🟡 F2 可能触发 | 重点对比 r=64 vs full |
| CD-SPI 全员 p > 0.05 | 🔴 H1/H2 被证伪 | 触发 Plan B |
| 集中式 head vs full 差距 < 2% | 🟠 F5 触发 | 联邦特异性 claim 不成立 |
| CKA/JS 与 CD-SPI 不一致 | 🟠 F4 触发 | 交叉验证失败，降级 CD-SPI |

---

## Phase 1：关键点续跑至 25 轮（预计 ~5-7 天）

### 选点策略

根据 Phase 0 的结果，选 **3 个关键配置** 从第 6 轮续跑到 25 轮：

| 优先级 | 配置 | 理由 | 续跑量 |
|:---:|:---|:---|:---|
| 1 | `head-only ReLU` | H1 的统计显著性需要足够轮次 | 6→25r (+20r = ~44h) |
| 2 | `full FT 1.4B` | H2 的主证据，效应量估计需要更多轮 | 6→25r (+20r = ~580h) |
| 3 | 容量转变点附近 1 个 | Phase 0 确定（可能是 LoRA r=256 或 partial last2） | 6→25r |

> **full FT 续跑 20 轮需 ~24 天**，这是整个计划中最大的时间块。如果 Phase 0 已经看到清晰的信号（p < 0.01, EVR > 0.6），可以只续到 15 轮而非 25 轮，省 ~12 天。

### Phase 1 产出

- 3 条完整的 CD-SPI 轮次曲线（含置信区间）
- 排列检验随轮次变化（证明信号累积）
- CD-SPI vs MSE correlation scatter（H4 初步证据）
- 论文 Figure 2-4 的原始数据

---

## Phase 2：补充实验（论文收尾阶段）

| # | 实验 | 配置 | 时间 | 优先级 |
|:---:|:---|:---|:---:|:---:|
| P2-1 | CKA 交叉验证 | 用 Phase 0 已有 checkpoints 计算 | 0 天 | P0 |
| P2-2 | OOD 跨域测试 | head-only + full FT, --ood flag | ~2 天 | P1 |
| P2-3 | 标签扰动消融 | head-only + full FT, --label-noise | ~2 天 | P1 |
| P2-4 | CD-SPI 噪声注入校准 | 合成数据，已知 ground truth | ~1 天 | P1 |

---

## 砍掉的实验（及理由）

| 配置 | 理由 |
|:---|:---|
| `partialft_last4_1.4b` | last2 已经桥接 LoRA→full，中间插值信息增量低 |
| `partialft_last8_1.4b` | 同上 |
| `partialft_attn_1.4b` | 架构消融放 rebuttal/camera-ready |
| `partialft_mlp_1.4b` | 同上 |
| `lora_r128_1.4b` | r=8/64/256 三个点已覆盖低中高，r=128 冗余 |
| M2 3 epochs | 420h 换 0.1% 精度提升，完全不合理 |

---

## 终止锚点（任何阶段触发立即停止）

| 锚点 | 触发条件 | 行动 |
|:---|:---|:---|
| **F2** | LoRA(r=64) 达 full FT 98%+ 且 CD-SPI 为噪声 | 论文转向 CD-SPI 纯诊断工具，不再宣称容量优势 |
| **F5** | 集中式 head-only vs full FT 差距 ≤ 2% | 联邦特异性 claim 降级 |
| **F4** | CKA/JS 与 CD-SPI 排序不一致 | CD-SPI 降级为辅助指标 |
| **整体** | Phase 0 全 9 点 CD-SPI 无单调趋势 | 论文转为"CD-SPI 揭示联邦 PRM 聚合无容量依赖" → **[Plan N](#plan-n-全容量谱-null-result-执行预案)** |
| **F0** | 全容量谱 CD-SPI sym 一致 ≈ 0.001（full FT = head-only） | **[Plan N](#plan-n-全容量谱-null-result-执行预案)**：Null Result 叙事，顶刊硬通货 |

---

## 执行顺序（一次只跑一个）

```
✅ 1. head-only GELU 10r            已完成 (2026-06-16)
── 2. head-only ReLU 5r             ~11h     一天内完成
   3. head-only Identity 5r          ~11h     一天内完成
   4. LoRA r=8 5r                    ~12h     一天内完成
   5. LoRA r=64 5r                   ~12h     一天内完成
   6. LoRA r=256 5r                  ~12.5h   一天内完成
── 第一梯队完成 (累计 ~3.5 天) ──
   7. partial FT last2 5r            ~3.1 天
   8. full FT 1.4B 5r                ~6.0 天
   9. M2 centralized 1ep             ~2-3 天
── Phase 0 完成 ──
  10. 数据分析 + 决策                半天
── Phase 1 启动 ──
  11. head-only ReLU 续跑 6→25r
  12. full FT 续跑 6→25r
  13. 转变点续跑 6→25r
── Phase 1 完成 ──
  14. Phase 2 补充实验
```

---

## Plan N：全容量谱 Null Result 执行预案

> **触发**：Phase 0 全 9 配置 CD-SPI sym 一致 ≈ 0.001–0.002，full FT 与 head-only 无差异。
> **本质**：容量不放大客户端表征发散。这不是失败——这是更值钱的发现。

### 为什么 Null Result 更强

| 阳性 | Null |
|:---|:---|
| "容量越大 drift 越大"（直觉） | "容量不影响 drift"（反直觉） |
| 谁先做谁发 | **谁先系统证明 null 谁定义基线** |
| CD-SPI 作为发散度量 | CD-SPI 作为证伪工具 |
| 实践："用 full FT" | 实践："**head-only 就够了**" |

### F0 触发后立即执行的 4 步

**Step 1 — 预注册等价性边界（跑完 Phase 0 当天）**
```
在 arXiv 预印本 / project page 上声明：
"若全容量谱 CD-SPI sym < 0.01 且 TOST 等价性检验显著，
 则接受 null hypothesis：Pythia-1.4B 联邦 PRM 训练中，
 客户端表征发散不依赖可训练容量。"
```
这防止审稿人说"你们看到 null 之后才改假说"。

**Step 2 — 三方交叉验证（1 天计算）**
- CKA（已观测 1.0）：特征空间一致性
- JS 散度：函数空间一致性
- 等价性检验 (TOST)：统计证明"无差异"而非"未检测到差异"

**Step 3 — 正控制实验（~1 天）**
- 合成已知 ground truth 发散的数据
- 验证 CD-SPI 在该数据上正确检测到信号
- → 证明工具本身有效，不是灵敏度不够

**Step 4 — 论文叙事重构（~1 周）**
- 标题方向：*"Where Client Drift Should Be But Isn't"*
- 核心图：CD-SPI sym 容量平线图 + TOST 等价性边界 + 正控制对比
- 攻击面预防御：§6-B 全部事项（见 PROJECT_FRAMEWORK.md）

### F0 触发后砍掉的实验

| 砍掉 | 理由 |
|:---|:---|
| Phase 1 续跑 25 轮（全部） | null result 不需要深度轮次——信号已在 5 轮内稳定 |
| Phase 2 OOD + 异质性消融 | 留给 rebuttal / follow-up |
| 架构消融 full set (Identity) | GELU+ReLU 两者一致已足够 |

**F0 触发后从触发到投稿的时间线：~2 周**（vs 阳性叙事 ~10 周）

### 当前观测 (2026-06-18)

| 配置 | CD-SPI sym | 状态 |
|:---|:---|:---|
| M3 head-only GELU 10r | 0.0011 | ✅ |
| M3 head-only ReLU 2r | 0.0011 | ✅ |
| Feasibility full FT 2r | 0.0016 | ✅ |
| M2 centralized | — | 🔄 运行中 |
| 中间容量点 (LoRA/partial) | — | ⏳ 待跑 |

> **趋势**：两个端点 + 两个激活函数 的 CD-SPI sym 一致在 0.001–0.002。如果接下来的 LoRA r=256 和 full FT 联邦也在这个量级，F0 触发基本确认。

---

## 当前状态

| 项目 | 值 |
|:---|:---|
| 当前运行 | `m2_centralized_full_1.4b.yaml` (Epoch 2/3) |
| 启动时间 | 2026-06-18 16:38 |
| 预计完成 | 2026-06-18 ~22:00 |
| 下一步 | M3 full FT 联邦 2r 快速验证 F0 |
