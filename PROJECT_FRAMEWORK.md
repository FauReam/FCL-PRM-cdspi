# FCL-PRM：CD-SPI 发散结构诊断框架 —— 项目框架

> **更新**：2026-06-11 · 基于两次 Adversarial Panel 结论重构
> **方向迁移**：从「容量叙事」（full FT vs head-only）→「CD-SPI 噪声/结构化发散诊断框架」

---

## 一、核心叙事

> 本文提出 **CD-SPI（Client Divergence Signal-Noise Partition Index）诊断框架**——一个结构化的两阶段统计诊断协议。
>
> **核心发现**（待实验验证）：在 dense backbone + FedAvg 基线下，低容量配置（head-only / LoRA r<256）的跨客户端参数差异本质上是**噪声**（不可与随机排列区分），而全参数 FT 产生的参数差异具有**结构化信号**（统计显著且与聚合性能相关）。
>
> **本文第一贡献是 CD-SPI 诊断框架本身**——它从发散类型的维度（而非发散幅度）揭示了一个被所有相关工作（CFL 2020, FedCDD 2025, OvA-LP 等）忽略的信号结构视角。容量（head-only / LoRA / full FT）仅作为**实验变量**，而非核心发现。

### 与旧叙事的区别

| 维度 | 旧叙事（已废弃） | 新叙事（当前） |
|:---|---|:---|
| **第一贡献** | full FT 的性能优势 | **CD-SPI 诊断框架**（噪声 vs 结构化发散） |
| **容量定位** | 核心发现 | **实验变量**，用于操控发散类型 |
| **核心问题** | "full FT 是否更好？" | "发散何时是噪声、何时是结构？这对聚合意味着什么？" |
| **惊讶元素** | 直觉性（容量越大越好） | 反直觉潜力（发散类型比容量大小更根本地决定聚合有效性） |
| **CD-SPI** | 验证工具 | **主要贡献**（虽算子非新，但分析维度新） |

---

## 二、理论框架：CD-SPI 诊断协议

### 2.1 CD-SPI 定义

```
CD-SPI(s) = 1 - mean_{i,j} cos(h_i(s), h_j(s))

其中 h_i(s) 是 client i 对 step s 在 backbone 倒数第二层的 hidden state
```

CD-SPI 不是新的相似度度量。余弦相似度在 FL 中已被广泛使用：
- **CFL (2020)**：使用成对余弦相似度分离定理做客户端聚类
- **FedCDD (ICLR 2025)**：使用余弦相似度矩阵诊断 FedLLM 发散

**CD-SPI 的新颖性在于分析框架**：前人用余弦相似度做**聚类**或报告**发散程度**，本文首次将其系统化为**区分发散类型（噪声 vs 结构化）的诊断协议**，并为发散类型与聚合有效性之间的因果关系提供系统的统计验证方案。

### 2.2 两阶段诊断协议

**阶段 1 — 排列检验（Permutation Test）**
- H0：观测到的 CD-SPI 与随机分配嵌入一致
- 三层递进 null distribution：
  - (a) 基准层：客户端标签置换
  - (b) 中间层：容量匹配的独立训练基线
  - (c) 最强层：随机特征映射替换
- 效应量为主：Cohen's d，p 值为辅

**阶段 2 — 主成分解释方差比（PCA EVR）**
- 对嵌入矩阵做 PCA，计算第一主成分的解释方差比
- EVR > 0.6 → 发散具有结构化（低维流形）
- EVR < 0.4 → 发散近噪声（高维球面）
- 与排列检验结果交叉验证

### 2.3 独立交叉验证

CD-SPI 作为主要诊断工具，其结论必须被至少一个独立指标重申：

| 指标 | 测量空间 | 与 CD-SPI 的互补性 |
|:---|:---|:---|
| **CKA** | 特征空间（Centered Kernel Alignment） | 对非线性变换鲁棒，不受测量不对称性影响 |
| **JS 散度** | 输出空间（reward 分布） | 函数空间验证，与参数空间正交 |
| **余弦/JS 对比** | 参数 vs 函数空间 | 分辨"语义发散"与"过拟合噪声" |

---

## 三、核心断言框架

### 3.1 实验性假设（H1-H4）

| 假说 | 描述 | 验证方式 | 优先级 |
|:---:|---|:---:|:---:|
| **H1** | 低容量配置的跨客户端差异不可与噪声区分 | 排列检验 p>0.05 + EVR<0.4 | P0 |
| **H2** | 全参数 FT 呈现统计显著的结构化模式 | 排列检验 p<0.05 + EVR>0.6 + 性能正相关 | P0 |
| **H3** | 存在容量转变阈值（LoRA rank 精细网格扫描） | 发散类型突变点的 rank 定位 | P1 |
| **H4** | 发散结构化程度与聚合性能有因果关系 | CD-SPI 正则化干预实验 | P2 |

### 3.2 证伪条件（F1-F5）

| 条件 | 描述 | 触发行动 |
|:---:|---|:---:|
| **F1** | head 维度 1024 显示结构化模式 | 修正 H1，增容后 head-only 可能有结构 |
| **F2** | LoRA(r=64) 达 full FT 95%+ 且 CD-SPI 为噪声 | H2 被证伪 → Plan B |
| **F3** | 仅定量差异无定性差异 | 二分法→连续谱叙事，降级论文贡献 |
| **F4** | CKA/JS 与 CD-SPI 不一致 | 放弃 CD-SPI 作为主要证据 |
| **F5** | 集中式差距 ≤ 2% | 联邦特异性 claim 不成立 |
| **F0** | 全容量谱 CD-SPI sym 一致 ≈ 0.001（full FT = head-only） | 🟢 **Plan N（Null Result）**：论文转向"CD-SPI 揭示联邦 PRM 无容量依赖发散" |

---

## 四、实验规划

### Phase 0 — 关键路径（2 周，决定论文生死）

| # | 实验 | 配置 | 关键输出 |
|:---:|---|:---:|:---:|
| P0-1 | 集中式基线 + 对称 CD-SPI | M2 centralized full FT 1.4B | 天花板验证 + 对称/非对称 CD-SPI 对比 |
| P0-2 | 容量连续谱 | LoRA r=8/64/128/256 + partial FT + head-only + full FT | 容量-发散-性能三元关系散点图 |
| P0-3 | 训练前 CD-SPI 基线 | 第 0 轮测量 | 控制初始化混淆 |

**终止锚点 P0 完成时判断：**
- 若 LoRA(r=64) 达 full FT 95%+ → F2 触发的可能性高，准备 Plan B
- 若集中式 head-only 与 full FT 差距 ≤ 2% → F5 触发
- 若全部正常 → 进入 Phase 1

### Phase 1 — 控制实验（3 周，验证结论鲁棒性）

| # | 实验 | 配置 | 关键输出 |
|:---:|---|:---:|:---:|
| P1-1 | 架构消融 | ReLU/GELU/Identity head | CD-SPI 排序一致性 |
| P1-2 | 三层排列检验 | 三层 null distribution | H1/H2 统计显著性 |
| P1-3 | CKA 交叉验证 | 全部容量配置 | 独立指标验证矩阵 |
| P1-4 | 合成数据校准 | 已知 ground truth 数据 | CD-SPI 精确率+召回率 |

### Phase 2 — 深度验证（3 周，论文深度）

| # | 实验 | 关键方法 |
|:---:|---|:---:|
| P2-1 | 散度指向性区分 | CD-SPI vs Wasserstein 距离对比 |
| P2-2 | 独立聚合指标 | FedDYN drift / SCAFFOLD 控制变分 |
| P2-3 | 反直觉条件测试 | 极端异质性下低容量结构化 vs 高容量噪声 |
| P2-4 | OOD + 异质性消融 | 跨域 OOD + Dirichlet/label_shift/mixed |
| P2-5 | CD-SPI 理论形式化 | 有界性定理 + 尺度不变性 + 收敛率联系 |

### 实验配置总览

| 配置名 | 模型类型 | 容量等级 | 阶段 |
|:---|---|---|:---:|
| `m2_centralized_full_1.4b` | full FT（集中式锚点） | 最高 | P0 |
| `m3_fedavg_head_1.4b` | head-only（256-dim MLP） | 最低 | P0 |
| `m3_fedavg_lora_r8_1.4b` | LoRA(r=8) | 低 | P0 |
| `m3_fedavg_lora_r64_1.4b` | LoRA(r=64) | 中低 | P0 |
| `m3_fedavg_lora_r128_1.4b` | LoRA(r=128) | 中 | P1 |
| `m3_fedavg_lora_r256_1.4b` | LoRA(r=256) | 中高 | P0 |
| `m3_fedavg_partialft_last2_1.4b` | partial FT（last 2） | 中高 | P0 |
| `m3_fedavg_partialft_last4_1.4b` | partial FT（last 4） | 高 | P0 |
| `m3_fedavg_partialft_last8_1.4b` | partial FT（last 8） | 高 | P0 |
| `m3_fedavg_partialft_mlp_1.4b` | partial FT（MLP only） | 中 | P1 |
| `m3_fedavg_partialft_attn_1.4b` | partial FT（Attn only） | 中 | P1 |
| `m3_fedavg_full_1.4b` | full FT | 最高 | P0 |
| `m3_fedavg_head_1.4b_relu` | head-only ReLU | 最低 | P1 |
| `m3_fedavg_head_1.4b_gelu` | head-only GELU | 最低 | P1 |
| `m3_fedavg_head_1.4b_identity` | head-only Identity | 最低 | P1 |

---

## 五、方法与架构

### 5.1 对称化测量（P0* 控制实验）

**问题**：当前 `_eval_cd_spi` 使用 `get_head_embedding()`，对 head-only 提取的是 head 层特征（含随机初始化权重），对 full FT 提取的是 backbone + head 的全参数特征。两种配置的**嵌入空间不对等**。

**修复**：新增 `get_backbone_embedding()`，从 backbone 倒数第二层 transformer 层后统一提取 hidden state。强制所有 CD-SPI 测量使用对称化嵌入。同时报告对称和非对称两种测量结果，供读者判断差异来源。

### 5.2 CD-SPI = permutation test + PCA EVR 双重标准

**不再使用原始 CD-SPI 作为评分指标**。改为两阶段诊断：

1. **排列检验**：计算观测 CD-SPI 在 null distribution 中的位置
2. **PCA EVR**：主成分解释方差比，补充发散的"结构维度"信息

### 5.3 架构无关性验证

三种 head 激活函数（ReLU, GELU, Identity），验证 CD-SPI 对 head 架构不敏感。

---

## 六、局限性声明

本文因果推断限于 **dense backbone (Pythia 系列) + FedAvg 聚合 + 标准异质性设定**。以下外推性需独立验证：

1. 非 FedAvg 聚合方法（FedProx, SCAFFOLD, FedDYN）
2. 不同模型架构（LLaMA, Qwen, Mistral）
3. 极端异质性（非 IID 程度超出实验设定）
4. PM（policy model）而非 PRM 场景

CD-SPI 作为诊断工具，其本身存在**测量不对称性、架构敏感性等已知局限**（详见本文方法论章节）。所有结论应结合交叉验证指标综合解读。

---

## 六-B：Plan N —— 全容量谱 Null Result 叙事预案

> **触发条件 F0**：全容量谱（head-only → LoRA r=8/64/256 → partial FT → full FT）CD-SPI sym 一致 ≈ 0.001–0.002，容量不影响客户端发散类型。
>
> **当前观测**：head-only 和 feasibility full FT 的 CD-SPI sym 已在该量级（0.0011/0.0016），中间容量点待验证。

### 为什么 Null Result 比阳性结果更强

| 维度 | 阳性叙事 (full FT > head) | Null 叙事 (容量无关) |
|:---|:---|:---|
| 惊奇度 | 直觉性 | **反直觉** — 全参数训练本应放大 drift |
| 可证伪性 | 模糊 | **清晰** — 单一断言，全面可测 |
| 护城河 | 弱 | **强** — 第一个系统性证明 null 的团队定义基线 |
| 实践意义 | "用 full FT" | **"不需要 full FT，head-only 就够了"** — 省算力 |
| 理论贡献 | CD-SPI 作为发散度量 | **CD-SPI 作为证伪工具** — 在应发现发散处未发现发散 |

### 论文标题示例

> *"Where Client Drift Should Be But Isn't: CD-SPI Reveals Capacity-Independent Representation Collapse in Federated PRM Training"*

### 核心叙事重构

1. **Motivation**：联邦学习文献普遍假设更多可训练参数 → 更大客户端 drift。联邦 PRM 领域未经检验地继承了这一假设。

2. **Method**：CD-SPI 诊断框架，系统化区分噪声型发散与结构化发散。

3. **Key Finding**：在 controlled dense backbone + FedAvg 设定下，从 256-dim head-only 到 1.4B full FT，客户端 backbone 表征保持 near-identical（cosine similarity > 0.999）。**容量不放大客户端发散。**

4. **Implication**：
   - **理论**：Pythia 类 dense backbone 的表征空间对联邦微调高度鲁棒，客户端 drift 假设在此架构下不成立
   - **实践**：联邦 PRM 训练不需要 full FT，head-only/LoRA 足以捕获跨客户端信号——省 10-100x 算力
   - **方法论**：CD-SPI 的价值不在于"发现发散"而在于**在预期发散处系统性地证明其不存在**

### 与阳性叙事的 CD-SPI 角色对比

| | 阳性叙事 | Null 叙事 |
|:---|:---|:---|
| CD-SPI 角色 | 发散度量工具 | **证伪工具 + 诊断框架** |
| 核心图表 | CD-SPI 随容量单调上升 | **CD-SPI 随容量呈平线**（含置信区间） |
| 统计证据 | 排列检验拒绝 H0 | **排列检验无法拒绝 H0 + 等价性检验 (TOST)** |
| 攻击面防御 | "CD-SPI 不够敏感" | "零发散不是测量问题——CKA/JS/函数空间散度三方交叉验证一致" |

### 必须预先准备的交叉验证数据

为防止审稿人攻击"CD-SPI 不够敏感所以测不到发散"：
- **CKA = 1.0**（已观测到）→ 特征空间无差异
- **JS 散度** → 输出空间无差异
- **函数空间余弦散度** → 参数空间无差异
- **等价性检验 (TOST)** → 统计证明"无差异"而非"未检测到差异"
- **合成数据正控制** → 在已知 ground truth 发散的数据上 CD-SPI 正确检测到信号（证明工具本身有效）

---

## 七、投稿策略

| 目标 | 期限 | 可行性 | 条件 |
|:---|---|:---:|:---|
| ICLR 2027 | 预计 2026 年 10 月 | 🟢 最现实 | 4 个月窗口：P0(2周)→P1(3周)→P2(3周)→论文(4周) |
| NeurIPS 2026 Workshop | 2026 年 7-8 月 | 🟡 可能 | 作为 mid-term checkpoint 获取反馈 |
