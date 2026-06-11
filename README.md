# FCL-PRM-fullft

**Federated Continual Process Reward Model — Full Parameter Fine-tuning**  
联邦持续过程奖励模型 · 全参数微调

跨机构联邦学习协同训练 step-level PRM。**当前方向：联邦全参数微调。**

> 利用 NVIDIA GB10 (121GB 统一内存) 首次在消费级 GPU 上实现 Pythia-1.4B 全参数联邦 PRM 微调。

---

## 核心假设

head-only PRM 容量不足是联邦 PRM 的根本瓶颈：
- 256-dim head 在 centralized 下已达 99.7% 准确率 → 天花板太低
- 全参数微调给模型足够的容量产生有意义的跨客户端差异 → 聚合才有意义
- 详见 [PROJECT_FRAMEWORK.md](PROJECT_FRAMEWORK.md)

## 实验里程碑

- **M2** centralized 全参数微调 Pythia-1.4B（锚点）
- **M3** FedAvg 全参数微调（核心实验）
- **可选** AttnRes backbone 对比实验

## 硬件要求

- 运行全参数 FT 需要 ≥ 20GB GPU 内存（Pythia-1.4B, BF16+Adam）
- 本设备 GB10 可运行到 Pythia-6.9B 全参数 FT

---

## 背景调研：DeepSeek MoE 与本项目的关系

> 2026-06：对 DeepSeek 独家 MoE 技术的系统性调研，分析其与联邦全参数 PRM 微调的相关性。

### 1. DeepSeek MoE 架构核心创新

DeepSeek 的 MoE（Mixture of Experts）建立在两大架构创新之上（[DeepSeekMoE, ACL 2024](https://arxiv.org/abs/2401.06066)）：

**① 细粒度专家分割**：将每个 FFN 专家拆成 m 个小专家（缩小中间隐藏维度至 1/m），路由组合从 C(16,2)=120 暴增至 C(64,8)≈44 亿种（m=4 时）。

**② 共享专家隔离**：设定始终激活的共享专家捕获共性知识，消除跨专家参数冗余，实际保持 1:3 共享/激活路由比例。

**DeepSeek-V3**（[arXiv:2412.19437](https://arxiv.org/abs/2412.19437)）进一步引入：
- **无损负载均衡 Loss-Free Balancing**（[arXiv:2408.15664](https://arxiv.org/abs/2408.15664)）：用 expert-wise bias 替代辅助损失，不产生干扰梯度，验证集困惑度更低（9.50 vs 9.56@1B），已被形式化为 primal-dual 方法（[Han & Zhong, NeurIPS 2025](https://arxiv.org/abs/2512.03915)）
- **多 Token 预测（MTP）**：每个位置预测 D 个未来 token
- **总参数量 671B，每 token 仅激活 37B**

### 2. 与本项目的技术关联

| 维度 | DeepSeek MoE | FCL-PRM-fullft（本项目） | 关系与互补性 |
|------|-------------|------------------------|-------------|
| **优化层次** | 模型架构（推理效率） | 训练方法（数据效率） | 🔄 正交可叠加 |
| **稀疏性** | 参数级稀疏（MoE 路由） | 数据级稀疏（联邦客户端选择） | 🔄 概念对齐 |
| **微调范式** | 偏好轻量路由微调（RoMA 仅 0.0095% 参数） | 全参数微调（FullFT） | ⚠️ 存在张力 |
| **跨域泛化** | 专家路由处理多任务 | CD-SPI 度量跨领域步骤嵌入多义性 | 🔄 互补分析工具 |
| **负载均衡** | Loss-Free Balancing 均衡专家利用 | FedAvg 聚合均衡客户端贡献 | 🔄 不同层次的均衡问题 |

### 3. 关键发现

**路由次优性 gap（RoMA, ICLR 2026）**：现有 MoE 模型（含 DeepSeekMoE-16B-A3B）存在 **10-20% 的路由次优性差距**，即路由选择并非最优（[arXiv:2511.07419](https://arxiv.org/abs/2511.07419)）。RoMA 通过流形正则化仅微调路由（0.0095% 参数）即可缩小该差距。这对本项目的启示：**全参数微调中是否也需要关注路由层的 specialization 保持？**

**MoE-DPO（[arXiv:2510.08256](https://arxiv.org/abs/2510.08256)）**：首次将偏好优化（DPO）扩展到 MoE 架构，通过变分推断定义混合 Bradley-Terry（MBT）似然。这是目前唯一在原理层面整合 MoE 与偏好对齐的工作——为将来将 PRM 引入 MoE 模型提供了理论基础。

**CDSP-MoE（[arXiv:2512.20291](https://arxiv.org/abs/2512.20291)）**：利用梯度冲突（负余弦相似度）作为结构监督信号修剪冲突路径，在 small-scale 视觉任务上验证。虽然规模有限，但其**梯度冲突分析**思路与本项目的 CD-SPI 跨客户端散度度量有概念上的共鸣。

### 4. 对本项目的启示

1. **Backbone 扩展性**：如果本项目的联邦 PRM 方法有效，未来可扩展到 MoE 架构 backbone（如 DeepSeekMoE-16B），利用稀疏激活特性降低每客户端内存需求——MoE 的 37B/671B 稀疏激活比与联邦场景天然契合。

2. **路由微调 vs 全参数微调**：MoE 领域的最新工作（RoMA、CDSP-MoE）倾向于微调路由层而非全参数，这与本项目的 full-parameter 假设形成对比——未来实验可设计 **Partial FT 扫描**，探索 MoE 架构下路由微调 vs 全参数微调的 Pareto frontier。

3. **专家专业化与客户端漂移**：DeepSeek MoE 的专家专业化思想与本项目的 CD-SPI 有多方面共鸣——联邦客户端漂移可类比为"客户端专业化"，实现跨客户端的步骤嵌入对齐可能比跨领域对齐更易收敛。

4. **开源基础模型资源**：DeepSeek-V3 等开源 MoE 模型的 Checkpoint 为未来实验提供了比 Pythia 更强大的 backbone 候选。

### 5. 参考文献

- DeepSeekMoE → [arXiv:2401.06066](https://arxiv.org/abs/2401.06066) (ACL 2024)
- DeepSeek-V3 → [arXiv:2412.19437](https://arxiv.org/abs/2412.19437)
- Loss-Free Balancing → [arXiv:2408.15664](https://arxiv.org/abs/2408.15664)
- Primal-Dual 分析 → [arXiv:2512.03915](https://arxiv.org/abs/2512.03915) (NeurIPS 2025)
- RoMA（路由流形对齐） → [arXiv:2511.07419](https://arxiv.org/abs/2511.07419) (ICLR 2026)
- MoE-DPO（偏好优化 × MoE） → [arXiv:2510.08256](https://arxiv.org/abs/2510.08256)
- CDSP-MoE（梯度冲突剪枝） → [arXiv:2512.20291](https://arxiv.org/abs/2512.20291)

---

## 专家组诊断

> **2026-06-11**：六专家 Adversarial Panel 对当前论文命题进行了系统性批判审查。
> 平均评分 **4.25/10**，识别出 **3 个致命弱点**（LoRA 基线缺失、CD-SPI 无统计完备性、99.7% 饱和 benchmark）。详见完整报告：

👉 **[EXPERT_PANEL_REPORT.md](./EXPERT_PANEL_REPORT.md)**

核心行动项：
- **P0**：补充 LoRA 基线实验、OOD 测试、CD-SPI 跨 seed 统计验证 → 投稿前必须完成
- **叙事迁移**：从 "full FT 更好" → "CD-SPI 诊断框架揭示聚合信号的结构性散度"（第一贡献转移）
- **Plan B**：若 full FT 仅边际优势，论文全面转向 CD-SPI 方法论贡献（标题改为 *CD-SPI: Diagnosing Aggregation Bottlenecks in Federated Reward Models*）

---

## 许可证

MIT（待定）
