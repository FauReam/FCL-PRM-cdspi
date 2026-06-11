# FCL-PRM

**Federated Continual Process Reward Model** — 联邦持续过程奖励模型

跨机构联邦学习协同训练 step-level PRM。**当前方向：联邦全参数微调。**

> 利用 NVIDIA GB10 (121GB 统一内存) 首次在消费级 GPU 上实现 Pythia-1.4B 全参数联邦 PRM 微调。

---

## 核心假设

head-only PRM 容量不足是联邦 PRM 的根本瓶颈：
- 256-dim head 在 centralized 下已达 99.7% 准确率 → 天花板太低
- 全参数微调给模型足够的容量产生有意义的跨客户端差异 → 聚合才有意义
- 详见 [PROJECT_FRAMEWORK.md](PROJECT_FRAMEWORK.md)

## 实验里程碑

- **M2** centralized 全参数微调 Pythia-1.4B（新锚点）
- **M3** FedAvg 全参数微调（核心实验）
- **M3-LoRA** FedAvg + LoRA 轻量对照

## 硬件要求

- 运行全参数 FT 需要 ≥ 20GB GPU 内存（Pythia-1.4B, BF16+Adam）
- 本设备 GB10 可运行到 Pythia-6.9B 全参数 FT

## 许可证

MIT（待定）
