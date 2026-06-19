# FCL-PRM 项目框架 — 待重写

> **2026-06-18**：旧 CD-SPI 诊断框架方向已被实验证伪（全配置 CD-SPI sym ≈ 0.001）。
> 新方向待确认，详见 [DIRECTION.md](DIRECTION.md)。

## 旧框架摘要（存档）

- 核心贡献：CD-SPI 诊断框架，区分噪声型与信号型客户端发散
- 实验变量：容量连续谱（head-only → LoRA → partial FT → full FT）
- 证伪原因：CD-SPI sym 在所有配置上无区分度，CKA 始终 = 1.0，数据过小导致过拟合
