# FCL-PRM

**Federated Process Reward Model** · 联邦过程奖励模型

方向重定向中 — 详见 [DIRECTION.md](DIRECTION.md)。

## 当前状态

- 旧 CD-SPI 诊断框架已被实验证伪（全配置 CD-SPI sym ≈ 0.001）
- 与导师（2026-06-18）讨论后确认：需从"诊断"转向"因果解释"
- 核心前提待解决：换小模型（70M）+ 大数据（PRM800K + Med-PRM）

## 硬件

NVIDIA DGX Spark GB10（121GB 统一内存，ARM64）

## 关键路径

- `scripts/run_federated.py` — 联邦模拟主入口
- `scripts/train_centralized_prm.py` — 集中式训练
- `src/fclprm/models/base_wrapper.py` — StepRewardModel
- `src/fclprm/metrics/cd_spi.py` — CD-SPI
- `src/fclprm/federated/simulator.py` — 联邦模拟调度
- `configs/` — 实验配置
