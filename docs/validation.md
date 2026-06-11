# FCL-PRM 验证方案

> 检验方法 + 预期结果，覆盖从单元测试到科学假设的全链路。

---

## 1. 三层验证体系

```
Layer 3: 科学假设验证 (M2-M6)
    ↓ 每个 milestone 的科学假设是否成立
Layer 2: 集成测试 (smoke test)
    ↓ 端到端流程能否跑通
Layer 1: 单元测试 (pytest)
    ↓ 每个模块的数学/逻辑正确性
```

---

## 2. Layer 1: 单元测试验证

### 运行方法

```bash
cd C:/Users/46326/project/FCL-PRM
pytest tests/ -v
```

### 通过标准

| 测试文件 | 检验内容 | 通过标志 |
|---|---|---|
| `test_data.py` | step 分割、batch padding | `test_split_cot_into_steps PASSED` |
| `test_models.py` | PRMHead 输出 shape | `test_prm_head_forward PASSED` |
| `test_federated.py` | FedAvg 数学正确性、trimmed mean | 4 个 test 全部 PASSED |
| `test_metrics.py` | CD-SPI 范围 [0,1]、边界情况 | 5 个 test 全部 PASSED |

**预期结果**：`4 passed in X.XXs`

---

## 3. Layer 2: 集成测试验证

### 3.1 M2 Smoke Test — 中心化训练能跑通

```bash
# 准备：下载 PRM800K 数据到 ./data/prm800k/
python scripts/train_centralized_prm.py \
    --config configs/m2_pythia_1b.yaml
```

**检验方法**：
- 不报错退出
- 日志中出现 `train_loss=` 且数值在下降
- checkpoint 文件生成在 `experiments/M2_centralized_prm/results/checkpoints/`

**预期结果**：
- 初始 val_loss ≈ 0.25（随机猜测，MSE）
- 训练后 val_loss < 0.10（有学习信号）
- 若 val_loss 不降 → 检查数据格式、学习率、loss 函数

### 3.2 M3 Smoke Test — 联邦仿真能跑通

```bash
# 准备：下载 VersaPRM 数据到 ./data/versaprm/
python scripts/run_federated.py \
    --config configs/m3_naive_fedavg.yaml
```

**检验方法**：
- 4 个 client 依次训练，无报错
- 每轮输出 `avg_loss=` 且数值收敛
- JSONL 日志写入 `experiments/M3_naive_fedavg_prm/results/logs/`

**预期结果**：
- 初始 avg_loss ≈ 0.25
- 50 轮后 avg_loss < 0.15
- 各 client loss 差异显著（> 0.05）→ 证明域异质性存在

### 3.3 M5 Smoke Test — CD-SPI 能计算

```bash
python scripts/compute_cd_spi.py \
    --config configs/m5_cd_spi.yaml
```

**检验方法**：
- 输出 4 个 step 的 CD-SPI 值
- 值在 [0, 1] 范围内
- JSON 结果文件生成

**预期结果**（关键）：
- 若 CD-SPI 普遍 < 0.1 → step 嵌入是跨域通用的（ Anchor-PRM 价值有限）
- 若部分 step CD-SPI > 0.3 → 存在显著 polysemy（ Anchor-PRM 有价值）
- 若 CD-SPI 全部 ≈ 0.5 → 数据噪声过大或模型未充分训练

---

## 4. Layer 3: 科学假设验证（核心）

### M2 假设：中心化 PRM 能学到 step-level 信号

| 指标 | 通过阈值 | 失败含义 |
|---|---|---|
| ProcessBench Accuracy | > 0.60 | PRM 未学到有效信号 |
| Best-of-N@64 vs Majority Vote | +5% | PRM 无鉴别力 |
| 训练 loss 收敛 | < 0.10 | 学习率/数据问题 |

**预期结果**：
- Pythia 1.4B + PRM head 在 PRM800K 上达到 ~0.65 accuracy
- Best-of-N@64 比 majority vote 提升 5-10%

### M3 假设：朴素 FedAvg 在 step-level PRM 上失败

| 指标 | 通过阈值 | 失败含义 |
|---|---|---|
| FedAvg 全局模型 vs 本地模型 | 全局 < 各本地平均 | FedAvg 反而有效（意料外） |
| 跨域评估 drop | > 15% | 域差异不足以造成失败 |

**预期结果**：
- 全局模型在 math 域表现好，在 medical 域表现差（反之亦然）
- 相比本地模型平均，全局模型性能下降 15-30%
- **这是 M3 的"成功"**：证明了 naive FedAvg 失败，为 M4 的 Anchor-PRM 提供动机

### M4 假设：Anchor-PRM 优于 Naive FedAvg

| 指标 | 通过阈值 | 失败含义 |
|---|---|---|
| Anchor-PRM 全局 vs FedAvg 全局 | +10% | 对齐机制无效 |
| CD-SPI 下降 | 20% ↓ | 对齐未改变嵌入结构 |

**预期结果**：
- Anchor-PRM 全局模型在所有域上平均优于 FedAvg 10%+
- 对齐后的 CD-SPI 比未对齐低 20%+

### M5 假设：CD-SPI 能区分 step 类别

| 指标 | 通过阈值 | 失败含义 |
|---|---|---|
| Logical connector CD-SPI | < 0.15 | 所有 step 都是 polysemous |
| Domain reference CD-SPI | > 0.40 | 所有 step 都是 universal |
| 类别间差异 | p < 0.05 (t-test) | CD-SPI 无统计显著性 |

**预期结果**：
| Step 类别 | 预期 CD-SPI | 解释 |
|---|---|---|
| Logical connector (therefore, because) | 0.05-0.15 | 跨域通用 |
| Variable definition (let x = ...) | 0.10-0.25 |  mostly 通用 |
| Domain reference (clinical term / API) | 0.40-0.70 | 高度 polysemous |
| Calculation (arithmetic) | 0.15-0.30 |  math/code 通用，medical 不同 |

**关键判断**：
- 若类别间 CD-SPI 差异显著 → P2 科学假设成立，可发 paper
- 若差异不显著 → 需要重新设计 step 分类或检查模型训练

### M6 假设：Step-level DP 有紧界需求

| 指标 | 通过阈值 | 失败含义 |
|---|---|---|
| MIA AUC (no DP) | > 0.70 | 攻击基线太弱 |
| MIA AUC (DP ε=4) | < 0.55 | DP 未能保护隐私 |
| Utility drop (ε=4 vs ε=∞) | < 15% | DP 噪声过大 |

**预期结果**：
- 无 DP 时 MIA AUC ≈ 0.75（成员可被明显区分）
- ε=4 时 MIA AUC ≈ 0.52（接近随机猜测）
- ε=4 时 utility 下降 < 10%（可用）
- **核心发现**：step-level 需要比 outcome-level 更强的 DP（σ 大 √T 倍）

---

## 5. 快速诊断清单

### 项目能跑起来的最低验证

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 单元测试通过
pytest tests/ -v

# 3. 能 import
python -c "from fclprm.models.base_wrapper import StepRewardModel; print('OK')"

# 4. 配置能加载
python -c "from fclprm.utils.config import ExperimentConfig; \
           c = ExperimentConfig('configs/m2_pythia_1b.yaml'); \
           print(c.get('model.backbone'))"

# 5. 数据加载器能实例化
python -c "from fclprm.data.prm800k import PRM800KLoader; \
           loader = PRM800KLoader('./data/prm800k'); \
           print('Loader created')"
```

**全部通过 = 项目骨架可用，可进入 M2 实验。**

### 常见问题排查

| 现象 | 原因 | 修复 |
|---|---|---|
| `pytest` 找不到模块 | 未安装包 | `pip install -e .` |
| `FileNotFoundError` | 数据未下载 | 下载 PRM800K / VersaPRM |
| CUDA OOM | batch size 太大 | 改 YAML 中 `batch_size` |
| loss 不下降 | 学习率太大/太小 | 调 `learning_rate` |
| CD-SPI 全为 0 | 模型未训练 | 先跑 M2/M3 训练 |
| CD-SPI 全为 1 | embedding 未归一化 | 检查 `get_step_embedding` |

---

## 6. 成功标准总结

| 层级 | 成功标志 |
|---|---|
| **工程成功** | pytest 全过 + 脚本不报错 + checkpoint 生成 |
| **方法成功** | Anchor-PRM > FedAvg + CD-SPI 有显著信号 |
| **科学成功** | Step polysemy 被量化 + Step-level DP 紧界被证明 |
| **投稿成功** | P1 投 NeurIPS/ICLR + P2 有 TMLR 级别发现 |
