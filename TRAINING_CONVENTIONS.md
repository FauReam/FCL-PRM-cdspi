# FCL-PRM 训练代码规范

> 适用范围：`scripts/` 下所有持续性训练/模拟进程，以及 `src/fclprm/federated/` 中的联邦模拟器。
> 凡运行时间超过 30 秒的进程，均须满足以下五条。

## 0. 终端断开存活 (Terminal Survival) — 必须通过 `nohup` 启动

### 背景

实验通过 Tailscale SSH 远程运行。终端窗口随时可能断开（网络抖动、笔记本合盖、SSH 超时）。
断开时 shell 向**前台进程组**发送 `SIGHUP`，默认行为是终止进程。

### 0.1 代码侧要求（必须满足）

| 要求 | 说明 |
|------|------|
| **零 stdin 读取** | 代码中不得出现 `input()`、`sys.stdin.read()`、`getpass`、交互式确认等——训练进程启动后不得等待任何用户输入 |
| **stdout/stderr 全量落盘** | 必须使用 Tee 或等价的日志镜像机制，所有打印内容同时写入时间戳 `.log` 文件。终端断开后日志不丢失 |
| **crash report 可事后读取** | 崩溃时写结构化 crash JSON 到 `crashes/` 目录，而非仅打印到 stderr（终端断开后 stderr 不可见） |
| **零 GUI/浏览器依赖** | 不得使用 `wandb.login()` 交互式认证、`matplotlib.pyplot.show()` 等需要图形会话的操作。`use_wandb` 必须默认为 `false` |

### 0.2 启动命令要求

所有训练进程**必须**用 `nohup` 包装，确保 `SIGHUP` 被忽略：

```bash
# 正确 ✓
nohup python scripts/run_federated.py --config configs/xxx.yaml > /dev/null 2>&1 &

# 正确 ✓ (日志已被 Tee 接管，stdout/stderr 可丢弃)
nohup python scripts/run_federated.py --config configs/xxx.yaml &

# 错误 ✗ —— 终端断开即死
python scripts/run_federated.py --config configs/xxx.yaml
```

**为什么不用 tmux/screen？** 它们是可选工具，不是代码规范层面的保证。`nohup` 是 POSIX 标准工具，无需额外安装，且与 Tee 日志机制互补（日志持久化 → 事后查看，nohup → 进程存活）。

### 0.3 事后监控

```bash
# 查看进程是否存活
ps aux | grep run_federated

# 实时查看日志（即使终端断开后重连）
tail -f experiments/smoke_versaprm/results/logs/*.log

# 查看最新 checkpoint
ls -lt experiments/smoke_versaprm/results/checkpoints/
```

### 0.4 当前状态检查清单

新增训练脚本时逐项确认：

- [ ] 代码无 `input()` / `sys.stdin.read()` / 交互式确认
- [ ] stdout 已通过 Tee 镜像到日志文件
- [ ] crash report 写入 JSON 文件（非仅 stderr）
- [ ] `use_wandb` 为 `false` 或不依赖交互式认证
- [ ] 启动命令包含 `nohup ... &`
- [ ] 实验目录可事后 `tail -f` 查看进度

---

## 1. 进度条 (Progress Bars) — tqdm 覆盖所有耗时层级

### 1.1 强制层级

| 层级 | 位置 | desc 格式 | `leave` |
|------|------|-----------|---------|
| 顶层循环（rounds） | 调度器主循环 | `"Rounds"` | `True` |
| 中层循环（clients / eval） | 每个 round 内的 client 遍历 | `"[Eval] Per-domain"` 等 | `True` |
| 内层循环（epochs） | client.local_train() | `[Client {id}] epoch {n}/{total}` | `False` |
| 最内层（batches） | DataLoader 遍历 | 不设 desc，靠 postfix 显示 | `False` |

### 1.2 必须包含的信息

- `total`：必须设置总步数，不可留 `?`
- `postfix`：实时刷新 `loss`，频率由 `log_interval` 控制
- epoch 结束时 `tqdm.write()` 输出 `avg_loss`

### 1.3 禁止的做法

```python
# ✗ 无进度条
for round_num in range(num_rounds):
    ...

# ✗ 进度条无 total
for batch in tqdm(loader, desc="training"):
    ...

# ✗ 进度条 leave=True 导致日志刷屏
for epoch in range(epochs):
    for batch in tqdm(loader, leave=True):  # 应 leave=False
        ...
```

### 1.4 正确示例

```python
# 顶层
round_pbar = tqdm(range(num_rounds), desc="Rounds", total=num_rounds,
                   position=0, leave=True)
for round_num in round_pbar:
    # 内层 (client.py)
    epoch_pbar = tqdm(range(num_epochs), desc=f"  [Client {cid}]", leave=False)
    for epoch in epoch_pbar:
        batch_pbar = tqdm(loader, desc=f"    epoch {epoch+1}/{num_epochs}",
                          total=len(loader), leave=False)
        for batch in batch_pbar:
            ...
            if batch_idx % log_interval == 0:
                batch_pbar.set_postfix(loss=f"{loss:.4f}")

    # 更新顶层 postfix
    round_pbar.set_postfix(loss=f"{avg_loss:.4f}", time=f"{elapsed:.0f}s")
```

---

## 2. 中间点保存 (Checkpointing) — 每轮保存全部权重

### 2.1 保存时机

| 事件 | 保存内容 | 粒度 |
|------|----------|------|
| **每轮聚合后** | global model 全部权重 | round 级 |
| **每轮聚合后** (可选) | 各 client model 权重（用于事后 CD-SPI） | round × client 级 |
| **中途崩溃** | history 快照 (JSON) | 即时 |
| **最终轮结束** | 最终模型 + 完整 history | 最终 |

### 2.2 文件命名规范

```
{checkpoint_dir}/
├── model_m{M}_r{round}_c-1.pt        # global model (round 1-based)
├── clients/
│   ├── model_m{M}_r{round}_c0.pt     # client 0
│   ├── model_m{M}_r{round}_c1.pt     # client 1
│   └── ...
├── history.json                       # 完整 history（每轮覆盖写入）
└── crashes/
    └── crash_r{round}_{timestamp}.json
```

### 2.3 保存流程（强制性步骤顺序）

```python
def save_checkpoint(model, optimizer, round_num, client_id, milestone, save_dir, device):
    # 1. 确保目录存在
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # 2. 全部 tensor → CPU（避免 GPU 内存在 I/O 期间被占用）
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}

    # 3. optimizer state_dict 也移 CPU（注意嵌套结构）
    raw = optimizer.state_dict()
    cpu_optim = {"state": {}, "param_groups": raw["param_groups"]}
    for pid, sv in raw["state"].items():
        cpu_optim["state"][pid] = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in sv.items()
        }

    # 4. 写入磁盘
    torch.save({
        "model_state_dict": cpu_state,
        "optimizer_state_dict": cpu_optim,
        "round_num": round_num,
        "client_id": client_id,
        "milestone": milestone,
    }, filepath)

    # 5. 释放 CPU 副本 + 清 GPU 缓存
    del cpu_state, cpu_optim
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
```

### 2.4 备用恢复 (Crash Recovery)

```python
# 每轮结束时覆盖写入 history.json（原子写入：先写 .tmp 再 rename）
tmp = history_path.with_suffix(".tmp.json")
with open(tmp, "w") as f:
    json.dump(clean_history, f, indent=2)
tmp.replace(history_path)
```

### 2.5 禁止的做法

```python
# ✗ 存 GPU tensor（I/O 时 OOM）
torch.save(model.state_dict(), path)

# ✗ 只存 head 不存 backbone（事后无法复现）
torch.save(model.head.state_dict(), path)

# ✗ 不释放临时变量（内存泄漏）
cpu_state = {k: v.cpu() ...}
torch.save(cpu_state, path)
# 缺少 del cpu_state; gc.collect()
```

---

## 3. 错误栈保存 (Error Traceback) — 结构化 crash report

### 3.1 三层错误处理

```
Layer 1: 进程级 — try/except 包裹 main()，捕获未处理异常
Layer 2: 调度级 — per-client try/except，单 client 崩溃不影响其他
Layer 3: 系统级 — SIGINT handler，Ctrl+C 优雅退出
```

### 3.2 进程级

```python
def main():
    try:
        # ... 全部训练逻辑 ...
    except KeyboardInterrupt:
        print("\n[INTERRUPT] User requested shutdown")
        sys.exit(130)
    except Exception as e:
        # 写入结构化 crash report
        crash_path = log_dir / "crashes" / f"crash_{timestamp}.json"
        crash_path.parent.mkdir(parents=True, exist_ok=True)
        with open(crash_path, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "error_type": type(e).__name__,
                "error_message": str(e),
                "traceback": traceback.format_exc(),
            }, f, indent=2, ensure_ascii=False)
        print(f"\n[FATAL] Crash report saved to: {crash_path}")
        raise
```

### 3.3 调度级（per-client 隔离）

```python
for client in active_clients:
    try:
        update = client.local_train(...)
    except Exception as e:
        _save_crash_report(f"round_{r}_client_{cid}", e)
        client_errors.append(cid)
        continue  # 跳过失败 client，其他继续
```

### 3.4 SIGINT 优雅退出

```python
def _setup_signal_handler():
    original = signal.getsignal(signal.SIGINT)
    def _handler(sig, frame):
        self._interrupted = True
        tqdm.write("\n  [SIGNAL] Finishing current round, then exiting...")
        signal.signal(signal.SIGINT, original)
    signal.signal(signal.SIGINT, _handler)

# 主循环中检查
if self._interrupted:
    self._save_history_snapshot()   # 保存已完成轮次
    break
```

### 3.5 Crash Report 格式

```json
{
  "timestamp": "2026-06-15T14:30:00",
  "stage": "round_3_client_2",
  "round": 3,
  "error_type": "RuntimeError",
  "error_message": "CUDA out of memory. Tried to allocate 2.00 GiB...",
  "traceback": "Traceback (most recent call last):\n  File ...\n..."
}
```

---

## 4. 运行日志保存 (Run Logging) — stdout/stderr 全量持久化

### 4.1 双通道日志

```
通道 A: ExperimentLogger → JSONL 结构化指标
通道 B: Tee(stdout) → 文本日志（含 tqdm 输出）
```

### 4.2 Tee 类 — stdout 镜像到文件

必须实现的功能：

```python
class Tee:
    """将 stdout 同时输出到终端和日志文件。"""

    def __init__(self, log_path: Path):
        self.terminal = sys.stdout
        self.log_file = log_path.open("w", encoding="utf-8")
        atexit.register(self._flush_and_close)  # 崩溃时也能刷盘

    def write(self, message: str):
        self.terminal.write(message)          # 终端始终看到全部
        self.terminal.flush()
        # 日志文件：处理 tqdm 的 \r 覆盖，避免刷屏
        if "\r" in message and "\n" not in message:
            self._file_buf = message.rsplit("\r", 1)[-1]  # 只保留最后一行
        elif "\n" in message:
            parts = self._file_buf.split("\n")
            for part in parts[:-1]:
                self.log_file.write(part + "\n")
            self._file_buf = parts[-1]
            self.log_file.flush()
```

### 4.3 使用方式

```python
def main():
    log_dir = Path(config.get("logging.log_dir"))
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"run_{datetime.now():%Y%m%d_%H%M%S}.log"

    original_stdout = sys.stdout
    tee = Tee(log_path)
    sys.stdout = tee
    try:
        # ... 全部训练逻辑（print / tqdm.write 自动入日志）...
    finally:
        sys.stdout = original_stdout
        tee.close()
```

### 4.4 ExperimentLogger — 结构化指标

```python
logger = ExperimentLogger(log_dir, experiment_id)
logger.log(
    milestone="M3",
    config_hash=config.hash(),
    metrics={
        "round": 5,
        "avg_loss": 0.234,
        "per_domain_mse": {"c0": 0.12, "c1": 0.34},
        "cd_spi_mean": 0.45,
    }
)
# 输出: {log_dir}/{experiment_id}.jsonl
# 每行一条 JSON record
```

### 4.5 日志目录结构

```
experiments/{exp_name}/
├── logs/
│   ├── run_20260615_143000.log        # Tee 文本日志
│   └── {experiment_id}.jsonl          # 结构化指标
├── checkpoints/
│   ├── model_mM3_r1_c-1.pt
│   ├── clients/
│   │   └── model_mM3_r1_c0.pt
│   └── ...
└── crashes/
    └── crash_r3_20260615_150000.json
```

---

## 5. 检查清单

新增训练脚本时，逐项确认：

### 终端存活 (§0)
- [ ] 代码无 `input()` / `sys.stdin.read()` / 交互式确认
- [ ] stdout 已通过 Tee 镜像到日志文件
- [ ] crash report 写入 JSON 文件（非仅 stderr）
- [ ] `use_wandb` 为 `false` 或不依赖交互式认证
- [ ] 启动命令包含 `nohup ... &`

### 进度条 (§1)
- [ ] 顶层循环有 `tqdm(..., desc="Rounds", total=N, leave=True)`
- [ ] 中层循环（clients/eval）有 `tqdm(..., leave=True)`
- [ ] 内层（epochs/batches）有 `tqdm(..., leave=False)`
- [ ] batch postfix 实时显示 loss，由 `log_interval` 控制频率

### 检查点 (§2)
- [ ] 每轮结束后保存 global model checkpoint（CPU 化 + gc + empty_cache）
- [ ] checkpoint 文件含完整权重（model + optimizer state_dict）
- [ ] history 每轮原子写入（`.tmp` → `rename`）

### 错误栈 (§3)
- [ ] `main()` 包裹 try/except，崩溃时写 crash JSON
- [ ] per-client training 包裹 try/except，单 client 失败不终止全局
- [ ] SIGINT handler 设置 `_interrupted` flag + 保存 snapshot
- [ ] `atexit` 注册日志 flush，确保崩溃时日志不丢失

### 运行日志 (§4)
- [ ] stdout 通过 Tee 镜像到时间戳日志文件
- [ ] 结构化指标通过 ExperimentLogger 写 JSONL
- [ ] `torch.cuda.empty_cache()` 在 checkpoint 后和 client 间调用

---

## 6. 当前实现参考

| 组件 | 文件 | 对应规范条目 |
|------|------|-------------|
| 多层 tqdm | `src/fclprm/federated/simulator.py:544-830` | §1 进度条 |
| checkpoint save | `src/fclprm/models/checkpoint.py:11-92` | §2 中间点保存 |
| history snapshot | `src/fclprm/federated/simulator.py:453-471` | §2.4 备用恢复 |
| crash report | `src/fclprm/federated/simulator.py:475-496` | §3 错误栈 |
| per-client isolation | `src/fclprm/federated/simulator.py:607-617` | §3.3 调度级 |
| SIGINT handler | `src/fclprm/federated/simulator.py:440-451` | §3.4 SIGINT |
| Tee 日志镜像 | `scripts/run_federated.py:135-193` | §4.2 Tee |
| ExperimentLogger | `src/fclprm/utils/logging.py:8-47` | §4.4 JSONL |
| GPU isolation (save→clear→reload) | `src/fclprm/federated/simulator.py:706-791` | — |
