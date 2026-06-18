# FCL-PRM 修改日志

> 记录效率优化、算法改进、架构调整、硬件适配类变更。
> 架构级决策写入 `docs/decisions.md`。

## 2026-06-18

### fix(federated): unwrap OptimizedModule for all state_dict operations
- **问题**: PyTorch 2.11 `torch.compile` 的 `OptimizedModule.state_dict()` 在所有 key 前添加 `_orig_mod.` 前缀，`load_state_dict()` 也要求带此前缀。
- **影响**: 编译版 global_model 的 state_dict 加载到未编译的 client model 时崩溃（全参数 FT / LoRA）；编译版 global_model 接受来自未编译 client update 的裸 state_dict 时聚合也可能失败。
- **修复**: 在 `server.py` 新增 `_unwrap()` / `_raw_model()` 方法，在 `aggregators.py` 新增 `_unwrap()` 辅助函数。所有 state_dict 读/写均通过 `_orig_mod` 解包后的原始模块操作。
- **影响文件**: `src/fclprm/federated/server.py`, `aggregators.py`, `simulator.py`

---

## 2026-06-17 — Eval DataLoader 多进程 + ARM64 compile 尝试

| 字段 | 内容 |
|------|------|
| **日期** | 2026-06-17 |
| **类型** | 性能优化 |
| **影响范围** | `src/fclprm/federated/simulator.py`, `src/fclprm/metrics/ood_eval.py`, `scripts/run_federated.py` |
| **变更前** | Eval DataLoader 硬编码单进程（`simulator.py:_eval_per_domain` 无 `num_workers`）；ARM64 平台直接跳过 `torch.compile` |
| **变更后** | 三个 eval 方法 + `ood_eval.py` 两个评估函数均支持 `num_workers`，>0 时启用 `prefetch_factor=2` + `persistent_workers`；ARM64 也尝试 `torch.compile(mode="default")`，失败优雅降级；`torch.no_grad()` 统一替换为 `torch.inference_mode()` |
| **预期效果** | eval 阶段快 ~20-30%（从 ~8h 降至 ~5-6h per 2-round run）；ARM64 compile 取决于 PyTorch/Triton 版本 |
| **风险** | 低 — 所有新参数默认值为 0，完全向后兼容；compile 有 try/except 保护 |
| **验证方式** | `venv/bin/python -c "import py_compile; ..."` 语法检查三文件通过；下次启动 `run_federated.py` 观察日志中的 `[INFO] Eval DataLoader` 和 compile 状态 |
| **关联 commit** | `8fc801b` |

---

## 2026-06-17 — OptimizedModule 克隆 + centralized --rounds 兼容性修复

| 字段 | 内容 |
|------|------|
| **日期** | 2026-06-17 |
| **类型** | Bug 修复 |
| **影响范围** | `src/fclprm/federated/simulator.py`, `scripts/train_centralized_prm.py` |
| **变更前** | (1) `torch.compile()` 成功时 `type(global_model)` 返回 `OptimizedModule`，`simulator.py` 中 `type(global_model)(backbone=...)` 抛出 `TypeError`，导致 full-FT / LoRA 实验全部 1 分钟内崩溃。(2) `train_centralized_prm.py` 未接受 `--rounds` 参数，batch 脚本传入导致 `unrecognized arguments` 退出。 |
| **变更后** | (1) `simulator.py` 检测 `hasattr(global_model, "_orig_mod")` 获取原始模型类后再实例化 client models。(2) `train_centralized_prm.py` 接受 `--rounds` 参数（ignored，仅作兼容）。 |
| **预期效果** | full-FT / LoRA / centralized 实验不再 1 分钟崩溃，批次 5/5 可启动。 |
| **风险** | 低 — `_orig_mod` 访问是 `torch.compile` 公开 API；`--rounds` 为可选参数。 |
| **验证方式** | 重新运行 batch 脚本，确认 2/5 通过（M2 centralized 仍可能因配置问题失败）。 |
| **关联 commit** | (本 commit) |
