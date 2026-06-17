# FCL-PRM 修改日志

> 记录效率优化、算法改进、架构调整、硬件适配类变更。
> 架构级决策写入 `docs/decisions.md`。

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
