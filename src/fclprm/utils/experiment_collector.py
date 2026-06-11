"""简易实验数据收集模块 — 先暂存，后续再整理。"""

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


def collect_experiment(
    log_file: str | Path,
    config_file: str | Path | None = None,
    output_file: str | Path = "paper/experiment_draft.json",
) -> dict[str, Any]:
    """读取 JSONL 实验日志，汇总关键指标，暂存到单个 JSON 文件。

    不修改训练代码，训练完成后手动调用即可。

    Args:
        log_file: JSONL 日志路径（如 experiments/M2_centralized_prm/results/logs/xxx.jsonl）。
        config_file: 对应的 YAML 配置文件路径（可选）。
        output_file: 汇总结果暂存路径，默认 paper/experiment_draft.json。

    Returns:
        汇总后的字典，包含 final_metrics、best_metrics、config_summary、raw_points。
    """
    log_file = Path(log_file)
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if not log_file.exists():
        raise FileNotFoundError(f"日志文件不存在: {log_file}")

    # 读取所有日志记录
    records: list[dict] = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    if not records:
        raise ValueError(f"日志文件为空: {log_file}")

    # 按 step 排序
    records.sort(key=lambda r: r.get("metrics", {}).get("step", 0))

    # 提取最终记录（最后一条）
    final_record = records[-1]
    final_metrics = final_record.get("metrics", {})

    # 提取最佳记录（按 val_loss 最小）
    best_record = min(
        records,
        key=lambda r: r.get("metrics", {}).get("val_loss", float("inf")),
    )
    best_metrics = best_record.get("metrics", {})

    # 按 epoch 聚合训练 loss（取平均）
    epoch_losses: dict[int, list[float]] = defaultdict(list)
    for r in records:
        m = r.get("metrics", {})
        epoch = m.get("epoch")
        train_loss = m.get("train_loss")
        if epoch is not None and train_loss is not None:
            epoch_losses[epoch].append(train_loss)

    aggregated_epochs = [
        {"epoch": ep, "avg_train_loss": sum(losses) / len(losses)}
        for ep, losses in sorted(epoch_losses.items())
    ]

    # 读取配置摘要（如果提供）
    config_summary: dict[str, Any] | None = None
    if config_file and Path(config_file).exists():
        with open(config_file, "r", encoding="utf-8") as f:
            config_summary = yaml.safe_load(f)

    # 组装汇总
    summary = {
        "experiment_id": records[0].get("experiment_id", "unknown"),
        "milestone": records[0].get("milestone", "unknown"),
        "collected_at": datetime.utcnow().isoformat() + "Z",
        "log_file": str(log_file),
        "total_records": len(records),
        "final_metrics": {
            "step": final_metrics.get("step"),
            "epoch": final_metrics.get("epoch"),
            "train_loss": final_metrics.get("train_loss"),
            "val_loss": final_metrics.get("val_loss"),
            "val_accuracy": final_metrics.get("val_accuracy"),
            "val_f1": final_metrics.get("val_f1"),
            "val_auc": final_metrics.get("val_auc"),
            "learning_rate": final_metrics.get("learning_rate"),
        },
        "best_metrics": {
            "step": best_metrics.get("step"),
            "epoch": best_metrics.get("epoch"),
            "val_loss": best_metrics.get("val_loss"),
            "val_accuracy": best_metrics.get("val_accuracy"),
            "val_f1": best_metrics.get("val_f1"),
            "val_auc": best_metrics.get("val_auc"),
        },
        "epoch_summary": aggregated_epochs,
        "raw_points": [
            {
                "step": r.get("metrics", {}).get("step"),
                "epoch": r.get("metrics", {}).get("epoch"),
                "train_loss": r.get("metrics", {}).get("train_loss"),
                "val_loss": r.get("metrics", {}).get("val_loss"),
                "val_accuracy": r.get("metrics", {}).get("val_accuracy"),
                "val_f1": r.get("metrics", {}).get("val_f1"),
                "val_auc": r.get("metrics", {}).get("val_auc"),
                "learning_rate": r.get("metrics", {}).get("learning_rate"),
            }
            for r in records
        ],
        "config_summary": config_summary,
    }

    # 写入暂存文件
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[collector] 实验数据已暂存: {output_file}")
    print(f"  总记录数: {len(records)}")
    print(f"  最终 val_loss: {final_metrics.get('val_loss', 'N/A')}")
    print(
        f"  最佳 val_loss: {best_metrics.get('val_loss', 'N/A')} (step {best_metrics.get('step', 'N/A')})"
    )
    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="收集实验日志到暂存 JSON")
    parser.add_argument("--log", type=str, required=True, help="JSONL 日志文件路径")
    parser.add_argument(
        "--config", type=str, default=None, help="YAML 配置文件路径（可选）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paper/experiment_draft.json",
        help="输出 JSON 路径",
    )
    args = parser.parse_args()

    collect_experiment(args.log, args.config, args.output)
