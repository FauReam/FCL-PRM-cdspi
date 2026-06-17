#!/usr/bin/env python3
"""M2: Train centralized PRM baseline on PRM800K.

Usage:
    python scripts/train_centralized_prm.py --config configs/m2_pythia_1b.yaml
"""

import argparse
import hashlib
import json as _json
import os
import signal
import sys
import traceback as _traceback
from datetime import datetime, timezone
from pathlib import Path

# Run offline by default — use cached models, never auto-update from HF Hub.
# Set HF_HUB_OFFLINE=0 to allow downloads of new models.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    AutoModel,
    AutoTokenizer,
    get_constant_schedule,
    get_cosine_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

from fclprm.data.prm800k import PRM800KLoader
from fclprm.data.versa_loader import VersaPRMLoader
from fclprm.data.utils import collate_step_batch
from fclprm.models.base_wrapper import StepRewardModel
from fclprm.models.checkpoint import save_checkpoint
from fclprm.utils.config import ExperimentConfig
from fclprm.utils.logging import ExperimentLogger
from fclprm.utils.seed import set_seed
from fclprm.utils.tee import Tee


def _load_hf_asset(load_fn, model_name: str, **kwargs):
    """Load from HF Hub with automatic fallback to local cache on network errors."""
    try:
        return load_fn(model_name, local_files_only=False, **kwargs)
    except Exception as e:
        err_name = type(e).__name__
        if any(x in err_name for x in ("ConnectTimeout", "ConnectionError", "HTTPError", "OfflineModeIsEnabled")):
            print(f"[WARN] Hub unreachable ({err_name}), falling back to local cache...")
            return load_fn(model_name, local_files_only=True, **kwargs)
        raise


def _get_data_cache_path(config: ExperimentConfig) -> Path:
    """Return a cache path for pre-tokenized step dataset."""
    key_parts = [
        config.get("data.data_dir", ""),
        config.get("data.split", "train"),
        str(config.get("data.max_length", 512)),
        config.get("model.backbone", ""),
        config.get("data.dataset", "prm800k"),
        str(config.get("experiment.seed", 42)),
    ]
    key = hashlib.md5("|".join(key_parts).encode()).hexdigest()
    cache_dir = Path(config.get("logging.log_dir", "./logs")).parent / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"step_dataset_{key}.pt"


# ── Global state for graceful shutdown ──────────────────────────────────
_interrupted = False


def _setup_sigint_handler() -> None:
    """Install SIGINT handler for graceful shutdown on Ctrl+C."""
    original = signal.getsignal(signal.SIGINT)

    def _handler(sig, frame):
        global _interrupted
        _interrupted = True
        print("\n  [SIGNAL] SIGINT received — finishing current epoch, then exiting...")
        print("  [SIGNAL] Press Ctrl+C again to force exit.")
        signal.signal(signal.SIGINT, original)

    signal.signal(signal.SIGINT, _handler)


def _save_crash_report(error: Exception, stage: str, checkpoint_dir: str) -> Path:
    """Save structured crash report for post-mortem debugging."""
    crash_dir = Path(checkpoint_dir).parent / "crashes"
    crash_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    crash_path = crash_dir / f"crash_{stage}_{timestamp}.json"
    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "error_type": type(error).__name__,
        "error_message": str(error),
        "traceback": _traceback.format_exc(),
    }
    with open(crash_path, "w") as f:
        _json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[FATAL] {report['error_type']}: {report['error_message']}")
    print(f"[FATAL] Crash report saved to: {crash_path}")
    return crash_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train centralized PRM baseline")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--rounds", type=int, default=None,
        help="(ignored; accepted for batch-script compatibility with federated entrypoints)"
    )
    args = parser.parse_args()

    config = ExperimentConfig(args.config)
    set_seed(config.get("experiment.seed", 42))

    device = config.get(
        "hardware.device", "cuda" if torch.cuda.is_available() else "cpu"
    )
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        device = torch.device("cuda:0")
    else:
        device = torch.device(device)

    # AMP scaler for mixed-precision training on RTX 4070
    use_amp = device.type == "cuda"
    scaler = GradScaler(device.type) if use_amp else None

    log_dir = Path(config.get("logging.log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = config.get("logging.checkpoint_dir", "./checkpoints")
    save_every = config.get("logging.save_every", 0)
    log_interval = config.get("training.log_interval", 10)

    # ── Mirror stdout to timestamped log file ─────────────────────────
    timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{config.get('experiment.name', 'm2_baseline')}_{timestamp_str}.log"
    tee = Tee(log_path)
    sys.stdout = tee
    print(f"[INFO] Logging mirrored to: {log_path}")

    logger = ExperimentLogger(
        log_dir=str(log_dir),
        experiment_id=config.get("experiment.name", "m2_baseline"),
    )

    # ── SIGINT handler ─────────────────────────────────────────────────
    _setup_sigint_handler()

    try:
        _run_training(
            config=config, args=args, device=device, use_amp=use_amp,
            scaler=scaler, log_dir=log_dir, checkpoint_dir=checkpoint_dir,
            save_every=save_every, log_interval=log_interval, logger=logger,
        )
    except KeyboardInterrupt:
        _save_crash_report(
            KeyboardInterrupt("User interrupted training"),
            "training", checkpoint_dir,
        )
    except Exception as e:
        _save_crash_report(e, "training", checkpoint_dir)
        raise
    finally:
        if sys.stdout is tee:
            sys.stdout = tee.terminal
            tee.close()
            print(f"[INFO] Log saved to: {log_path}")


def _save_training_snapshot(
    model, optimizer, global_step: int, epoch: int,
    best_val_loss: float, checkpoint_dir: str, config: ExperimentConfig,
) -> Path:
    """Save a recovery snapshot of the current training state.

    Called on SIGINT or at end of each epoch for crash recovery.
    Writes a minimal JSON manifest alongside a checkpoint so that
    `--resume` can pick up where it left off.
    """
    snapshot_dir = Path(checkpoint_dir) / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_dir / "latest_training_state.pt"

    # Save full model + optimizer state
    cpu_state = {k: v.cpu() for k, v in model.state_dict().items()}
    raw_optim = optimizer.state_dict()
    cpu_optim: dict = {"state": {}, "param_groups": raw_optim["param_groups"]}
    for pid, sv in raw_optim["state"].items():
        cpu_optim["state"][pid] = {
            k: v.cpu() if isinstance(v, torch.Tensor) else v
            for k, v in sv.items()
        }

    torch.save({
        "model_state_dict": cpu_state,
        "optimizer_state_dict": cpu_optim,
        "global_step": global_step,
        "epoch": epoch,
        "best_val_loss": best_val_loss,
    }, snapshot_path)

    # Write companion manifest
    manifest_path = snapshot_dir / "latest_training_state.json"
    with open(manifest_path, "w") as f:
        _json.dump({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "global_step": global_step,
            "epoch": epoch + 1,
            "best_val_loss": best_val_loss,
            "config_hash": config.hash(),
        }, f, indent=2)

    del cpu_state, cpu_optim
    import gc
    gc.collect()
    return snapshot_path


def _run_training(
    config: ExperimentConfig,
    args: argparse.Namespace,
    device: torch.device,
    use_amp: bool,
    scaler: GradScaler | None,
    log_dir: Path,
    checkpoint_dir: str,
    save_every: int,
    log_interval: int,
    logger: ExperimentLogger,
) -> None:
    """Core training logic — factored out so main() can wrap it in try/except."""

    print(f"[M2] Initializing model: {config.get('model.backbone')}")
    try:
        tokenizer = _load_hf_asset(
            AutoTokenizer.from_pretrained, config.get("model.backbone")
        )
    except OSError as e:
        print(f"[ERROR] Failed to load tokenizer for '{config.get('model.backbone')}'.")
        print(f"  {e}")
        print("  Please ensure the model name is correct and you have internet access,")
        print(
            "  or download the model locally and set local_files_only=True in config."
        )
        return
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    freeze_backbone = config.get("model.freeze_backbone", True)
    load_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        backbone = _load_hf_asset(
            AutoModel.from_pretrained,
            config.get("model.backbone"),
            dtype=load_dtype,
        )
    except OSError as e:
        print(f"[ERROR] Failed to load backbone for '{config.get('model.backbone')}'.")
        print(f"  {e}")
        print("  Please ensure the model name is correct and you have internet access,")
        print(
            "  or download the model locally and set local_files_only=True in config."
        )
        return
    attnres_config = config.get("model.attnres", None)
    lora_config = config.get("model.lora", None)
    partial_ft_layers = config.get("model.partial_ft_layers", 0)
    partial_ft_mode = config.get("model.partial_ft_mode", "last_n")
    head_activation = config.get("model.head_activation", "relu")

    if attnres_config is not None:
        print(f"[M2] Block AttnRes enabled: {attnres_config.get('num_blocks', 8)} blocks")
    if lora_config is not None:
        print(f"[M2] LoRA enabled: r={lora_config.get('r', 8)}")
    if partial_ft_layers > 0:
        print(f"[M2] Partial FT: mode={partial_ft_mode}, last {partial_ft_layers} layers unfrozen")
    if head_activation != "relu":
        print(f"[M2] Head activation: {head_activation} (architecture ablation)")

    model = StepRewardModel(
        backbone=backbone,
        head_dim=config.get("model.prm_head_dim", 256),
        freeze_backbone=freeze_backbone,
        attnres=attnres_config,
        lora_config=lora_config,
        partial_ft_layers=partial_ft_layers,
        partial_ft_mode=partial_ft_mode,
        head_activation=head_activation,
    )
    model.to(device)

    parts = []
    if lora_config is not None:
        parts.append(f"LoRA(r={lora_config.get('r', 8)})")
    elif partial_ft_layers > 0:
        parts.append(f"partial-FT({partial_ft_mode}, last {partial_ft_layers})")
    elif freeze_backbone:
        parts.append("head-only")
    else:
        parts.append("full-parameter")
    if attnres_config is not None:
        parts.append("AttnRes")
    mode = " + ".join(parts)
    print(f"[M2] Training mode: {mode} ({load_dtype})")

    # torch.compile needs Triton, which requires x86-64 + CUDA.
    # On ARM64 (NVIDIA GB10) Triton cannot compile its CUDA utils.
    is_arm64 = sys.platform.startswith("linux") and (
        hasattr(os, "uname") and os.uname().machine in ("aarch64", "arm64")
    )
    if device.type == "cuda" and hasattr(torch, "compile") and sys.platform != "win32" and not is_arm64:
        print("[M2] Enabling torch.compile for faster training...")
        model = torch.compile(model, mode="max-autotune")
    elif device.type == "cuda" and (sys.platform == "win32" or is_arm64):
        print(f"[M2] Skipping torch.compile ({'ARM64' if is_arm64 else 'Windows'}, Triton unavailable).")

    print(f"[M2] Loading data from: {config.get('data.data_dir')}")
    dataset_name = config.get("data.dataset", "prm800k")

    cache_path = _get_data_cache_path(config)
    if cache_path.exists():
        print(f"[M2] Loading cached dataset from {cache_path}")
        all_samples = torch.load(cache_path)
    else:
        if dataset_name == "versaprm":
            loader = VersaPRMLoader(data_dir=config.get("data.data_dir"))
            domains = config.get("data.domains", ["math", "code", "medical", "general"])
            samples_per_domain = config.get("data.samples_per_domain", 500)
            max_length = config.get("data.max_length", 512)

            all_samples = []
            total_cots = 0
            total_steps = 0
            for domain in tqdm(domains, desc="[M2] Loading domains"):
                domain_samples = loader.load_domain(domain)[:samples_per_domain]
                total_cots += len(domain_samples)
                for sample in tqdm(
                    domain_samples,
                    desc=f"  Tokenizing {domain}",
                    leave=False,
                ):
                    question = sample.get("question", "")
                    steps = sample.get("steps", [])
                    labels = sample.get("labels", [])
                    total_steps += len(steps)
                    for step_text, label in zip(steps, labels):
                        text = f"{question}\n{step_text}"
                        encoded = tokenizer(
                            text,
                            padding=False,
                            truncation=False,
                            max_length=None,
                            return_tensors=None,
                        )
                        # Filter: drop steps exceeding max_length.
                        # Truncation introduces overfitting to incomplete steps.
                        if len(encoded["input_ids"]) > max_length:
                            continue
                        all_samples.append(
                            {
                                "input_ids": torch.tensor(
                                    encoded["input_ids"], dtype=torch.long
                                ),
                                "attention_mask": torch.tensor(
                                    encoded["attention_mask"], dtype=torch.long
                                ),
                                "label": float(label),
                            }
                        )
            print(
                f"[M2] Loaded {total_cots} CoT samples "
                f"({total_steps} steps) from {len(domains)} domains"
            )
        else:
            loader = PRM800KLoader(
                data_dir=config.get("data.data_dir"),
                split=config.get("data.split", "train"),
            )

            print("[M2] Loading PRM800K raw samples...")
            raw_samples = loader.load()
            print(f"[M2] Loaded {len(raw_samples)} CoT samples")

            try:
                all_samples = loader.build_step_dataset(
                    tokenizer=tokenizer,
                    max_length=config.get("data.max_length", 512),
                )
            except FileNotFoundError as e:
                print(f"[ERROR] Data not found: {e}")
                print("Please download PRM800K data to the specified data_dir.")
                return
        torch.save(all_samples, cache_path)
        print(f"[M2] Cached dataset to {cache_path}")

    print(f"[M2] Total step-level samples: {len(all_samples)}")

    # Train/val split
    val_ratio = config.get("data.val_split", 0.1)
    n_val = int(len(all_samples) * val_ratio)
    train_samples = all_samples[n_val:]
    val_samples = all_samples[:n_val]
    print(
        f"[M2] Train/val split: {len(train_samples)} train, "
        f"{len(val_samples)} val ({val_ratio:.0%})"
    )

    num_workers = config.get("hardware.num_workers", 0)
    if sys.platform == "win32" and num_workers > 0:
        print(
            f"[WARN] Windows detected: forcing num_workers=0 "
            f"(config had {num_workers}). DataLoader with num_workers>0 "
            f"is ~50x slower on Windows due to spawn overhead."
        )
        num_workers = 0
    train_loader = DataLoader(
        train_samples,
        batch_size=config.get("training.batch_size", 32),
        shuffle=True,
        collate_fn=collate_step_batch,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=(num_workers > 0),
    )
    val_loader = DataLoader(
        val_samples,
        batch_size=config.get("evaluation.eval_batch_size", 64),
        shuffle=False,
        collate_fn=collate_step_batch,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=(num_workers > 0),
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=config.get("training.learning_rate", 1e-4),
        weight_decay=config.get("training.weight_decay", 0.01),
    )

    # Scheduler
    scheduler_name = config.get("training.scheduler", "constant")
    warmup_steps = config.get("training.warmup_steps", 0)
    num_epochs = config.get("training.num_epochs", 3)
    total_steps = num_epochs * len(train_loader)

    if scheduler_name == "cosine":
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    elif scheduler_name == "linear":
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    else:
        scheduler = get_constant_schedule(optimizer)

    print(
        f"[M2] Training for {num_epochs} epochs, {len(train_samples)} samples, "
        f"scheduler={scheduler_name}, warmup={warmup_steps}"
    )

    global_step = 0
    best_val_loss = float("inf")
    start_epoch = 0
    skip_batches = 0
    last_best_ckpt = None
    epoch_ckpts = []

    if args.resume and Path(args.resume).exists():
        from fclprm.models.checkpoint import load_checkpoint
        print(f"[M2] Resuming from checkpoint: {args.resume}")
        meta = load_checkpoint(args.resume, model, optimizer)
        global_step = meta["round_num"]
        start_epoch = global_step // len(train_loader)
        skip_batches = global_step % len(train_loader)
        print(
            f"[M2] Resumed at step {global_step}, epoch {start_epoch + 1}, "
            f"skipping {skip_batches} batches"
        )
        if scheduler is not None and global_step > 0:
            for _ in range(global_step):
                scheduler.step()
            print(f"[M2] Scheduler fast-forwarded to step {global_step}")

    for epoch in range(start_epoch, num_epochs):
        global _interrupted
        if _interrupted:
            print(f"\n  [STOP] Interrupted before epoch {epoch + 1}")
            _save_training_snapshot(
                model, optimizer, global_step, epoch, best_val_loss,
                checkpoint_dir, config,
            )
            break

        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"[M2] Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        )
        for batch in pbar:
            if _interrupted:
                break
            # Skip batches already trained before resume
            if epoch == start_epoch and num_batches <= skip_batches:
                num_batches += 1
                continue

            num_batches += 1
            global_step += 1

            # Periodic save every save_every steps
            if save_every > 0 and global_step % save_every == 0:
                step_ckpt = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    round_num=global_step,
                    client_id=-1,
                    milestone="M2",
                    save_dir=checkpoint_dir,
                    device=device,
                )
                print(f"  [CKPT] Saved periodic checkpoint at step {global_step}: {Path(step_ckpt).name}")

            input_ids = batch["input_ids"].to(
                device, non_blocking=device.type == "cuda"
            )
            attention_mask = batch["attention_mask"].to(
                device, non_blocking=device.type == "cuda"
            )
            labels = batch["labels"].to(device, non_blocking=device.type == "cuda")

            optimizer.zero_grad()
            with autocast(device_type=device.type, enabled=use_amp):
                predictions = model(input_ids, attention_mask)
                loss = F.mse_loss(predictions, labels)

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
            else:
                loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                config.get("training.max_grad_norm", 1.0),
            )
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            if scheduler is not None:
                scheduler.step()

            epoch_loss += loss.item()

            current_lr = (
                scheduler.get_last_lr()[0]
                if scheduler is not None
                else optimizer.param_groups[0]["lr"]
            )
            if num_batches % log_interval == 0:
                pbar.set_postfix(
                    {
                        "loss": f"{loss.item():.4f}",
                        "step": global_step,
                        "lr": f"{current_lr:.2e}",
                    }
                )

            if global_step % config.get("evaluation.eval_every", 500) == 0:
                val_metrics = evaluate(model, val_loader, device)
                current_lr = (
                    scheduler.get_last_lr()[0]
                    if scheduler is not None
                    else optimizer.param_groups[0]["lr"]
                )
                print(
                    f"  Step {global_step}: "
                    f"train_loss={loss.item():.4f}, "
                    f"val_loss={val_metrics['loss']:.4f}, "
                    f"val_acc={val_metrics['accuracy']:.4f}, "
                    f"val_f1={val_metrics['f1']:.4f}, "
                    f"lr={current_lr:.2e}"
                )
                logger.log(
                    milestone="M2",
                    config_hash=config.hash(),
                    metrics={
                        "step": global_step,
                        "epoch": epoch + 1,
                        "train_loss": loss.item(),
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                        "val_f1": val_metrics["f1"],
                        "val_auc": val_metrics["auc"],
                        "learning_rate": current_lr,
                    },
                )

                if val_metrics["loss"] < best_val_loss:
                    best_val_loss = val_metrics["loss"]
                    if last_best_ckpt is not None and Path(last_best_ckpt).exists():
                        Path(last_best_ckpt).unlink()
                        print(f"  [CKPT] Removed old best: {Path(last_best_ckpt).name}")
                    last_best_ckpt = save_checkpoint(
                        model=model,
                        optimizer=optimizer,
                        round_num=global_step,
                        client_id=-1,
                        milestone="M2",
                        save_dir=checkpoint_dir,
                    )
                    print(f"  [CKPT] Saved new best (val_loss={best_val_loss:.4f}): {Path(last_best_ckpt).name}")

        # Periodic save every save_every steps regardless of eval
        effective_batches = (
            num_batches - skip_batches
            if epoch == start_epoch and skip_batches > 0
            else num_batches
        )
        avg_loss = epoch_loss / max(effective_batches, 1)
        current_lr = (
            scheduler.get_last_lr()[0]
            if scheduler is not None
            else optimizer.param_groups[0]["lr"]
        )
        print(
            f"[M2] Epoch {epoch + 1}/{num_epochs}: "
            f"avg_train_loss={avg_loss:.4f}, lr={current_lr:.2e}"
        )

        # Epoch checkpoint: save at the end of every epoch, keep last 3
        epoch_ckpt = save_checkpoint(
            model=model,
            optimizer=optimizer,
            round_num=epoch + 1,
            client_id=-1,
            milestone="M2",
            save_dir=checkpoint_dir,
        )
        epoch_ckpts.append(epoch_ckpt)
        if len(epoch_ckpts) > 3:
            old_ckpt = epoch_ckpts.pop(0)
            if Path(old_ckpt).exists():
                Path(old_ckpt).unlink()
                print(f"  [CKPT] Removed old epoch checkpoint: {Path(old_ckpt).name}")
        print(f"  [CKPT] Saved epoch {epoch + 1} checkpoint: {Path(epoch_ckpt).name}")

        # ── Save training snapshot for crash recovery ─────────────────
        _save_training_snapshot(
            model, optimizer, global_step, epoch,
            best_val_loss, checkpoint_dir, config,
        )

    # Save final checkpoint at end of training
    final_ckpt = save_checkpoint(
        model=model,
        optimizer=optimizer,
        round_num=global_step,
        client_id=-1,
        milestone="M2",
        save_dir=checkpoint_dir,
    )
    print(f"[M2] Saved final checkpoint: {Path(final_ckpt).name}")
    print("[M2] Training complete.")


def evaluate(model, dataloader, device: str) -> dict:
    """Evaluate model on validation set.

    Returns:
        Dict with keys: loss, accuracy, f1, auc.
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="[M2] Evaluating", leave=False, unit="batch"):
            input_ids = batch["input_ids"].to(
                device, non_blocking=device.type == "cuda"
            )
            attention_mask = batch["attention_mask"].to(
                device, non_blocking=device.type == "cuda"
            )
            labels = batch["labels"].to(device, non_blocking=device.type == "cuda")

            with autocast(device_type=device.type, enabled=device.type == "cuda"):
                predictions = model(input_ids, attention_mask)
                loss = F.mse_loss(predictions, labels)
            total_loss += loss.item()
            num_batches += 1

            all_preds.extend(predictions.cpu().numpy().tolist())
            all_labels.extend(labels.cpu().numpy().tolist())

    import numpy as np

    preds = np.array(all_preds)
    labels = np.array(all_labels)

    # Binary classification with threshold 0.5
    pred_labels = (preds >= 0.5).astype(int)
    true_labels = (labels >= 0.5).astype(int)

    accuracy = accuracy_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels, zero_division=0.0)

    # AUC requires both classes present
    if len(np.unique(true_labels)) > 1:
        auc = roc_auc_score(true_labels, preds)
    else:
        auc = float("nan")

    return {
        "loss": total_loss / max(num_batches, 1),
        "accuracy": accuracy,
        "f1": f1,
        "auc": auc,
    }


if __name__ == "__main__":
    main()
