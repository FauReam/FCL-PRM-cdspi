#!/usr/bin/env python3
"""M2: Train centralized PRM baseline on PRM800K.

Usage:
    python scripts/train_centralized_prm.py --config configs/m2_pythia_1b.yaml
"""

import argparse
import hashlib
import sys
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Train centralized PRM baseline")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
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

    log_dir = config.get("logging.log_dir", "./logs")
    checkpoint_dir = config.get("logging.checkpoint_dir", "./checkpoints")
    save_every = config.get("logging.save_every", 0)

    logger = ExperimentLogger(
        log_dir=log_dir,
        experiment_id=config.get("experiment.name", "m2_baseline"),
    )

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

    try:
        backbone = _load_hf_asset(
            AutoModel.from_pretrained,
            config.get("model.backbone"),
            dtype=torch.float32,
        )
    except OSError as e:
        print(f"[ERROR] Failed to load backbone for '{config.get('model.backbone')}'.")
        print(f"  {e}")
        print("  Please ensure the model name is correct and you have internet access,")
        print(
            "  or download the model locally and set local_files_only=True in config."
        )
        return
    model = StepRewardModel(
        backbone=backbone,
        head_dim=config.get("model.prm_head_dim", 256),
    )
    model.to(device)

    if device.type == "cuda" and hasattr(torch, "compile") and sys.platform != "win32":
        print("[M2] Enabling torch.compile for faster training on RTX 4070...")
        model = torch.compile(model, mode="max-autotune")
    elif device.type == "cuda" and sys.platform == "win32":
        print("[M2] Skipping torch.compile on Windows (Triton unavailable).")

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
                            truncation=True,
                            max_length=max_length,
                            return_tensors=None,
                        )
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
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        pbar = tqdm(
            train_loader,
            desc=f"[M2] Epoch {epoch + 1}/{num_epochs}",
            unit="batch",
        )
        for batch in pbar:
            num_batches += 1
            global_step += 1

            # Skip batches already trained before resume
            if epoch == start_epoch and num_batches <= skip_batches:
                continue

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
            if num_batches % 10 == 0:
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
