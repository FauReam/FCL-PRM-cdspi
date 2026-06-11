#!/usr/bin/env python3
"""M3-M6: Federated PRM simulation main entry point.

Usage:
    python scripts/run_federated.py --config configs/m3_naive_fedavg.yaml
    python scripts/run_federated.py --config configs/m4_anchor_prm.yaml

Stdout/stderr are automatically mirrored to a timestamped log file under
logging.log_dir so you do not need shell pipes like Tee-Object.
"""

import argparse
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from fclprm.data.versa_loader import VersaPRMLoader
from fclprm.federated.simulator import FederatedSimulator
from fclprm.models.base_wrapper import StepRewardModel
from fclprm.utils.config import ExperimentConfig
from fclprm.utils.logging import ExperimentLogger
from fclprm.utils.seed import set_seed


def _save_global_checkpoint(model, round_num: int, milestone: str, checkpoint_dir: str, device: str) -> None:
    """Save global model checkpoint after each round (CPU-only, no GPU memory spike).

    checkpoint_round is 1-based: round_num=0 (Round 1) -> r1.pt.
    This makes the filename directly reflect how many rounds have been completed.
    """
    import gc

    checkpoint_round = round_num + 1
    save_path = Path(checkpoint_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f"model_m{milestone}_r{checkpoint_round}_c-1.pt"
    filepath = save_path / filename
    # Move all tensors to CPU before saving to avoid holding GPU memory during I/O
    state_items = list(model.state_dict().items())
    cpu_state = {}
    for k, v in tqdm(state_items, desc=f"  [Checkpoint] Moving round {checkpoint_round} to CPU", leave=False):
        cpu_state[k] = v.cpu()
    torch.save(
        {
            "model_state_dict": cpu_state,
            "round_num": checkpoint_round,
            "client_id": -1,
            "milestone": milestone,
        },
        filepath,
    )
    # Explicitly free the CPU-side temporary state dict and flush GPU cache
    # so that checkpoint I/O does not linger in memory.
    del cpu_state
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"  [Checkpoint] Saved round {checkpoint_round} to {filepath}")


def _save_client_checkpoint(client_model, client_id: int, round_num: int, milestone: str, checkpoint_dir: str, device: str) -> None:
    """Save a single client model checkpoint for post-hoc CD-SPI evaluation.

    Client checkpoints are small (only head parameters are trainable; backbone
    is shared and frozen).  Saving them enables事后 (post-hoc) verification of
    cross-client embedding divergence via eval_federated.py.
    """
    import gc

    checkpoint_round = round_num + 1
    # Store client checkpoints in a sub-directory to avoid cluttering global ckpts
    client_dir = Path(checkpoint_dir) / "clients"
    client_dir.mkdir(parents=True, exist_ok=True)
    filename = f"model_m{milestone}_r{checkpoint_round}_c{client_id}.pt"
    filepath = client_dir / filename

    cpu_state = {}
    for k, v in client_model.state_dict().items():
        cpu_state[k] = v.cpu()
    torch.save(
        {
            "model_state_dict": cpu_state,
            "round_num": checkpoint_round,
            "client_id": client_id,
            "milestone": milestone,
        },
        filepath,
    )
    del cpu_state
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"  [Checkpoint] Saved client {client_id} round {checkpoint_round} to {filepath}")


def _find_latest_checkpoint(checkpoint_dir: str, milestone: str):
    """Scan checkpoint_dir for the latest global model checkpoint.

    Returns:
        (round_num, filepath) or (None, None) if no checkpoint found.
        round_num is 1-based: r1.pt means Round 1 has been completed.
        For backward compatibility, legacy r0.pt (old 0-based naming) is
        treated as "1 round completed" since the old code saved r0 after
        finishing Round 1.
    """
    import re

    save_path = Path(checkpoint_dir)
    if not save_path.exists():
        return None, None
    pattern = re.compile(rf"model_m{re.escape(milestone)}_r(\d+)_c-1\.pt")
    checkpoints = []
    for f in save_path.iterdir():
        match = pattern.match(f.name)
        if match:
            checkpoints.append((int(match.group(1)), f))
    if not checkpoints:
        return None, None

    return max(checkpoints, key=lambda x: x[0])


class Tee:
    """Mirror stdout to both terminal and a log file."""

    def __init__(self, log_path: Path) -> None:
        self.terminal = sys.stdout
        self.log_file = log_path.open("w", encoding="utf-8")
        self._file_buf = ""

    def write(self, message: str) -> None:
        # Terminal always sees everything (handles \r correctly)
        self.terminal.write(message)
        self.terminal.flush()

        # For the log file: accumulate; on \n write the complete line.
        # tqdm uses \r to overwrite the same line — we keep only the last
        # segment so the log file does not get cluttered with repeats.
        self._file_buf += message
        if "\r" in message and "\n" not in message:
            self._file_buf = self._file_buf.rsplit("\r", 1)[-1]
        elif "\n" in message:
            parts = self._file_buf.split("\n")
            for part in parts[:-1]:
                self.log_file.write(part + "\n")
            self._file_buf = parts[-1]
            self.log_file.flush()

    def flush(self) -> None:
        self.terminal.flush()
        if self._file_buf:
            self.log_file.write(self._file_buf + "\n")
            self._file_buf = ""
            self.log_file.flush()

    def isatty(self) -> bool:
        return self.terminal.isatty()

    def close(self) -> None:
        if self._file_buf:
            self.log_file.write(self._file_buf + "\n")
        self.log_file.close()


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run federated PRM simulation")
    parser.add_argument(
        "--config", type=str, required=True, help="Path to YAML config file"
    )
    args = parser.parse_args()

    config = ExperimentConfig(args.config)
    set_seed(config.get("experiment.seed", 42))

    device = config.get(
        "hardware.device", "cuda" if torch.cuda.is_available() else "cpu"
    )

    log_dir = Path(config.get("logging.log_dir", "./logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Mirror stdout to a timestamped log file so PowerShell pipes are unnecessary
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{config.get('experiment.name', 'run')}_{timestamp}.log"
    tee = Tee(log_path)
    sys.stdout = tee
    print(f"[INFO] Logging mirrored to: {log_path}")

    logger = ExperimentLogger(
        log_dir=log_dir,
        experiment_id=config.get("experiment.name", "federated_prm"),
    )

    print(
        f"[{config.get('experiment.milestone')}] Initializing model: {config.get('model.backbone')}"
    )
    try:
        with tqdm(total=1, desc="Loading tokenizer", leave=False) as pbar:
            tokenizer = _load_hf_asset(
                AutoTokenizer.from_pretrained, config.get("model.backbone")
            )
            pbar.update(1)
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

    # Load dtype: BF16 saves ~33% memory vs FP32; especially important for full FT.
    freeze_backbone = config.get("model.freeze_backbone", True)
    load_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    try:
        with tqdm(total=1, desc="Loading backbone", leave=False) as pbar:
            backbone = _load_hf_asset(
                AutoModel.from_pretrained,
                config.get("model.backbone"),
                dtype=load_dtype,
            )
            pbar.update(1)
    except OSError as e:
        print(f"[ERROR] Failed to load backbone for '{config.get('model.backbone')}'.")
        print(f"  {e}")
        print("  Please ensure the model name is correct and you have internet access,")
        print(
            "  or download the model locally and set local_files_only=True in config."
        )
        return
    attnres_config = config.get("model.attnres", None)
    if attnres_config is not None:
        print(f"[INFO] Block AttnRes enabled: {attnres_config.get('num_blocks', 8)} blocks")
    with tqdm(total=1, desc="Building global model", leave=False) as pbar:
        global_model = StepRewardModel(
            backbone=backbone,
            head_dim=config.get("model.prm_head_dim", 256),
            freeze_backbone=freeze_backbone,
            attnres=attnres_config,
        )
        pbar.update(1)

    # torch.compile speeds up forward passes.
    # When AttnRes is enabled, compile the entire model after AttnRes wrapping.
    if device == "cuda":
        compile_mode = "reduce-overhead"
        global_model = torch.compile(global_model, mode=compile_mode)
        print(f"[INFO] model compiled with torch.compile ({compile_mode})")

    mode = "head-only" if freeze_backbone else "full-parameter"
    extra = " + AttnRes" if attnres_config is not None else ""
    print(f"[INFO] Training mode: {mode}{extra} ({load_dtype})")

    # Resume from latest checkpoint if available
    checkpoint_dir = config.get("logging.checkpoint_dir", "./checkpoints")
    milestone = config.get("experiment.milestone", "M3")
    latest_round, latest_path = _find_latest_checkpoint(checkpoint_dir, milestone)
    start_round = 0
    if latest_round is not None and latest_path is not None:
        print(f"[INFO] Resuming from checkpoint: {latest_path} (completed {latest_round} rounds)")
        state_items = list(global_model.state_dict().keys())
        with tqdm(total=1, desc="Loading checkpoint", leave=False) as pbar:
            checkpoint = torch.load(str(latest_path), map_location="cpu")
            global_model.load_state_dict(checkpoint["model_state_dict"])
            pbar.update(1)
        start_round = latest_round
        print(f"[INFO] Will start from round {start_round + 1}")

    print(
        f"[{config.get('experiment.milestone')}] Loading data from: {config.get('data.data_dir')}"
    )
    versa_loader = VersaPRMLoader(
        data_dir=config.get("data.data_dir"),
    )

    try:
        versa_loader.load()
    except FileNotFoundError as e:
        print(f"[ERROR] Data not found: {e}")
        print("Please download VersaPRM data to the specified data_dir.")
        return

    # Build step-level datasets for each client domain
    num_clients = config.get("federated.num_clients", 4)
    domains = config.get("data.domains", ["math", "code", "medical", "general"])
    max_length = config.get("data.max_length", 512)
    samples_per_client = config.get("data.samples_per_client", 5000)

    client_data = []
    anchor_candidates = []  # (domain, step_text) for Anchor-PRM selection
    for i in tqdm(range(num_clients), desc="Clients", position=0, leave=True):
        domain = domains[i % len(domains)]
        domain_samples = versa_loader.load_domain(domain)

        # Tokenize steps with progress bar
        step_samples = []
        sample_iter = domain_samples[:samples_per_client]
        for sample in tqdm(
            sample_iter,
            desc=f"  Tokenizing {domain}",
            total=len(sample_iter),
            position=1,
            leave=False,
        ):
            question = sample.get("question", "")
            steps = sample.get("steps", [])
            labels = sample.get("labels", [])

            if len(steps) != len(labels):
                raise ValueError(
                    f"steps/labels length mismatch in domain '{domain}': "
                    f"{len(steps)} steps vs {len(labels)} labels"
                )
            for step_text, label in zip(steps, labels):
                text = f"{question}\n{step_text}"
                # Collect raw text for anchor step selection (Anchor-PRM only)
                anchor_candidates.append((domain, text))
                encoded = tokenizer(
                    text,
                    padding=False,
                    truncation=True,
                    max_length=max_length,
                    return_tensors=None,
                )
                step_samples.append(
                    {
                        "input_ids": torch.tensor(encoded["input_ids"]),
                        "attention_mask": torch.tensor(encoded["attention_mask"]),
                        "label": float(label),
                    }
                )

        client_data.append(step_samples)
        tqdm.write(f"  Client {i} ({domain}): {len(step_samples)} step samples")

    print(f"[{config.get('experiment.milestone')}] Starting federated simulation")
    print(f"  Rounds: {config.get('federated.num_rounds')}")
    print(f"  Clients: {num_clients}")
    print(f"  Aggregation: {config.get('federated.aggregation')}")

    # Anchor-PRM aggregation needs a small set of shared anchor steps to
    # extract per-client head embeddings. The steps can be provided explicitly
    # via `anchor.steps` in YAML, or sampled dynamically from client data using
    # `anchor.anchor_selection` (random / diverse / domain-balanced).
    aggregation_rule = config.get("federated.aggregation", "fedavg")
    anchor_inputs = None
    anchor_steps = None
    if aggregation_rule == "anchor_prm":
        anchor_steps_config = config.get("anchor.steps", None)
        if anchor_steps_config is not None:
            # Explicit anchor steps from config
            anchor_steps = anchor_steps_config
        else:
            # Dynamic selection from client data
            num_anchor_steps = config.get("anchor.num_anchor_steps", 100)
            anchor_selection = config.get("anchor.anchor_selection", "diverse")
            anchor_seed = config.get("experiment.seed", 42)
            rng = random.Random(anchor_seed)

            if num_anchor_steps > len(anchor_candidates):
                print(
                    f"[WARN] anchor.num_anchor_steps ({num_anchor_steps}) exceeds "
                    f"available candidates ({len(anchor_candidates)}). Using all."
                )
                num_anchor_steps = len(anchor_candidates)

            if anchor_selection == "random":
                rng.shuffle(anchor_candidates)
                anchor_steps = [text for (_, text) in anchor_candidates[:num_anchor_steps]]
            else:
                # diverse / domain-balanced: stratified sampling per domain
                by_domain = defaultdict(list)
                for domain, text in anchor_candidates:
                    by_domain[domain].append(text)

                selected = []
                domain_list = list(by_domain.keys())
                rng.shuffle(domain_list)
                per_domain = num_anchor_steps // len(domain_list)
                remainder = num_anchor_steps % len(domain_list)

                for idx, domain in enumerate(domain_list):
                    n = per_domain + (1 if idx < remainder else 0)
                    domain_texts = by_domain[domain][:]
                    rng.shuffle(domain_texts)
                    selected.extend(domain_texts[:n])

                rng.shuffle(selected)
                anchor_steps = selected[:num_anchor_steps]

            print(
                f"  Anchor selection: {anchor_selection} | "
                f"requested={config.get('anchor.num_anchor_steps', 100)} | "
                f"selected={len(anchor_steps)}"
            )

        with tqdm(total=1, desc="Encoding anchor steps", leave=False) as pbar:
            anchor_encoded = tokenizer(
                anchor_steps,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            anchor_inputs = {
                "input_ids": anchor_encoded["input_ids"],
                "attention_mask": anchor_encoded["attention_mask"],
            }
            pbar.update(1)
        print(f"  Anchor steps: {len(anchor_steps)}")

        # alignment_weight is part of the config for future extensions
        # (e.g., soft alignment loss); current permutation-rebasin hard
        # alignment does not use it.
        alignment_weight = config.get("anchor.alignment_weight", None)
        if alignment_weight is not None:
            print(f"  Anchor alignment_weight: {alignment_weight} (recorded, not used)")

    # DP configuration
    dp_enabled = config.get("dp.enabled", False)
    dp_epsilon = config.get("dp.epsilon", 4.0)
    dp_delta = config.get("dp.delta", 1e-5)
    dp_max_grad_norm = config.get("dp.max_grad_norm", 1.0)
    if dp_enabled:
        print(
            f"  DP-SGD: enabled (epsilon={dp_epsilon}, delta={dp_delta}, max_grad_norm={dp_max_grad_norm})"
        )

    simulator = FederatedSimulator(
        num_clients=num_clients,
        num_rounds=config.get("federated.num_rounds", 50),
        global_model=global_model,
        client_data=client_data,
        aggregation_rule=aggregation_rule,
        seed=config.get("experiment.seed", 42),
        anchor_inputs=anchor_inputs,
        anchor_steps=anchor_steps,
        dp_enabled=dp_enabled,
        dp_epsilon=dp_epsilon,
        dp_delta=dp_delta,
        compute_cd_spi=config.get("evaluation.compute_cd_spi", False),
        eval_every=config.get("evaluation.eval_every", 1),
        participation_rate=config.get("federated.participation_rate", 1.0),
        skip_eval=config.get("evaluation.skip_eval", False),
    )

    save_every = config.get("logging.save_every", 10)

    save_client_ckpts = config.get("logging.save_client_checkpoints", False)
    if save_client_ckpts:
        print("  Client checkpoints: enabled (saved per round alongside global ckpt)")

    def _on_round_end(round_num: int, model: torch.nn.Module) -> None:
        if (round_num + 1) % save_every == 0:
            _save_global_checkpoint(
                model,
                round_num=round_num,
                milestone=milestone,
                checkpoint_dir=checkpoint_dir,
                device=device,
            )
            if save_client_ckpts:
                for client in simulator.clients:
                    _save_client_checkpoint(
                        client.model,
                        client_id=client.client_id,
                        round_num=round_num,
                        milestone=milestone,
                        checkpoint_dir=checkpoint_dir,
                        device=device,
                    )

    results = simulator.run(
        local_epochs=config.get("federated.local_epochs", 2),
        local_batch_size=config.get("federated.local_batch_size", 32),
        local_lr=config.get("federated.local_learning_rate", 1e-4),
        device=device,
        max_grad_norm=dp_max_grad_norm,
        log_interval=config.get("training.log_interval", 5),
        start_round=start_round,
        on_round_end=_on_round_end,
        scheduler=config.get("training.scheduler", None),
        max_steps_per_epoch=config.get("training.max_steps_per_epoch", None),
        num_workers=config.get("hardware.num_workers", 0),
    )

    # Save final model to disk regardless of save_every interval
    final_round = results['num_rounds'] - 1
    _save_global_checkpoint(
        results['final_model'],
        round_num=final_round,
        milestone=milestone,
        checkpoint_dir=checkpoint_dir,
        device=device,
    )

    print(f"[{config.get('experiment.milestone')}] Simulation complete")
    print(f"  Final avg loss: {results['history'][-1]['avg_loss']:.4f}")
    print(f"  Total time: {results.get('total_sec', 0):.1f}s")

    # Log per-round metrics with real-time rates
    for record in results["history"]:
        metrics = {
            "round": record["round"],
            "avg_loss": record["avg_loss"],
            "client_losses": record["client_losses"],
            "round_sec": record.get("round_sec", 0),
            "steps_per_sec": record.get("steps_per_sec", 0),
            "samples_per_sec": record.get("samples_per_sec", 0),
        }
        if "per_domain_mse" in record:
            metrics["per_domain_mse"] = record["per_domain_mse"]
        if "cd_spi" in record:
            metrics["cd_spi"] = record["cd_spi"]
        logger.log(
            milestone=config.get("experiment.milestone"),
            config_hash=config.hash(),
            metrics=metrics,
        )

    final_record = results["history"][-1]
    final_metrics = {
        "final_loss": final_record["avg_loss"],
        "num_rounds": results["num_rounds"],
        "total_sec": results.get("total_sec", 0),
    }
    if "per_domain_mse" in final_record:
        final_metrics["per_domain_mse"] = final_record["per_domain_mse"]
    if "cd_spi" in final_record:
        final_metrics["cd_spi"] = final_record["cd_spi"]
    logger.log(
        milestone=config.get("experiment.milestone"),
        config_hash=config.hash(),
        metrics=final_metrics,
    )

    # Restore stdout and close log file
    sys.stdout = tee.terminal
    tee.close()
    print(f"[INFO] Log saved to: {log_path}")


if __name__ == "__main__":
    main()
