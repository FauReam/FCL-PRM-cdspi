"""Single-machine multi-process federated simulation scheduler.

Safety features:
  - SIGINT handler: graceful shutdown on Ctrl+C (saves partial results)
  - Auto-save history: per-round JSONL dump for crash recovery
  - Per-client error isolation: one client crash logs error but continues
  - Structured error logging: all exceptions captured with traceback
"""

import json
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Callable, Optional

import torch
import torch.nn as nn

from fclprm.federated.client import FederatedClient
from fclprm.federated.server import FederatedServer
from fclprm.utils.seed import set_seed


class FederatedSimulator:
    """Simulate multiple federated clients on a single machine.

    Uses sequential client execution (not true multiprocessing) for simplicity.
    Each client is trained one after another on the same device.
    """

    def __init__(
        self,
        num_clients: int,
        num_rounds: int,
        global_model: nn.Module,
        client_data: list[list[dict]],
        aggregation_rule: str = "fedavg",
        seed: int = 42,
        anchor_inputs: dict | None = None,
        anchor_steps: list[str] | None = None,
        dp_enabled: bool = False,
        dp_epsilon: float = 4.0,
        dp_delta: float = 1e-5,
        compute_cd_spi: bool = False,
        eval_every: int = 1,
        participation_rate: float = 1.0,
        skip_eval: bool = False,
        eval_ood: bool = False,
        ood_domains: list[str] | None = None,
        eval_label_noise: bool = False,
        label_noise_ratios: list[float] | None = None,
        compute_symmetrical_cd_spi: bool = False,
        work_dir: str | None = None,
    ) -> None:
        """Initialize simulator.

        Args:
            num_clients: Number of simulated clients.
            num_rounds: Total federated training rounds.
            global_model: Initial global model.
            client_data: List of data splits, one per client.
            aggregation_rule: Aggregation strategy name.
            seed: Random seed.
            anchor_inputs: Optional dict with pre-tokenized anchor inputs:
                {"input_ids": LongTensor (N, L), "attention_mask": LongTensor
                (N, L)}. Required when aggregation_rule == "anchor_prm".
            anchor_steps: Anchor step texts (logged for reproducibility,
                not used in math).
            dp_enabled: Whether to enable DP-SGD on clients.
            dp_epsilon: Privacy budget epsilon (if DP enabled).
            dp_delta: Privacy budget delta (if DP enabled).
            compute_cd_spi: Whether to compute CD-SPI metric each round
                (asymmetrical: using head embeddings).
            eval_every: Evaluate per-domain MSE every N rounds (1 = every round).
            participation_rate: Fraction of clients participating each round
                (1.0 = all clients, <1.0 = random subset).
            eval_ood: Whether to run cross-domain OOD evaluation.
            ood_domains: Domain labels for each client (for OOD eval).
            eval_label_noise: Whether to evaluate under label perturbation.
            label_noise_ratios: Label flip ratios for noise robustness test.
            compute_symmetrical_cd_spi: Whether to compute symmetrical CD-SPI
                (using backbone penultimate layer) and CKA as independent
                cross-validation. Addresses expert panel P0* requirement.
        """
        self.num_clients = num_clients
        self.num_rounds = num_rounds
        self.global_model = global_model
        self.client_data = client_data
        self.aggregation_rule = aggregation_rule
        self.seed = seed
        self.anchor_inputs = anchor_inputs
        self.anchor_steps = anchor_steps or []
        self.dp_enabled = dp_enabled
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta
        self.compute_cd_spi = compute_cd_spi
        self.eval_every = max(eval_every, 1)
        self.participation_rate = max(0.0, min(1.0, participation_rate))
        self.skip_eval = skip_eval
        self.eval_ood = eval_ood
        self.ood_domains = ood_domains or []
        self.eval_label_noise = eval_label_noise
        self.label_noise_ratios = label_noise_ratios or [0.0, 0.1, 0.2]
        self.compute_symmetrical_cd_spi = compute_symmetrical_cd_spi
        self.work_dir = work_dir
        self._interrupted = False
        self._current_round = 0
        self._history_path: Path | None = None
        self._crash_dir: Path | None = None
        if work_dir:
            wd = Path(work_dir)
            wd.mkdir(parents=True, exist_ok=True)
            self._history_path = wd / "history.json"
            self._crash_dir = wd / "crashes"
            self._crash_dir.mkdir(parents=True, exist_ok=True)

        if aggregation_rule == "anchor_prm" and anchor_inputs is None:
            raise ValueError(
                "anchor_prm aggregation requires anchor_inputs (pre-tokenized "
                "input_ids + attention_mask) to compute per-client embeddings"
            )

        self.server = FederatedServer(
            global_model=global_model,
            aggregation_rule=aggregation_rule,
            anchor_steps=self.anchor_steps,
        )

        self.clients: list[FederatedClient] = []
        for i in range(num_clients):
            client_model = (
                type(global_model)(
                    backbone=global_model.backbone,
                    head_dim=global_model.head.head_dim,
                )
                if hasattr(global_model, "backbone")
                else global_model
            )

            client = FederatedClient(
                client_id=i,
                model=client_model,
                train_data=client_data[i],
                dp_enabled=dp_enabled,
                dp_epsilon=dp_epsilon,
                dp_delta=dp_delta,
            )
            self.clients.append(client)

        # Cache model config for GPU isolation (save -> clear -> reload)
        self._backbone_name = (
            global_model.backbone.name_or_path
            if hasattr(global_model, "backbone")
            else None
        )
        self._head_dim = (
            global_model.head.head_dim
            if hasattr(global_model, "head")
            else 256
        )

        self.history: list[dict] = []

    def _eval_per_domain(self, device: str, batch_size: int = 32) -> dict:
        """Evaluate global model on each client's local data (per-domain MSE).

        Uses the current global model (post-aggregation) to run inference on
        every client's local partition. This reveals cross-domain performance
        gaps that global avg_loss would otherwise mask.
        """
        from torch.utils.data import DataLoader
        import torch.nn.functional as F
        from tqdm import tqdm

        from fclprm.data.utils import collate_step_batch

        eval_device = device
        global_model = self.server.get_global_model()
        global_model.to(eval_device)
        global_model.eval()

        metrics: dict[str, float] = {}
        with torch.no_grad():
            for client in tqdm(self.clients, desc="[Eval] Per-domain", leave=True):
                loader = DataLoader(
                    client.train_data,
                    batch_size=batch_size,
                    collate_fn=collate_step_batch,
                )
                total_mse = 0.0
                num_batches = 0
                for batch in tqdm(loader, desc=f"Client {client.client_id}", leave=False, total=len(loader)):
                    input_ids = batch["input_ids"].to(eval_device)
                    attention_mask = batch["attention_mask"].to(eval_device)
                    labels = batch["labels"].to(eval_device)
                    preds = global_model(input_ids, attention_mask)
                    mse = F.mse_loss(preds, labels)
                    total_mse += mse.item()
                    num_batches += 1
                avg_mse = total_mse / max(num_batches, 1)
                metrics[f"client_{client.client_id}_mse"] = avg_mse
                tqdm.write(f"  [Eval] Client {client.client_id} MSE={avg_mse:.4f}")
        global_model.to(device)
        return metrics

    def _extract_anchor_embeddings(
        self, client: FederatedClient, device: str
    ) -> torch.Tensor:
        """Run the (post-trained) client model over anchor inputs.

        Returns post-ReLU head features of shape (N, head_dim) on CPU,
        ready to ship to the server-side aligner.
        """
        extract_device = "cpu"
        client.model.to(extract_device)
        client.model.eval()
        input_ids = self.anchor_inputs["input_ids"].to(extract_device)
        attention_mask = self.anchor_inputs["attention_mask"].to(extract_device)
        with torch.no_grad():
            embs = client.model.get_head_embedding(input_ids, attention_mask)
        client.model.to(device)
        return embs.detach().cpu()

    def _eval_cd_spi(self, device: str) -> dict[str, float]:
        """Compute CD-SPI across clients using the current local models.

        Uses post-ReLU head-intermediate embeddings (via get_head_embedding)
        so the metric reflects cross-domain polysemy learned by client-specific
        head weights. The backbone is frozen and identical across clients, so
        backbone-only embeddings would always yield CD-SPI ~ 0.

        Returns:
            Dict with 'cd_spi_mean' and per-step details.
        """
        from tqdm import tqdm

        if self.anchor_inputs is None or not self.anchor_steps:
            return {}

        from fclprm.metrics.cd_spi import compute_cd_spi_batch

        eval_device = device
        all_client_embeddings: dict[str, list[torch.Tensor]] = {}
        for client in tqdm(self.clients, desc="[CD-SPI] Extracting", leave=True):
            client.model.to(eval_device)
            client.model.eval()
            input_ids = self.anchor_inputs["input_ids"].to(eval_device)
            attention_mask = self.anchor_inputs["attention_mask"].to(eval_device)
            with torch.no_grad():
                embs = client.model.get_head_embedding(input_ids, attention_mask)
            all_client_embeddings[str(client.client_id)] = [
                e for e in embs.detach().cpu()
            ]
            client.model.to(device)

        per_step = compute_cd_spi_batch(self.anchor_steps, all_client_embeddings)
        mean_cd_spi = sum(per_step.values()) / len(per_step) if per_step else 0.0
        return {"cd_spi_mean": mean_cd_spi, "cd_spi_per_step": per_step}

    def _eval_cd_spi_symmetrical(self, device: str) -> dict:
        """Compute symmetrical CD-SPI using backbone penultimate-layer embeddings.

        Uses get_backbone_embedding() instead of get_head_embedding() so that
        both head-only and full-FT configs are measured in the same embedding
        space (backbone penultimate layer). This eliminates the measurement
        asymmetry identified by the expert panel.

        For head-only configs (frozen backbone), symmetrical CD-SPI will be ~0
        because backbone is identical across clients. For full FT, it captures
        actual backbone divergence. The contrast between this and the
        asymmetrical head-embedding CD-SPI reveals whether head-only
        divergence is real structure or random noise.

        Also computes PCA EVR (Phase 2 of diagnostic protocol) and CKA
        (independent cross-validation).

        Returns:
            Dict with 'cd_spi_sym_mean', 'cd_spi_sym_per_step', 'pca_evr',
            and 'cka' metrics.
        """
        from tqdm import tqdm

        if self.anchor_inputs is None or not self.anchor_steps:
            return {}

        from fclprm.metrics.cd_spi import compute_cd_spi_batch, compute_pca_evr
        from fclprm.metrics.cd_spi_stats import permutation_test_cd_spi
        from fclprm.metrics.cka import compute_client_cka_matrix

        eval_device = device
        all_client_embeddings: dict[str, list[torch.Tensor]] = {}
        for client in tqdm(self.clients, desc="[CD-SPI sym] Extracting", leave=True):
            client.model.to(eval_device)
            client.model.eval()
            input_ids = self.anchor_inputs["input_ids"].to(eval_device)
            attention_mask = self.anchor_inputs["attention_mask"].to(eval_device)
            with torch.no_grad():
                # Use backbone penultimate layer for symmetrical measurement
                embs = client.model.get_backbone_embedding(input_ids, attention_mask)
            all_client_embeddings[str(client.client_id)] = [
                e for e in embs.detach().cpu()
            ]
            client.model.to(device)

        per_step = compute_cd_spi_batch(self.anchor_steps, all_client_embeddings)
        mean_cd_spi = sum(per_step.values()) / len(per_step) if per_step else 0.0

        # PCA EVR (Phase 2 diagnostic)
        # Average embeddings across steps for each client, then compute PCA
        client_avg_embs = {
            cid: torch.stack(embs).mean(dim=0)
            for cid, embs in all_client_embeddings.items()
        }
        pca_evr = compute_pca_evr(client_avg_embs)

        # Permutation test
        perm_result = permutation_test_cd_spi(client_avg_embs)

        # CKA cross-validation
        client_feature_matrix = {
            cid: torch.stack(embs)
            for cid, embs in all_client_embeddings.items()
        }
        cka_result = compute_client_cka_matrix(client_feature_matrix)

        return {
            "cd_spi_sym_mean": mean_cd_spi,
            "cd_spi_sym_per_step": per_step,
            "pca_evr": pca_evr,
            "permutation_test": {
                "p_value": perm_result.get("p_value"),
                "significant": perm_result.get("significant"),
                "effect_size": perm_result.get("effect_size"),
            },
            "cka": cka_result,
        }

    def _eval_function_space_divergence(self, device: str) -> dict:
        """Compute function-space output divergence across clients.

        Uses client reward predictions on shared anchor steps to compute
        output cosine divergence and JS divergence — complementing CD-SPI's
        parameter-space measurement with function-space validation.
        """
        from torch.utils.data import DataLoader
        from fclprm.data.utils import collate_step_batch
        from fclprm.metrics.cd_spi_stats import (
            compute_output_cosine_divergence,
            compute_js_output_divergence,
        )

        if self.anchor_inputs is None or len(self.clients) < 2:
            return {}

        # Collect predictions from each client model on anchor inputs
        client_preds: dict[str, torch.Tensor] = {}
        input_ids = self.anchor_inputs["input_ids"].to(device)
        attention_mask = self.anchor_inputs["attention_mask"].to(device)

        with torch.no_grad():
            for client in self.clients:
                client.model.to(device)
                client.model.eval()
                preds = client.model(input_ids, attention_mask)
                client_preds[str(client.client_id)] = preds.detach().cpu()

        output_cos_div = compute_output_cosine_divergence(client_preds)
        js_div = compute_js_output_divergence(client_preds)

        return {
            "output_cosine_divergence": output_cos_div,
            "output_js_divergence": js_div,
        }

    def _eval_ood_cross_domain(self, device: str, batch_size: int) -> dict:
        """Cross-domain OOD evaluation: test each client on all other domains.

        Returns per-client OOD MSE, aggregated into a single summary.
        """
        from fclprm.metrics.ood_eval import build_cross_domain_test_splits, evaluate_cross_domain

        if not self.ood_domains or len(self.ood_domains) < 2:
            return {}

        global_model = self.server.get_global_model()
        global_model.to(device)
        global_model.eval()

        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                global_model.backbone.config._name_or_path
                if hasattr(global_model.backbone.config, "_name_or_path")
                else global_model.backbone.config.name_or_path
                if hasattr(global_model.backbone.config, "name_or_path")
                else "EleutherAI/pythia-1.4b"
            )
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        except Exception:
            return {}

        ood_splits = build_cross_domain_test_splits(
            self.client_data, self.ood_domains
        )
        with torch.no_grad():
            results = evaluate_cross_domain(
                global_model, tokenizer, ood_splits,
                device=device, batch_size=batch_size,
            )
        return results

    def _eval_label_noise(self, device: str, batch_size: int) -> dict:
        """Evaluate global model under label perturbation at multiple noise levels."""
        from fclprm.metrics.ood_eval import (
            build_perturbation_test_sets,
            evaluate_label_noise_robustness,
        )

        global_model = self.server.get_global_model()
        global_model.to(device)
        global_model.eval()

        test_sets = build_perturbation_test_sets(
            self.client_data,
            flip_ratios=self.label_noise_ratios,
            seed=self.seed,
        )
        with torch.no_grad():
            results = evaluate_label_noise_robustness(
                global_model, test_sets,
                device=device, batch_size=batch_size,
            )
        return results

    # ── Safety: signal handler for graceful interrupt ──────────────────────

    def _setup_signal_handler(self) -> None:
        """Install SIGINT handler for graceful shutdown on Ctrl + C."""
        from tqdm import tqdm
        original = signal.getsignal(signal.SIGINT)

        def _handler(sig, frame):
            self._interrupted = True
            tqdm.write("\n  [SIGNAL] SIGINT received -- finishing current round, then exiting...")
            tqdm.write("  [SIGNAL] Press Ctrl + C again to force exit.")
            signal.signal(signal.SIGINT, original)

        signal.signal(signal.SIGINT, _handler)

    def _save_history_snapshot(self) -> None:
        """Save current history to disk for crash recovery."""
        from tqdm import tqdm
        if self._history_path is None or not self.history:
            return
        try:
            tmp = self._history_path.with_suffix(".tmp.json")
            with open(tmp, "w") as f:
                def _convert(v):
                    if isinstance(v, torch.Tensor):
                        return v.item()
                    if isinstance(v, dict):
                        return {k: _convert(x) for k, x in v.items()}
                    if isinstance(v, list):
                        return [_convert(x) for x in v]
                    return v
                clean = [_convert(e) for e in self.history]
                json.dump(clean, f, indent=2, ensure_ascii=False)
            tmp.replace(self._history_path)
        except Exception as e:
            tqdm.write(f"  [WARN] Failed to save history snapshot: {e}")

    def _save_crash_report(self, stage: str, error: Exception) -> None:
        """Save structured crash report for debugging."""
        from tqdm import tqdm
        if self._crash_dir is None:
            return
        try:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            report = {
                "timestamp": timestamp,
                "stage": stage,
                "round": self._current_round,
                "error_type": type(error).__name__,
                "error_message": str(error),
                "traceback": traceback.format_exc(),
            }
            log_path = self._crash_dir / f"crash_r{self._current_round}_{timestamp}.json"
            with open(log_path, "w") as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            tqdm.write(f"\n  [CRASH] {report['error_type']}: {report['error_message']}")
            tqdm.write(f"  [CRASH] Report saved to: {log_path}")
        except Exception as e2:
            tqdm.write(f"  [CRASH] Failed to save crash report: {e2}")

    def run(
        self,
        local_epochs: int = 2,
        local_batch_size: int = 32,
        local_lr: float = 1e-4,
        device: str = "cuda",
        max_grad_norm: float = 1.0,
        log_interval: int = 5,
        start_round: int = 0,
        on_round_end: Optional[Callable[[int, nn.Module], None]] = None,
        scheduler: str | None = None,
        max_steps_per_epoch: int | None = None,
        num_workers: int = 0,
    ) -> dict:
        """Run the full federated training simulation.

        Args:
            local_epochs: Number of local epochs per client per round.
            local_batch_size: Local training batch size.
            local_lr: Local learning rate.
            device: Device for training.
            max_grad_norm: Per-sample gradient clipping bound (used when DP enabled).
            log_interval: Print real-time client metrics every N batches.
            start_round: Resume from this round number (0-based).
            on_round_end: Optional callback(round_num, global_model) invoked after
                each round's server aggregation.
            scheduler: Learning rate scheduler name ("cosine", "constant", or None).
            max_steps_per_epoch: If set, limit each local epoch to this many
                batches. Useful for fast verification.
            num_workers: Number of DataLoader workers. 0 = main-process only
                (Windows-compatible); 2–4 on Linux/CUDA for pipelined prefetch.

        Returns:
            Dict of training history and final metrics.
        """
        import random

        from tqdm import tqdm

        set_seed(self.seed)
        rng = random.Random(self.seed)
        global_start = time.perf_counter()

        # Install signal handler for graceful interrupt
        self._setup_signal_handler()

        round_pbar = tqdm(
            range(start_round, self.num_rounds),
            desc="Rounds",
            initial=start_round,
            total=self.num_rounds,
            position=0,
            leave=True,
        )
        for round_num in round_pbar:
            # Check interrupt flag (from SIGINT handler)
            if self._interrupted:
                tqdm.write(f"\n  [STOP] Interrupted at round {round_num}")
                self._save_history_snapshot()
                break

            self._current_round = round_num
            round_start = time.perf_counter()
            tqdm.write(f"\n[Round {round_num + 1}/{self.num_rounds}] Starting")
            if device == "cuda":
                allocated = torch.cuda.memory_allocated() / 1024**3
                reserved = torch.cuda.memory_reserved() / 1024**3
                tqdm.write(f"  [GPU] allocated={allocated:.2f}GB reserved={reserved:.2f}GB")

            # Select participating clients based on participation_rate
            if self.participation_rate < 1.0:
                n_participate = max(1, int(self.num_clients * self.participation_rate))
                active_clients = rng.sample(self.clients, n_participate)
                active_clients.sort(key=lambda c: c.client_id)
                tqdm.write(
                    f"  [Participation] {len(active_clients)}/{self.num_clients} clients selected: "
                    f"{[c.client_id for c in active_clients]}"
                )
            else:
                active_clients = self.clients

            # 1. Broadcast global model to all clients (even non-participating
            #    ones need the latest model for eval)
            global_state = self.server.broadcast()
            for client in self.clients:
                client.model.load_state_dict(global_state)

            # 2. Local training on participating clients only
            client_updates = []
            round_losses = []
            total_client_steps = 0
            total_client_samples = 0
            client_errors = []
            for client in active_clients:
                if self._interrupted:
                    break
                tqdm.write(f"  [Round {round_num + 1}] Client {client.client_id} training...")
                try:
                    update = client.local_train(
                        num_epochs=local_epochs,
                        batch_size=local_batch_size,
                        learning_rate=local_lr,
                        device=device,
                        max_grad_norm=max_grad_norm,
                        log_interval=log_interval,
                        scheduler=scheduler,
                        max_steps_per_epoch=max_steps_per_epoch,
                        num_workers=num_workers,
                    )
                except KeyboardInterrupt:
                    tqdm.write("\n  [INTERRUPT] KeyboardInterrupt during client training")
                    self._interrupted = True
                    self._save_history_snapshot()
                    raise
                except Exception as e:
                    self._save_crash_report(
                        f"round_{round_num}_client_{client.client_id}", e
                    )
                    client_errors.append(client.client_id)
                    continue  # Skip failed client, continue with others
                if self.aggregation_rule == "anchor_prm":
                    update["anchor_embeddings"] = self._extract_anchor_embeddings(
                        client, device=device
                    )
                client_updates.append(update)
                round_losses.append(update["loss"])
                total_client_steps += update.get("num_batches", 0)
                total_client_samples += update.get("num_samples", 0)
                tqdm.write(
                    f"  [Round {round_num + 1}] Client {client.client_id} done | "
                    f"loss={update['loss']:.4f} | "
                    f"batches={update['num_batches']} | "
                    f"time={update['elapsed_sec']}s | "
                    f"steps/s={update['steps_per_sec']} | "
                    f"samples/s={update['samples_per_sec']}"
                )

                # Clear GPU cache between clients to prevent OOM accumulation
                if device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.empty_cache()

            # Warn if any clients failed; skip round if too few updates
            if client_errors:
                tqdm.write(f"  [WARN] Clients failed this round: {client_errors}")
                if len(client_updates) < 2:
                    tqdm.write(f"  [WARN] Only {len(client_updates)} client(s) succeeded, "
                               f"skipping aggregation and eval")
                    self._save_history_snapshot()
                    continue

            # 3. CD-SPI (pre-aggregation, always if requested)
            # This is cheap (only N_anchor steps forward) and runs before
            # aggregation so it reflects client-specific head weights.
            cd_spi_metrics: dict[str, float] = {}
            func_divergence: dict[str, float] = {}
            cd_spi_sym_metrics: dict = {}
            ood_results: dict = {}
            label_noise_results: dict = {}
            if self.compute_cd_spi and len(client_updates) >= 2:
                cd_spi_metrics = self._eval_cd_spi(device=device)
                if cd_spi_metrics:
                    tqdm.write(
                        f"  [CD-SPI] mean={cd_spi_metrics.get('cd_spi_mean', 0.0):.4f}"
                    )
                    per_step = cd_spi_metrics.get("cd_spi_per_step", {})
                    if per_step:
                        for step_text, val in list(per_step.items())[:3]:
                            short = (
                                step_text[:50] + "..."
                                if len(step_text) > 50
                                else step_text
                            )
                            tqdm.write(f"    {short}: {val:.4f}")

                # Function-space divergence (complements parameter-space CD-SPI)
                func_divergence = self._eval_function_space_divergence(device=device)
                if func_divergence:
                    tqdm.write(
                        f"  [Func-div] output_cos={func_divergence.get('output_cosine_divergence', 0.0):.4f} "
                        f"js={func_divergence.get('output_js_divergence', 0.0):.4f}"
                    )

                # Symmetrical CD-SPI (backbone penultimate layer) + CKA + PCA EVR
                # Independent cross-validation addressing expert panel P0*
                cd_spi_sym_metrics: dict = {}
                if self.compute_symmetrical_cd_spi and len(client_updates) >= 2:
                    cd_spi_sym_metrics = self._eval_cd_spi_symmetrical(device=device)
                    if cd_spi_sym_metrics:
                        tqdm.write(
                            f"  [CD-SPI sym] mean={cd_spi_sym_metrics.get('cd_spi_sym_mean', 0.0):.4f}"
                        )
                        pca = cd_spi_sym_metrics.get("pca_evr", {})
                        if pca:
                            tqdm.write(
                                f"  [PCA EVR] first={pca.get('evr_first', 0.0):.4f} "
                                f"interpretation={pca.get('interpretation', 'N/A')}"
                            )
                        cka = cd_spi_sym_metrics.get("cka", {})
                        if cka:
                            tqdm.write(
                                f"  [CKA] mean={cka.get('cka_mean', 0.0):.4f}"
                            )

            per_domain: dict[str, float] = {}

            if not self.skip_eval:
                # ---- GPU Isolation: Save -> Clear -> Reload for Validation ----
                temp_path = Path(f"./.temp_model_r{round_num}.pt")
                state = {
                    k: v.cpu() for k, v in self.server.get_global_model().state_dict().items()
                }
                torch.save(state, temp_path)
                del state

                # Move original models to CPU to free GPU
                self.global_model.cpu()
                for client in self.clients:
                    client.model.cpu()
                torch.cuda.empty_cache()
                tqdm.write("  [GPU] Cleared after training")

                # Re-create a fresh GPU model copy for validation
                from transformers import AutoModel
                from fclprm.models.base_wrapper import StepRewardModel

                val_backbone = AutoModel.from_pretrained(
                    self._backbone_name,
                    torch_dtype=torch.float32,
                )
                val_model = StepRewardModel(
                    backbone=val_backbone,
                    head_dim=self._head_dim,
                )
                val_model.load_state_dict(torch.load(str(temp_path), map_location=device))
                val_model.to(device)
                val_model.eval()
                # Temporarily replace server model with validation copy
                original_model = self.server.global_model
                self.server.global_model = val_model

                # NOTE: CD-SPI is already computed above (pre-aggregation, using
                # client-specific models). Do NOT recompute here with the global
                # validation copy, as that would always yield ~0.

                # Server aggregation (on CPU, only head parameters)
                self.server.global_model = original_model
                self.server.aggregate(client_updates)
                # Swap back to validation copy for eval
                self.server.global_model = val_model

                # Per-domain evaluation (post-aggregation global model)
                if (round_num + 1) % self.eval_every == 0:
                    per_domain = self._eval_per_domain(device=device, batch_size=local_batch_size)
                    domain_str = " | ".join(
                        f"c{k.split('_')[1]}={v:.4f}" for k, v in sorted(per_domain.items())
                    )
                    tqdm.write(f"  [Per-domain MSE] {domain_str}")
                else:
                    tqdm.write(f"  [Per-domain MSE] skipped (eval_every={self.eval_every})")
                    for c in self.clients:
                        per_domain[f"client_{c.client_id}_mse"] = float("nan")
                    domain_str = "skipped"

                # OOD cross-domain evaluation
                ood_results: dict = {}
                if self.eval_ood and (round_num + 1) % self.eval_every == 0:
                    ood_results = self._eval_ood_cross_domain(
                        device=device, batch_size=local_batch_size
                    )
                    if ood_results:
                        tqdm.write(f"  [OOD Cross-domain] {ood_results}")

                # Label noise robustness
                label_noise_results: dict = {}
                if self.eval_label_noise and (round_num + 1) % self.eval_every == 0:
                    label_noise_results = self._eval_label_noise(
                        device=device, batch_size=local_batch_size
                    )
                    if label_noise_results:
                        tqdm.write(f"  [Label Noise] {label_noise_results}")

                # Clean up validation copy and restore original model
                self.server.global_model = original_model
                del val_model
                del val_backbone
                torch.cuda.empty_cache()
                tqdm.write("  [GPU] Cleared after eval")

                # Move original models back to GPU for next round
                self.global_model.to(device)
                for client in self.clients:
                    client.model.to(device)
                temp_path.unlink(missing_ok=True)
            else:
                # Train-only mode: skip all eval, just aggregate
                self.server.aggregate(client_updates)
                domain_str = "skipped (train-only)"
                tqdm.write(f"  [Eval] skipped (train-only mode)")

            round_elapsed = time.perf_counter() - round_start
            avg_loss = sum(round_losses) / len(round_losses) if round_losses else 0.0
            round_steps_per_sec = total_client_steps / round_elapsed if round_elapsed > 0 else 0.0
            round_samples_per_sec = total_client_samples / round_elapsed if round_elapsed > 0 else 0.0

            history_entry: dict = {
                "round": round_num,
                "avg_loss": avg_loss,
                "client_losses": round_losses,
                "round_sec": round(round_elapsed, 2),
                "steps_per_sec": round(round_steps_per_sec, 2),
                "samples_per_sec": round(round_samples_per_sec, 2),
                "per_domain_mse": per_domain,
            }
            if cd_spi_metrics:
                history_entry["cd_spi"] = cd_spi_metrics
            if func_divergence:
                history_entry["func_divergence"] = func_divergence
            if cd_spi_sym_metrics:
                history_entry["cd_spi_sym"] = cd_spi_sym_metrics
            if ood_results:
                history_entry["ood"] = ood_results
            if label_noise_results:
                history_entry["label_noise"] = label_noise_results
            self.history.append(history_entry)

            round_pbar.set_postfix(
                loss=f"{avg_loss:.4f}",
                time=f"{round_elapsed:.0f}s",
                mse=domain_str.replace(" ", "") if domain_str != "skipped" else "skip",
            )
            tqdm.write(
                f"[Round {round_num + 1}/{self.num_rounds}] Summary | "
                f"avg_loss={avg_loss:.4f} | "
                f"time={round_elapsed:.1f}s | "
                f"steps/s={round_steps_per_sec:.1f} | "
                f"samples/s={round_samples_per_sec:.1f}"
            )

            if on_round_end is not None:
                on_round_end(round_num, self.server.get_global_model())

            # Auto-save history snapshot for crash recovery
            self._save_history_snapshot()

        round_pbar.close()
        total_elapsed = time.perf_counter() - global_start
        tqdm.write(f"\n[Done] Total time: {total_elapsed:.1f}s")

        return {
            "history": self.history,
            "final_model": self.server.get_global_model(),
            "num_rounds": self.num_rounds,
            "num_clients": self.num_clients,
            "total_sec": round(total_elapsed, 2),
        }
