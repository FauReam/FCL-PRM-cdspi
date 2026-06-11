"""Single federated client: local PRM training + optional DP-SGD."""

import time
from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


class FederatedClient:
    """Client-local training for PRM.

    Responsibilities:
        - Load local data partition
        - Train PRM head (backbone frozen)
        - Apply DP-SGD if privacy is enabled
        - Return model delta + step embeddings for aggregation
    """

    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        train_data: list[dict],
        dp_enabled: bool = False,
        dp_epsilon: float = 4.0,
        dp_delta: float = 1e-5,
    ) -> None:
        """Initialize client.

        Args:
            client_id: Unique client identifier.
            model: Local model instance.
            train_data: Client's local training data (list of step samples).
            dp_enabled: Whether to apply DP-SGD during local training.
            dp_epsilon: Privacy budget epsilon (if DP enabled).
            dp_delta: Privacy budget delta (if DP enabled).
        """
        self.client_id = client_id
        self.model = model
        self.train_data = train_data
        self.dp_enabled = dp_enabled
        self.dp_epsilon = dp_epsilon
        self.dp_delta = dp_delta

    def local_train(
        self,
        num_epochs: int,
        batch_size: int = 32,
        learning_rate: float = 1e-4,
        device: str = "cuda",
        max_grad_norm: float = 1.0,
        log_interval: int = 5,
        scheduler: str | None = None,
        max_steps_per_epoch: int | None = None,
        num_workers: int = 0,
    ) -> dict:
        """Run local training for specified epochs.

        Args:
            num_epochs: Number of local epochs per round.
            batch_size: Local batch size.
            learning_rate: Local learning rate.
            device: Device for training.
            max_grad_norm: Per-sample gradient clipping bound (used when DP enabled).
            log_interval: Print real-time metrics every N batches.
            scheduler: Learning rate scheduler name ("cosine", "constant", or None).
            max_steps_per_epoch: If set, limit each epoch to this many batches.
                Useful for fast pipeline verification without reducing dataset size.
            num_workers: Number of DataLoader workers. 0 = main-process only
                (Windows-compatible); 2–4 on Linux for pipelined prefetch.

        Returns:
            Dict containing model state dict and training metrics.
        """
        self.model.to(device)
        self.model.train()

        # Only optimize parameters that require grad (PRM head)
        optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=learning_rate,
        )

        # LR scheduler
        lr_scheduler = None
        if scheduler == "cosine":
            from torch.optim.lr_scheduler import CosineAnnealingLR

            lr_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
        elif scheduler == "constant" or scheduler is None:
            pass
        else:
            tqdm.write(f"  [WARN] Unknown scheduler '{scheduler}', using constant LR")

        # Simple dataloader from list of dicts
        from fclprm.data.utils import collate_step_batch

        loader_kwargs: dict = {
            "batch_size": batch_size,
            "shuffle": True,
            "collate_fn": collate_step_batch,
            "pin_memory": True,
        }
        if num_workers > 0:
            loader_kwargs.update({
                "num_workers": num_workers,
                "prefetch_factor": 2,
                "persistent_workers": True,
            })
        loader = DataLoader(self.train_data, **loader_kwargs)

        # Apply DP-SGD if enabled
        model = self.model
        if self.dp_enabled:
            from fclprm.federated.dp import StepLevelDPSGD

            dp_engine = StepLevelDPSGD(
                epsilon=self.dp_epsilon,
                delta=self.dp_delta,
                max_grad_norm=max_grad_norm,
            )
            model, optimizer, loader = dp_engine.make_private(
                model=model,
                optimizer=optimizer,
                data_loader=loader,
                epochs=num_epochs,
            )

        total_loss = 0.0
        num_batches = 0
        epoch_loss = 0.0
        epoch_batches = 0
        start_time = time.perf_counter()
        interval_start = start_time
        interval_batches = 0
        interval_samples = 0

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            epoch_batches = 0
            batch_iter = iter(loader)
            if max_steps_per_epoch is not None:
                from itertools import islice
                batch_iter = islice(batch_iter, max_steps_per_epoch)
            pbar = tqdm(
                batch_iter,
                desc=f"  [Client {self.client_id}] epoch {epoch + 1}/{num_epochs}",
                leave=False,
                total=max_steps_per_epoch if max_steps_per_epoch else len(loader),
            )
            for batch in pbar:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                optimizer.zero_grad()
                predictions = model(input_ids, attention_mask)
                loss = F.mse_loss(predictions, labels)
                loss.backward()
                optimizer.step()

                # Force CUDA sync on Windows to surface async errors early
                if device == "cuda":
                    torch.cuda.synchronize()

                total_loss += loss.item()
                num_batches += 1
                epoch_loss += loss.item()
                epoch_batches += 1

                pbar.set_postfix(loss=f"{loss.item():.4f}")

            if lr_scheduler is not None:
                lr_scheduler.step()

            if epoch_batches > 0:
                lr_str = f" lr={optimizer.param_groups[0]['lr']:.2e}" if lr_scheduler else ""
                tqdm.write(
                    f"  [Client {self.client_id}] epoch {epoch + 1}/{num_epochs} done | "
                    f"avg_loss={epoch_loss / epoch_batches:.4f}{lr_str}"
                )

        elapsed = time.perf_counter() - start_time
        avg_loss = total_loss / max(num_batches, 1)
        steps_per_sec = num_batches / elapsed if elapsed > 0 else 0.0
        samples_per_sec = (num_batches * batch_size) / elapsed if elapsed > 0 else 0.0

        # Opacus wraps the model in GradSampleModule, which prefixes
        # state_dict keys with "_module.". Strip the prefix before
        # shipping to the server so aggregators can match parameter names.
        raw_model = model._module if hasattr(model, "_module") else model
        return {
            "client_id": self.client_id,
            "state_dict": {k: v.cpu().clone() for k, v in raw_model.state_dict().items()},
            "loss": avg_loss,
            "num_samples": len(self.train_data),
            "num_batches": num_batches,
            "elapsed_sec": round(elapsed, 3),
            "steps_per_sec": round(steps_per_sec, 2),
            "samples_per_sec": round(samples_per_sec, 2),
        }

    def get_step_embeddings(
        self,
        steps: list[str],
        tokenizer,
        device: str = "cuda",
        max_length: int = 512,
    ) -> torch.Tensor:
        """Extract step embeddings from local model for CD-SPI.

        Args:
            steps: List of step strings.
            tokenizer: HuggingFace tokenizer.
            device: Device for inference.
            max_length: Max token length.

        Returns:
            Step embeddings tensor of shape (len(steps), hidden_dim).
        """
        from tqdm import tqdm

        self.model.to(device)
        self.model.eval()

        with tqdm(total=1, desc=f"  [Client {self.client_id}] Tokenize steps", leave=False) as pbar:
            encoded = tokenizer(
                steps,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            pbar.update(1)

        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            with tqdm(total=1, desc=f"  [Client {self.client_id}] Extract embeddings", leave=False) as pbar:
                embeddings = self.model.get_step_embedding(input_ids, attention_mask)
                pbar.update(1)

        return embeddings.cpu()
