"""Structured JSONL logging for experiment tracking."""

import json
from datetime import datetime, timezone
from pathlib import Path


class ExperimentLogger:
    """Log experiment metrics to JSONL file.

    Each line is a JSON object with:
        - timestamp: ISO format datetime
        - experiment_id: Unique experiment identifier
        - milestone: Milestone tag (e.g., "M2")
        - config_hash: Hash of experiment configuration
        - metrics: Dict of metric values
    """

    def __init__(self, log_dir: str, experiment_id: str) -> None:
        """Initialize logger.

        Args:
            log_dir: Directory for log files.
            experiment_id: Unique experiment identifier.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_file = self.log_dir / f"{experiment_id}.jsonl"
        self.experiment_id = experiment_id

    def log(self, milestone: str, config_hash: str, metrics: dict) -> None:
        """Log a single metrics record.

        Args:
            milestone: Milestone tag.
            config_hash: Configuration hash.
            metrics: Metric values dict.
        """
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "experiment_id": self.experiment_id,
            "milestone": milestone,
            "config_hash": config_hash,
            "metrics": metrics,
        }
        with open(self.log_file, "a") as f:
            f.write(json.dumps(record) + "\n")
