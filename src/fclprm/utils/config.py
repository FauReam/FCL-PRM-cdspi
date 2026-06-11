"""YAML/JSON configuration loading and validation."""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


class ExperimentConfig:
    """Experiment configuration container.

    Loads YAML config and provides dict-like access with dot notation.
    """

    def __init__(self, config_path: str) -> None:
        """Load configuration from file.

        Args:
            config_path: Path to YAML config file.
        """
        self.config_path = Path(config_path)
        self._config = self._load()

    def _load(self) -> dict[str, Any]:
        """Load and parse config file."""
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    def get(self, key: str, default: Any = None) -> Any:
        """Get config value by dot-separated key.

        Args:
            key: Dot-separated key (e.g., "model.backbone").
            default: Default value if key not found.

        Returns:
            Config value or default.
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        return value

    def require(self, key: str) -> Any:
        """Get config value by dot-separated key, raising if missing.

        Args:
            key: Dot-separated key (e.g., "model.backbone").

        Returns:
            Config value.

        Raises:
            KeyError: If the key is not found in the config.
        """
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                raise KeyError(
                    f"Required config key '{key}' not found in {self.config_path}"
                )
        return value

    def to_dict(self) -> dict[str, Any]:
        """Return full config as dict.

        Returns:
            Configuration dictionary.
        """
        return self._config.copy()

    def hash(self) -> str:
        """Compute deterministic hash of config for reproducibility.

        Returns:
            Hex digest string.
        """
        config_str = json.dumps(self._config, sort_keys=True)
        return hashlib.sha256(config_str.encode("utf-8")).hexdigest()[:16]

    def validate_keys(self, required: list[str]) -> list[str]:
        """Check that all required keys are present.

        Args:
            required: List of dot-separated keys that must exist.

        Returns:
            List of missing keys (empty if all present).
        """
        missing = []
        for key in required:
            try:
                self.require(key)
            except KeyError:
                missing.append(key)
        return missing
