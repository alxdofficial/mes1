"""Configuration loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


class ConfigLoader:
    """Load configuration from YAML and environment."""

    def __init__(self, config_dir: Path | str | None = None):
        if config_dir is None:
            self._project_root = self._find_project_root()
            self._config_dir = self._project_root / "config"
        else:
            self._config_dir = Path(config_dir)
            self._project_root = self._config_dir.parent

        env_file = self._project_root / ".env"
        if env_file.exists():
            load_dotenv(env_file)

        self._settings: dict[str, Any] = {}

    def _find_project_root(self) -> Path:
        current = Path(__file__).resolve()
        for parent in current.parents:
            if (parent / "pyproject.toml").exists():
                return parent
        return Path(__file__).resolve().parents[3]

    def load(self) -> ConfigLoader:
        filepath = self._config_dir / "settings.yaml"
        if filepath.exists():
            with open(filepath) as f:
                self._settings = yaml.safe_load(f) or {}
        return self

    @property
    def settings(self) -> dict[str, Any]:
        return self._settings

    @property
    def project_root(self) -> Path:
        return self._project_root

    def get_path(self, key: str) -> Path:
        value = self.get(f"paths.{key}", key)
        return self._project_root / value

    def get(self, key: str, default: Any = None) -> Any:
        parts = key.split(".")
        value: Any = self._settings
        for part in parts:
            if isinstance(value, dict) and part in value:
                value = value[part]
            else:
                return default
        return value

    def get_env(self, key: str, default: str | None = None) -> str | None:
        return os.getenv(key, default)

    def require_env(self, key: str) -> str:
        value = os.getenv(key)
        if value is None:
            raise ValueError(f"Required env var '{key}' not set")
        return value


_config: ConfigLoader | None = None


def get_config() -> ConfigLoader:
    global _config
    if _config is None:
        _config = ConfigLoader().load()
    return _config
