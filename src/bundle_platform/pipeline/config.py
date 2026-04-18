"""
Global configuration management for bundle_platform.

API key lookup priority (first match wins):
  1. ANTHROPIC_API_KEY environment variable
  2. ~/.config/bundle_platform/config  (written by `bundle_platform setup`)
  3. .env file in the current working directory

This allows:
  - Installed use (uv tool install): key stored once in ~/.config/bundle_platform/config
  - CI/CD: key passed via environment variable
  - Dev: key in project-local .env
"""

import os
from pathlib import Path

from dotenv import dotenv_values

_DEFAULT_CONFIG_DIR = Path.home() / ".config" / "bundle_platform"
_CONFIG_FILENAME = "config"


def load_api_key(
    config_dir: Path | None = None,
    dotenv_path: Path | None = None,
) -> str | None:
    """
    Load the Anthropic API key using the priority chain.

    Args:
        config_dir:   Override the global config directory (used in tests).
        dotenv_path:  Override the .env file path (used in tests).

    Returns:
        The API key string, or None if not found anywhere.
    """
    # 1. Environment variable
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key.strip()

    # 2. Global config file
    cfg_dir = config_dir or _DEFAULT_CONFIG_DIR
    config_file = cfg_dir / _CONFIG_FILENAME
    if config_file.exists():
        for line in config_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip()

    # 3. .env file in current directory
    env_file = dotenv_path or Path(".env")
    if env_file.exists():
        values = dotenv_values(env_file)
        key = values.get("ANTHROPIC_API_KEY")
        if key:
            return key.strip()

    return None


def save_api_key(key: str, config_dir: Path | None = None) -> None:
    """
    Write the API key to the global config file with private permissions (600).

    Args:
        key:        The Anthropic API key to store.
        config_dir: Override the global config directory (used in tests).
    """
    cfg_dir = config_dir or _DEFAULT_CONFIG_DIR
    cfg_dir.mkdir(parents=True, exist_ok=True)
    config_file = cfg_dir / _CONFIG_FILENAME
    config_file.write_text(f"ANTHROPIC_API_KEY={key}\n")
    config_file.chmod(0o600)
