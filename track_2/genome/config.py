from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .hashing import sha256_json
from .io import read_yaml


def load_config(path: str | Path) -> dict[str, Any]:
    value = read_yaml(path)
    if not isinstance(value, Mapping):
        raise ValueError("configuration root must be a mapping")
    config = dict(value)
    config["_config_path"] = str(Path(path).resolve())
    config["_config_hash"] = sha256_json({k: v for k, v in config.items() if not k.startswith("_")})
    return config


def require(config: Mapping[str, Any], *path: str) -> Any:
    value: Any = config
    walked: list[str] = []
    for key in path:
        walked.append(key)
        if not isinstance(value, Mapping) or key not in value:
            raise KeyError(f"missing configuration value: {'.'.join(walked)}")
        value = value[key]
    return value


def resolve_config_path(config: Mapping[str, Any], value: str | Path) -> Path:
    """Resolve one configured path relative to the YAML file that declared it."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    config_path = config.get("_config_path")
    base = Path(str(config_path)).resolve().parent if config_path else Path.cwd()
    return (base / path).resolve()
