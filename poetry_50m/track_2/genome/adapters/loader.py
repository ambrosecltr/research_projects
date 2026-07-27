from __future__ import annotations

import importlib
import inspect
import sys
from pathlib import Path
from typing import Any, Mapping

from .base import Track1Adapter


def import_object(path: str) -> Any:
    if ":" in path:
        module_name, attribute = path.split(":", 1)
    else:
        module_name, attribute = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def load_adapter(config: Mapping[str, Any]) -> Track1Adapter:
    adapter_config = config.get("adapter", config)
    if not isinstance(adapter_config, Mapping):
        raise ValueError("adapter configuration must be a mapping")
    project_root = adapter_config.get("project_root")
    if project_root:
        root_path = Path(project_root).expanduser()
        if not root_path.is_absolute():
            config_path = config.get("_config_path")
            base = Path(str(config_path)).resolve().parent if config_path else Path.cwd()
            root_path = base / root_path
        root_path = root_path.resolve()
        candidates = [root_path]
        if (root_path / "src").is_dir():
            candidates.insert(0, root_path / "src")
        for candidate in reversed(candidates):
            value = str(candidate)
            if value not in sys.path:
                sys.path.insert(0, value)
    factory_path = adapter_config.get("factory")
    if not factory_path:
        raise KeyError("adapter.factory is required")
    factory = import_object(str(factory_path))
    kwargs = dict(adapter_config.get("kwargs", {}))
    signature = inspect.signature(factory)
    if "config" in signature.parameters and "config" not in kwargs:
        kwargs["config"] = dict(config)
    adapter = factory(**kwargs) if callable(factory) else factory
    if inspect.isclass(adapter):
        adapter = adapter()
    if not isinstance(adapter, Track1Adapter):
        raise TypeError(f"adapter factory returned {type(adapter)!r}, expected Track1Adapter")
    return adapter
