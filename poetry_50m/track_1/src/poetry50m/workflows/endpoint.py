"""Offline endpoint-informed teacher analysis."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

from poetry50m.model import DecoderOnlyTransformer, ModelConfig
from poetry50m.trajectory import analyze_endpoint_geometry_paths

JsonWriter = Callable[[Path, object], None]
ModelConfigLoader = Callable[[Path], ModelConfig]


def endpoint_analyze_command(
    args: argparse.Namespace,
    *,
    model_config_loader: ModelConfigLoader,
    write_json: JsonWriter,
) -> int:
    """Write one durable offline report; it is never a sealed-target fit input."""

    model = (
        DecoderOnlyTransformer(model_config_loader(Path(args.model_config)))
        if args.model_config is not None
        else None
    )
    report = analyze_endpoint_geometry_paths(
        tuple(Path(path) for path in args.snapshots),
        model=model,
    )
    write_json(Path(args.output), report.to_mapping())
    return 0
