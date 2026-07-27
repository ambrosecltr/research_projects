"""Safe local loading and strict coordinate validation for weights-only snapshots."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import BinaryIO

import torch
from torch import Tensor

from poetry50m.trajectory._persistence import atomic_write
from poetry50m.trajectory.types import SNAPSHOT_FORMAT, SnapshotMetadata, WeightSnapshot


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be a mapping")
    if any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _load_weights_only(path: Path) -> object:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as error:
        raise RuntimeError(
            "PyTorch with torch.load(weights_only=True) support is required"
        ) from error


def load_weight_snapshot(path: Path, *, expected: WeightSnapshot | None = None) -> WeightSnapshot:
    """Load one trusted-local, tensor-only snapshot and validate every coordinate.

    The loader intentionally accepts only the Track 1 plain-dictionary format.
    ``weights_only=True`` prevents arbitrary pickle execution, but the caller is
    still responsible for choosing a local path they trust.
    """

    resolved = path.expanduser().resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"snapshot path is not a regular file: {resolved}")
    payload = _mapping(_load_weights_only(resolved), "snapshot payload")
    if set(payload) != {"format", "metadata", "state_dict"}:
        raise ValueError("snapshot payload must contain exactly format, metadata, and state_dict")
    if payload["format"] != SNAPSHOT_FORMAT:
        raise ValueError(f"unsupported snapshot format: {payload['format']!r}")
    metadata = SnapshotMetadata.from_mapping(_mapping(payload["metadata"], "metadata"))
    raw_state = _mapping(payload["state_dict"], "state_dict")
    state_dict: dict[str, Tensor] = {}
    for name, tensor in raw_state.items():
        if not isinstance(tensor, Tensor):
            raise TypeError(f"state_dict[{name!r}] must be a tensor")
        state_dict[name] = tensor.detach().cpu().contiguous()
    snapshot = WeightSnapshot(metadata=metadata, state_dict=state_dict, source_path=resolved)
    if expected is not None:
        assert_identical_coordinates(expected, snapshot)
    return snapshot


def save_weight_snapshot(path: Path, snapshot: WeightSnapshot) -> None:
    """Write the restricted plain-dictionary format used by ``load_weight_snapshot``."""

    def write(handle: BinaryIO) -> None:
        torch.save(
            {
                "format": SNAPSHOT_FORMAT,
                "metadata": snapshot.metadata.to_mapping(),
                "state_dict": {
                    name: tensor.detach().cpu() for name, tensor in snapshot.state_dict.items()
                },
            },
            handle,
        )

    atomic_write(path, write)


def assert_identical_coordinates(left: WeightSnapshot, right: WeightSnapshot) -> None:
    """Reject shape, dtype, architecture, corpus, or configuration drift explicitly."""

    if tuple(left.state_dict) != tuple(right.state_dict):
        raise ValueError(
            "state_dict tensor names or order differ; raw trajectory coordinates are unsafe"
        )
    for name in left.state_dict:
        left_tensor, right_tensor = left.state_dict[name], right.state_dict[name]
        if left_tensor.shape != right_tensor.shape:
            raise ValueError(
                f"tensor shape differs for {name}: "
                f"{tuple(left_tensor.shape)} != {tuple(right_tensor.shape)}"
            )
        if left_tensor.dtype != right_tensor.dtype:
            raise ValueError(
                f"tensor dtype differs for {name}: {left_tensor.dtype} != {right_tensor.dtype}"
            )
    for field_name in (
        "architecture_signature",
        "corpus_signature",
        "model_config_hash",
        "tokenizer_hash",
        "code_signature",
        "training_config_hash",
    ):
        if getattr(left.metadata, field_name) != getattr(right.metadata, field_name):
            raise ValueError(f"{field_name} differs; raw coordinate operations are unsafe")


def assert_single_run_trajectory(snapshots: tuple[WeightSnapshot, ...]) -> None:
    """Ensure a fit uses one continuous, strictly increasing training trajectory."""

    if len(snapshots) < 2:
        raise ValueError("trajectory fitting requires at least two snapshots")
    reference = snapshots[0]
    previous_step = -1
    seen_ids: set[str] = set()
    for snapshot in snapshots:
        assert_identical_coordinates(reference, snapshot)
        metadata = snapshot.metadata
        if metadata.run_id != reference.metadata.run_id:
            raise ValueError("a raw trajectory fit cannot mix run IDs")
        if metadata.initialization_id != reference.metadata.initialization_id:
            raise ValueError("a raw trajectory fit cannot mix initialization IDs")
        if metadata.data_order_id != reference.metadata.data_order_id:
            raise ValueError("a raw trajectory fit cannot mix data orders")
        if metadata.step <= previous_step:
            raise ValueError("trajectory checkpoint steps must be strictly increasing")
        if metadata.checkpoint_id in seen_ids:
            raise ValueError("trajectory checkpoint IDs must be unique")
        previous_step = metadata.step
        seen_ids.add(metadata.checkpoint_id)
