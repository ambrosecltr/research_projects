from __future__ import annotations

from pathlib import Path
from typing import Mapping

import torch
from safetensors.torch import load_file, save_file

from .hashing import sha256_json, sha256_tensor


def state_id(state: Mapping[str, torch.Tensor]) -> str:
    return sha256_json(
        {
            name: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "bytes_sha256": sha256_tensor(tensor),
            }
            for name, tensor in sorted(state.items())
        }
    )


def direct_fp16_delta_bytes(state: Mapping[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * 2 for tensor in state.values())


def load_state(path: str | Path) -> dict[str, torch.Tensor]:
    return dict(load_file(str(path), device="cpu"))


def save_state(path: str | Path, state: Mapping[str, torch.Tensor]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save_file({name: tensor.detach().cpu().contiguous() for name, tensor in state.items()}, str(destination))
