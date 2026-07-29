"""Archived V1 neural auto-decoder for historical reproduction only."""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from ..codecs.common import make_manifest, make_records
from ..mgp.opcodes import COPY_FROM_TIED, NEURAL_BLOCK_FIELD
from ..state import compute_delta
from ..tensor_inventory import tied_owner_map
from ..types import GenomeComponent, GenomeProgram, TensorSpec
from .block_decoder import (
    BlockDecoderConfig,
    GenomeCodeBank,
    LazyDeltaBlockDataset,
    RoleConditionedBlockDecoder,
    save_interpreter,
)


@dataclass(frozen=True)
class AutodecoderTrainingConfig:
    seed: int = 1701
    updates: int = 2000
    batch_size: int = 128
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip_norm: float = 1.0
    device: str = "cpu"
    log_every: int = 100

    def __post_init__(self) -> None:
        for name in ("seed", "updates", "batch_size", "log_every"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")
        for name in ("learning_rate", "weight_decay", "grad_clip_norm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            minimum_ok = float(value) > 0 if name != "weight_decay" else float(value) >= 0
            if not math.isfinite(float(value)) or not minimum_ok:
                qualifier = "positive" if name != "weight_decay" else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if not isinstance(self.device, str) or not self.device:
            raise ValueError("device must be a non-empty string")


@dataclass
class AutodecoderResult:
    program: GenomeProgram
    interpreter_info: dict[str, Any]
    metrics: list[dict[str, float]]
    role_scales: dict[str, float]
    decoder_config: BlockDecoderConfig
    training_config: AutodecoderTrainingConfig


def _role_scales(
    delta: Mapping[str, torch.Tensor], specs: Sequence[TensorSpec]
) -> dict[str, float]:
    values: dict[str, list[torch.Tensor]] = {}
    for spec in specs:
        values.setdefault(spec.role, []).append(delta[spec.name].flatten().abs())
    scales = {}
    for role, chunks in values.items():
        # RMS is deterministic and cheap; later experiments can replace it with robust quantiles.
        sum_sq = sum(float(chunk.square().sum().item()) for chunk in chunks)
        count = sum(chunk.numel() for chunk in chunks)
        scales[role] = max((sum_sq / max(count, 1)) ** 0.5, 1e-8)
    return scales


def fit_autodecoder(
    base_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    tensor_specs: Sequence[TensorSpec],
    *,
    tied_groups: Sequence[Sequence[str]] = (),
    interpreter_path: str | Path,
    candidate_id: str = "g0_neural_block",
    decoder_config: BlockDecoderConfig | None = None,
    training_config: AutodecoderTrainingConfig | None = None,
    manifest_metadata: Mapping[str, Any] | None = None,
) -> AutodecoderResult:
    decoder_config = decoder_config or BlockDecoderConfig()
    training_config = training_config or AutodecoderTrainingConfig()
    if Path(interpreter_path).exists():
        raise FileExistsError(interpreter_path)
    random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    device = torch.device(training_config.device)

    delta = compute_delta(base_state, target_state, tensor_specs)
    aliases = tied_owner_map(tied_groups)
    roles = sorted({spec.role for spec in tensor_specs if spec.name not in aliases})
    role_to_id = {role: index for index, role in enumerate(roles)}
    layer_values = sorted(
        {spec.layer_index for spec in tensor_specs}, key=lambda x: -1 if x is None else x
    )
    layer_to_slot = {layer: index for index, layer in enumerate(layer_values)}
    role_scales = _role_scales(delta, [spec for spec in tensor_specs if spec.name not in aliases])

    dataset = LazyDeltaBlockDataset(
        base_state,
        delta,
        tensor_specs,
        tied_aliases=aliases,
        role_to_id=role_to_id,
        layer_to_slot=layer_to_slot,
        role_scales=role_scales,
        config=decoder_config,
    )
    decoder = RoleConditionedBlockDecoder(len(role_to_id), decoder_config).to(device)
    codes = GenomeCodeBank(len(layer_to_slot), len(tensor_specs), decoder_config).to(device)
    optimizer = torch.optim.AdamW(
        [*decoder.parameters(), *codes.parameters()],
        lr=training_config.learning_rate,
        weight_decay=training_config.weight_decay,
    )
    generator = torch.Generator(device="cpu").manual_seed(training_config.seed + 1)
    metrics: list[dict[str, float]] = []
    decoder.train()
    for update in range(1, training_config.updates + 1):
        indices = torch.randint(
            0,
            len(dataset),
            (min(training_config.batch_size, len(dataset)),),
            generator=generator,
        )
        inputs, targets = dataset.make_batch(indices, codes, device=device)
        prediction = decoder(**inputs)
        loss = torch.nn.functional.mse_loss(prediction, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*decoder.parameters(), *codes.parameters()], training_config.grad_clip_norm
        )
        optimizer.step()
        if (
            update == 1
            or update % training_config.log_every == 0
            or update == training_config.updates
        ):
            metrics.append({"update": float(update), "block_mse": float(loss.item())})

    decoder.eval()
    interpreter_info = save_interpreter(
        decoder,
        role_to_id=role_to_id,
        layer_to_slot=layer_to_slot,
        role_scales=role_scales,
        path=interpreter_path,
    )

    records, record_aliases = make_records(tensor_specs, tied_groups)
    payload: dict[str, torch.Tensor] = {
        "shared.global_code": codes.global_code.detach().cpu(),
        "shared.layer_codes": codes.layer_codes.detach().cpu(),
    }
    max_real_layer = max([layer for layer in layer_values if layer is not None], default=0)
    for record in records:
        spec = tensor_specs[record.canonical_index]
        if record.tensor_name in record_aliases:
            record.components.append(
                GenomeComponent(
                    COPY_FROM_TIED, arguments={"owner": record_aliases[record.tensor_name]}
                )
            )
            continue
        code_key = f"t{record.canonical_index:05d}.tensor_code"
        payload[code_key] = codes.tensor_codes[record.canonical_index].detach().cpu()
        normalized_layer = (
            -1.0 if spec.layer_index is None else float(spec.layer_index) / max(max_real_layer, 1)
        )
        matrix_rows = 1 if len(spec.shape) < 2 else spec.shape[0]
        matrix_cols = spec.numel // matrix_rows
        record.components.append(
            GenomeComponent(
                NEURAL_BLOCK_FIELD,
                payload_keys=[code_key],
                arguments={
                    "role": spec.role,
                    "shape": list(spec.shape),
                    "matrix_shape": [matrix_rows, matrix_cols],
                    "layer_slot": layer_to_slot[spec.layer_index],
                    "normalized_layer": normalized_layer,
                    "global_code_key": "shared.global_code",
                    "layer_codes_key": "shared.layer_codes",
                    "block_rows": decoder_config.block_rows,
                    "block_cols": decoder_config.block_cols,
                },
            )
        )

    manifest = make_manifest(
        candidate_id=candidate_id, codec="neural_block_field", metadata=manifest_metadata
    )
    manifest.update(
        {
            "interpreter_id": Path(interpreter_path).name,
            "interpreter_manifest_sha256": interpreter_info["manifest_sha256"],
            "interpreter_decoder_sha256": interpreter_info["decoder_sha256"],
            "interpreter_bytes": interpreter_info["bytes"],
            "shared_payload_keys": ["shared.global_code", "shared.layer_codes"],
            "codec_config": {
                "decoder": decoder_config.to_dict(),
                "training": asdict(training_config),
                "role_to_id": role_to_id,
                "layer_to_slot": {
                    "none" if k is None else str(k): v for k, v in layer_to_slot.items()
                },
                "role_scales": role_scales,
            },
        }
    )
    program = GenomeProgram(manifest=manifest, records=records, payload_tensors=payload)
    return AutodecoderResult(
        program=program,
        interpreter_info=interpreter_info,
        metrics=metrics,
        role_scales=role_scales,
        decoder_config=decoder_config,
        training_config=training_config,
    )
