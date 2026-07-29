from __future__ import annotations

import math
import random
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import save_file
from torch import nn

from ..codecs.common import make_manifest, make_records
from ..hashing import sha256_directory, sha256_file, sha256_json
from ..io import (
    directory_size,
    read_json,
    replace_directory_atomic,
    resolve_artifact_directory,
    resolve_artifact_relative_file,
    temporary_directory,
    write_json,
)
from ..mgp.opcodes import COPY_FROM_TIED, NEURAL_BLOCK_FIELD
from ..mgp.serializer import save_program
from ..polypythia.lives import CanonicalModelLife
from ..state import compute_delta
from ..tensor_inventory import tied_owner_map
from ..types import GenomeComponent, GenomeProgram, TensorSpec
from .block_decoder import (
    BlockDecoderConfig,
    BlockReference,
    NeuralBlockInterpreter,
    RoleConditionedBlockDecoder,
    load_interpreter,
    make_block_features,
    save_interpreter,
    tensor_matrix_shape,
    tensor_matrix_view,
)
from .compiler import GenomeCodeLayout


@dataclass(frozen=True)
class SharedDecoderTrainingConfig:
    seed: int = 1701
    updates: int = 50_000
    batch_size: int = 256
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    code_weight_decay: float = 1e-6
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    log_every: int = 500

    def __post_init__(self) -> None:
        for name in ("seed", "updates", "batch_size", "log_every"):
            value = getattr(self, name)
            minimum = 0 if name == "seed" else 1
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                qualifier = "non-negative" if minimum == 0 else "positive"
                raise ValueError(f"{name} must be a {qualifier} integer")
        for name in (
            "learning_rate",
            "weight_decay",
            "code_weight_decay",
            "grad_clip_norm",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be numeric")
            positive = name in {"learning_rate", "grad_clip_norm"}
            if not math.isfinite(float(value)) or (
                float(value) <= 0 if positive else float(value) < 0
            ):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if not self.device:
            raise ValueError("device cannot be empty")


@dataclass(frozen=True)
class LatentCodeFitConfig:
    seed: int = 3141
    updates: int = 20_000
    batch_size: int = 256
    learning_rate: float = 1e-3
    weight_decay: float = 1e-6
    grad_clip_norm: float = 1.0
    device: str = "cuda"
    log_every: int = 200

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
            positive = name != "weight_decay"
            if not math.isfinite(float(value)) or (
                float(value) <= 0 if positive else float(value) < 0
            ):
                qualifier = "positive" if positive else "non-negative"
                raise ValueError(f"{name} must be finite and {qualifier}")
        if not self.device:
            raise ValueError("device cannot be empty")


class MultiLifeGenomeCodes(nn.Module):
    def __init__(
        self,
        *,
        life_count: int,
        layer_count: int,
        tensor_count: int,
        block_count: int,
        config: BlockDecoderConfig,
    ) -> None:
        super().__init__()
        for name, value in (
            ("life_count", life_count),
            ("layer_count", layer_count),
            ("tensor_count", tensor_count),
            ("block_count", block_count),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        self.global_codes = nn.Parameter(torch.empty(life_count, config.global_code_dim))
        self.layer_codes = nn.Parameter(torch.empty(life_count, layer_count, config.layer_code_dim))
        self.tensor_codes = nn.Parameter(
            torch.empty(life_count, tensor_count, config.tensor_code_dim)
        )
        self.life_count = life_count
        self.block_count = block_count
        self.block_code_dim = config.block_code_dim
        self.block_codes = (
            nn.Embedding(
                life_count * block_count,
                config.block_code_dim,
                sparse=True,
            )
            if config.block_code_dim and config.block_code_mode == "network"
            else None
        )
        nn.init.normal_(self.global_codes, std=0.02)
        nn.init.normal_(self.layer_codes, std=0.02)
        nn.init.normal_(self.tensor_codes, std=0.02)
        if self.block_codes is not None:
            nn.init.normal_(self.block_codes.weight, std=0.02)

    def block_for_batch(
        self,
        life_indices: torch.Tensor,
        reference_indices: torch.Tensor,
    ) -> torch.Tensor | None:
        if self.block_codes is None:
            return None
        flat_indices = life_indices * self.block_count + reference_indices
        return self.block_codes(flat_indices)

    def for_life(self, index: int) -> dict[str, torch.Tensor]:
        result = {
            "global_code": self.global_codes[index],
            "layer_codes": self.layer_codes[index],
            "tensor_codes": self.tensor_codes[index],
        }
        if self.block_codes is not None:
            start = index * self.block_count
            result["block_codes"] = self.block_codes.weight[start : start + self.block_count]
        return result


@dataclass(frozen=True)
class BlockBatch:
    life_indices: torch.Tensor
    reference_indices: torch.Tensor
    layer_slots: torch.Tensor
    tensor_indices: torch.Tensor
    role_ids: torch.Tensor
    features: torch.Tensor
    targets: torch.Tensor
    valid_masks: torch.Tensor


def masked_block_mse(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or target.shape != valid_mask.shape:
        raise ValueError("prediction, target, and valid block mask shapes must match")
    valid_count = valid_mask.sum()
    if valid_count <= 0:
        raise ValueError("block batch contains no valid target values")
    return ((prediction - target).square() * valid_mask).sum() / valid_count


class MultiLifeBlockSampler:
    def __init__(
        self,
        *,
        base_states: Sequence[Mapping[str, torch.Tensor]],
        target_states: Sequence[Mapping[str, torch.Tensor]],
        tensor_specs: Sequence[TensorSpec],
        tied_groups: Sequence[Sequence[str]],
        decoder_config: BlockDecoderConfig,
        role_to_id: Mapping[str, int] | None = None,
        layer_to_slot: Mapping[int | None, int] | None = None,
        role_scales: Mapping[str, float] | None = None,
        tensor_scales: Mapping[str, float] | None = None,
    ) -> None:
        if not base_states or len(base_states) != len(target_states):
            raise ValueError("base and target states must contain the same non-zero life count")
        self.base_states = list(base_states)
        self.delta_states = [
            compute_delta(base, target, tensor_specs)
            for base, target in zip(base_states, target_states, strict=True)
        ]
        self.tensor_specs = list(tensor_specs)
        self.spec_by_name = {spec.name: spec for spec in tensor_specs}
        self.decoder_config = decoder_config
        aliases = tied_owner_map(tied_groups)
        roles = sorted({spec.role for spec in tensor_specs if spec.name not in aliases})
        self.role_to_id = (
            {role: index for index, role in enumerate(roles)}
            if role_to_id is None
            else {str(role): int(index) for role, index in role_to_id.items()}
        )
        if set(self.role_to_id) != set(roles):
            raise ValueError("decoder roles do not match the model-life tensor roles")
        layers = sorted(
            {spec.layer_index for spec in tensor_specs},
            key=lambda value: -1 if value is None else value,
        )
        self.layer_to_slot = (
            {layer: index for index, layer in enumerate(layers)}
            if layer_to_slot is None
            else {layer: int(index) for layer, index in layer_to_slot.items()}
        )
        if set(self.layer_to_slot) != set(layers):
            raise ValueError("decoder layer slots do not match the model-life tensor layers")
        self.role_scales = (
            self._calculate_role_scales(aliases)
            if role_scales is None
            else {str(role): float(scale) for role, scale in role_scales.items()}
        )
        if set(self.role_scales) != set(roles):
            raise ValueError("decoder role scales do not match tensor roles")
        tensor_names = {spec.name for spec in tensor_specs if spec.name not in aliases}
        self.tensor_scales = (
            self._calculate_tensor_scales(aliases)
            if tensor_scales is None
            else {str(name): float(scale) for name, scale in tensor_scales.items()}
        )
        if set(self.tensor_scales) != tensor_names:
            raise ValueError("decoder tensor scales do not match untied tensor inventory")
        if any(not math.isfinite(scale) or scale <= 0 for scale in self.tensor_scales.values()):
            raise ValueError("decoder tensor scales must be finite and positive")
        self.references: list[BlockReference] = []
        self.references_by_role: dict[str, list[int]] = {role: [] for role in roles}
        for spec in tensor_specs:
            if spec.name in aliases:
                continue
            rows, cols = tensor_matrix_shape(spec.shape)
            for row_start in range(0, rows, decoder_config.block_rows):
                for col_start in range(0, cols, decoder_config.block_cols):
                    reference = BlockReference(
                        tensor_name=spec.name,
                        tensor_index=spec.canonical_index,
                        role_id=self.role_to_id[spec.role],
                        layer_slot=self.layer_to_slot[spec.layer_index],
                        block_row=row_start // decoder_config.block_rows,
                        block_col=col_start // decoder_config.block_cols,
                        row_start=row_start,
                        row_end=min(row_start + decoder_config.block_rows, rows),
                        col_start=col_start,
                        col_end=min(col_start + decoder_config.block_cols, cols),
                        rows=rows,
                        cols=cols,
                    )
                    self.references_by_role[spec.role].append(len(self.references))
                    self.references.append(reference)
        if not self.references or any(not indices for indices in self.references_by_role.values()):
            raise ValueError("each decoder role must contain at least one block")
        self.roles = tuple(sorted(self.references_by_role))

    @property
    def life_count(self) -> int:
        return len(self.base_states)

    def _calculate_role_scales(self, aliases: Mapping[str, str]) -> dict[str, float]:
        sums: dict[str, float] = {}
        counts: dict[str, int] = {}
        for delta in self.delta_states:
            for spec in self.tensor_specs:
                if spec.name in aliases:
                    continue
                value = delta[spec.name].to(torch.float32)
                sums[spec.role] = sums.get(spec.role, 0.0) + float(value.square().sum().item())
                counts[spec.role] = counts.get(spec.role, 0) + value.numel()
        return {role: max((sums[role] / counts[role]) ** 0.5, 1e-8) for role in sorted(sums)}

    def _calculate_tensor_scales(self, aliases: Mapping[str, str]) -> dict[str, float]:
        result = {}
        for spec in self.tensor_specs:
            if spec.name in aliases:
                continue
            squared_sum = sum(
                float(delta[spec.name].to(torch.float32).square().sum().item())
                for delta in self.delta_states
            )
            count = len(self.delta_states) * spec.numel
            result[spec.name] = max((squared_sum / count) ** 0.5, 1e-8)
        return result

    def _features(self, life_index: int, reference: BlockReference) -> torch.Tensor:
        base = tensor_matrix_view(
            self.base_states[life_index][reference.tensor_name].to(torch.float32)
        )
        block = base[
            reference.row_start : reference.row_end,
            reference.col_start : reference.col_end,
        ]
        spec = self.spec_by_name[reference.tensor_name]
        real_layers = [layer for layer in self.layer_to_slot if layer is not None]
        max_layer = max(real_layers, default=0)
        metadata = torch.tensor(
            [
                -1.0 if spec.layer_index is None else spec.layer_index / max(max_layer, 1),
                reference.block_row
                / max(
                    math.ceil(reference.rows / self.decoder_config.block_rows) - 1,
                    1,
                ),
                reference.block_col
                / max(
                    math.ceil(reference.cols / self.decoder_config.block_cols) - 1,
                    1,
                ),
                math.log1p(reference.rows) / 16.0,
                math.log1p(reference.cols) / 16.0,
                float(block.mean().item()),
                float(block.std(unbiased=False).item()) if block.numel() > 1 else 0.0,
            ],
            dtype=torch.float32,
        )
        return make_block_features(metadata, block, self.decoder_config)

    def make_batch(
        self,
        *,
        batch_size: int,
        generator: torch.Generator,
        device: torch.device,
    ) -> BlockBatch:
        life_indices = torch.randint(
            0,
            self.life_count,
            (batch_size,),
            generator=generator,
        )
        reference_indices = torch.randint(
            0,
            len(self.references),
            (batch_size,),
            generator=generator,
        )
        balanced_count = batch_size // 2
        role_slots = torch.randint(
            0,
            len(self.roles),
            (balanced_count,),
            generator=generator,
        )
        offsets = torch.randint(
            0,
            2**31 - 1,
            (balanced_count,),
            generator=generator,
        )
        for index, (role_slot, offset) in enumerate(
            zip(role_slots.tolist(), offsets.tolist(), strict=True)
        ):
            role = self.roles[role_slot]
            candidates = self.references_by_role[role]
            reference_indices[index] = candidates[offset % len(candidates)]
        return self.make_indexed_batch(
            life_indices=life_indices,
            reference_indices=reference_indices,
            device=device,
        )

    def make_indexed_batch(
        self,
        *,
        life_indices: torch.Tensor,
        reference_indices: torch.Tensor,
        device: torch.device,
    ) -> BlockBatch:
        life_indices = life_indices.detach().to(dtype=torch.long, device="cpu")
        reference_indices = reference_indices.detach().to(dtype=torch.long, device="cpu")
        if (
            life_indices.ndim != 1
            or reference_indices.ndim != 1
            or life_indices.shape != reference_indices.shape
            or life_indices.numel() < 1
        ):
            raise ValueError("life and reference indices must be matching non-empty vectors")
        if life_indices.min().item() < 0 or life_indices.max().item() >= self.life_count:
            raise IndexError("life index is outside the block sampler")
        if reference_indices.min().item() < 0 or reference_indices.max().item() >= len(
            self.references
        ):
            raise IndexError("reference index is outside the block sampler")
        references = [self.references[index] for index in reference_indices.tolist()]
        features = torch.stack(
            [
                self._features(life_index, reference)
                for life_index, reference in zip(life_indices.tolist(), references, strict=True)
            ]
        ).to(device)
        targets = torch.zeros(
            life_indices.numel(),
            self.decoder_config.block_rows,
            self.decoder_config.block_cols,
            dtype=torch.float32,
            device=device,
        )
        valid_masks = torch.zeros_like(targets)
        for index, (life_index, reference) in enumerate(
            zip(life_indices.tolist(), references, strict=True)
        ):
            delta = tensor_matrix_view(self.delta_states[life_index][reference.tensor_name])
            block = delta[
                reference.row_start : reference.row_end,
                reference.col_start : reference.col_end,
            ].to(device=device, dtype=torch.float32)
            scale = self.tensor_scales[reference.tensor_name]
            targets[
                index,
                : reference.row_end - reference.row_start,
                : reference.col_end - reference.col_start,
            ] = block / scale
            valid_masks[
                index,
                : reference.row_end - reference.row_start,
                : reference.col_end - reference.col_start,
            ] = 1.0
        return BlockBatch(
            life_indices=life_indices.to(device),
            reference_indices=reference_indices.to(device),
            layer_slots=torch.tensor(
                [reference.layer_slot for reference in references],
                dtype=torch.long,
                device=device,
            ),
            tensor_indices=torch.tensor(
                [reference.tensor_index for reference in references],
                dtype=torch.long,
                device=device,
            ),
            role_ids=torch.tensor(
                [reference.role_id for reference in references],
                dtype=torch.long,
                device=device,
            ),
            features=features,
            targets=targets,
            valid_masks=valid_masks,
        )


def _decoder_inputs(
    codes: MultiLifeGenomeCodes,
    batch: BlockBatch,
) -> dict[str, torch.Tensor]:
    result = {
        "global_codes": codes.global_codes[batch.life_indices],
        "layer_codes": codes.layer_codes[
            batch.life_indices,
            batch.layer_slots,
        ],
        "tensor_codes": codes.tensor_codes[
            batch.life_indices,
            batch.tensor_indices,
        ],
        "role_ids": batch.role_ids,
        "features": batch.features,
    }
    block_codes = codes.block_for_batch(batch.life_indices, batch.reference_indices)
    if block_codes is not None:
        result["block_codes"] = block_codes
    elif codes.block_code_dim:
        result["block_codes"] = torch.zeros(
            batch.life_indices.shape[0],
            codes.block_code_dim,
            dtype=batch.features.dtype,
            device=batch.features.device,
        )
    return result


def _block_code_storage_dtype(config: BlockDecoderConfig) -> torch.dtype:
    return torch.float16 if config.block_code_storage_dtype == "float16" else torch.float32


@torch.no_grad()
def materialize_residual_block_codes(
    *,
    decoder: RoleConditionedBlockDecoder,
    codes: MultiLifeGenomeCodes,
    sampler: MultiLifeBlockSampler,
    life_index: int,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    config = decoder.config
    if config.block_code_mode != "residual":
        raise ValueError("residual block-code materialization requires a residual decoder")
    if life_index < 0 or life_index >= sampler.life_count:
        raise IndexError("life index is outside the block sampler")
    result = torch.empty(
        len(sampler.references),
        config.block_code_dim,
        dtype=_block_code_storage_dtype(config),
        device="cpu",
    )
    for start in range(0, len(sampler.references), batch_size):
        stop = min(start + batch_size, len(sampler.references))
        batch = sampler.make_indexed_batch(
            life_indices=torch.full((stop - start,), life_index, dtype=torch.long),
            reference_indices=torch.arange(start, stop, dtype=torch.long),
            device=device,
        )
        structured = decoder(**_decoder_inputs(codes, batch))
        residual = (batch.targets - structured) * batch.valid_masks
        result[start:stop] = (
            residual.flatten(1)
            .to(dtype=_block_code_storage_dtype(config), device="cpu")
            .contiguous()
        )
    return result


def genome_program_from_codes(
    *,
    codes: Mapping[str, torch.Tensor],
    tensor_specs: Sequence[TensorSpec],
    tied_groups: Sequence[Sequence[str]],
    decoder_config: BlockDecoderConfig,
    role_to_id: Mapping[str, int],
    layer_to_slot: Mapping[int | None, int],
    role_scales: Mapping[str, float],
    tensor_scales: Mapping[str, float],
    interpreter_info: Mapping[str, Any],
    candidate_id: str,
    manifest_metadata: Mapping[str, Any],
) -> GenomeProgram:
    expected = {"global_code", "layer_codes", "tensor_codes"}
    if decoder_config.block_code_dim:
        expected.add("block_codes")
    if set(codes) != expected:
        raise ValueError(
            f"genome code keys differ; missing={sorted(expected - set(codes))}, "
            f"extra={sorted(set(codes) - expected)}"
        )
    records, aliases = make_records(tensor_specs, tied_groups)
    payload: dict[str, torch.Tensor] = {
        "shared.global_code": codes["global_code"].detach().to(torch.float32).cpu(),
        "shared.layer_codes": codes["layer_codes"].detach().to(torch.float32).cpu(),
    }
    max_layer = max(
        (spec.layer_index for spec in tensor_specs if spec.layer_index is not None),
        default=0,
    )
    block_cursor = 0
    for record in records:
        spec = tensor_specs[record.canonical_index]
        if record.tensor_name in aliases:
            record.components.append(
                GenomeComponent(
                    COPY_FROM_TIED,
                    arguments={"owner": aliases[record.tensor_name]},
                )
            )
            continue
        key = f"t{record.canonical_index:05d}.tensor_code"
        payload[key] = (
            codes["tensor_codes"][record.canonical_index].detach().to(torch.float32).cpu()
        )
        rows, cols = tensor_matrix_shape(spec.shape)
        payload_keys = [key]
        arguments: dict[str, Any] = {
            "role": spec.role,
            "shape": list(spec.shape),
            "matrix_shape": [rows, cols],
            "layer_slot": layer_to_slot[spec.layer_index],
            "normalized_layer": (
                -1.0 if spec.layer_index is None else spec.layer_index / max(max_layer, 1)
            ),
            "global_code_key": "shared.global_code",
            "layer_codes_key": "shared.layer_codes",
            "block_rows": decoder_config.block_rows,
            "block_cols": decoder_config.block_cols,
            "scale": float(tensor_scales[record.tensor_name]),
        }
        if decoder_config.block_code_dim:
            block_count = math.ceil(rows / decoder_config.block_rows) * math.ceil(
                cols / decoder_config.block_cols
            )
            block_key = f"t{record.canonical_index:05d}.block_codes"
            payload[block_key] = (
                codes["block_codes"][block_cursor : block_cursor + block_count]
                .detach()
                .to(_block_code_storage_dtype(decoder_config))
                .cpu()
            )
            payload_keys.append(block_key)
            arguments["block_codes_key"] = block_key
            block_cursor += block_count
        record.components.append(
            GenomeComponent(
                NEURAL_BLOCK_FIELD,
                payload_keys=payload_keys,
                arguments=arguments,
            )
        )
    if decoder_config.block_code_dim and block_cursor != codes["block_codes"].shape[0]:
        raise ValueError("block code count does not match the untied tensor blocks")
    manifest = make_manifest(
        candidate_id=candidate_id,
        codec="shared_neural_genome_decoder",
        metadata=manifest_metadata,
    )
    manifest.update(
        {
            "research_level": manifest_metadata.get("research_level", "G0"),
            "interpreter_id": Path(str(interpreter_info["path"])).name,
            "interpreter_manifest_sha256": interpreter_info["manifest_sha256"],
            "interpreter_decoder_sha256": interpreter_info["decoder_sha256"],
            "interpreter_bytes": int(interpreter_info["bytes"]),
            "shared_payload_keys": ["shared.global_code", "shared.layer_codes"],
            "codec_config": {
                "decoder": decoder_config.to_dict(),
                "role_to_id": {str(key): int(value) for key, value in role_to_id.items()},
                "layer_to_slot": {
                    "none" if key is None else str(key): int(value)
                    for key, value in layer_to_slot.items()
                },
                "role_scales": {str(key): float(value) for key, value in role_scales.items()},
                "tensor_scales": {str(key): float(value) for key, value in tensor_scales.items()},
                "block_count": block_cursor,
            },
        }
    )
    return GenomeProgram(manifest=manifest, records=records, payload_tensors=payload)


def interpreter_artifact_info(path: Path) -> dict[str, Any]:
    manifest_path = path / "manifest.json"
    manifest = read_json(manifest_path)
    return {
        "path": str(path),
        "manifest_sha256": sha256_file(manifest_path),
        "decoder_sha256": manifest["decoder_sha256"],
        "bytes": directory_size(path),
    }


def train_shared_decoder(
    lives: Sequence[CanonicalModelLife],
    *,
    tensor_specs: Sequence[TensorSpec],
    tied_groups: Sequence[Sequence[str]],
    output_path: str | Path,
    decoder_config: BlockDecoderConfig,
    training_config: SharedDecoderTrainingConfig,
) -> dict[str, Any]:
    if not lives:
        raise ValueError("shared decoder training requires model lives")
    if any(life.split != "training" for life in lives):
        raise ValueError("shared decoder training accepts only training-split lives")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    feature_probe = make_block_features(
        torch.zeros(7),
        torch.zeros(decoder_config.block_rows, decoder_config.block_cols),
        decoder_config,
    )
    if feature_probe.numel() != decoder_config.feature_dim:
        raise ValueError("shared decoder feature construction does not match feature_dim")
    random.seed(training_config.seed)
    torch.manual_seed(training_config.seed)
    device = torch.device(training_config.device)
    base_states = [life.load_base() for life in lives]
    target_states = [life.load_target() for life in lives]
    sampler = MultiLifeBlockSampler(
        base_states=base_states,
        target_states=target_states,
        tensor_specs=tensor_specs,
        tied_groups=tied_groups,
        decoder_config=decoder_config,
    )
    decoder = RoleConditionedBlockDecoder(
        len(sampler.role_to_id),
        decoder_config,
    ).to(device)
    codes = MultiLifeGenomeCodes(
        life_count=len(lives),
        layer_count=len(sampler.layer_to_slot),
        tensor_count=len(tensor_specs),
        block_count=len(sampler.references),
        config=decoder_config,
    ).to(device)
    dense_code_parameters = [
        codes.global_codes,
        codes.layer_codes,
        codes.tensor_codes,
    ]
    optimizer = torch.optim.AdamW(
        [
            {
                "params": decoder.parameters(),
                "weight_decay": training_config.weight_decay,
            },
            {
                "params": dense_code_parameters,
                "weight_decay": training_config.code_weight_decay,
            },
        ],
        lr=training_config.learning_rate,
    )
    block_optimizer = (
        torch.optim.SparseAdam(
            [codes.block_codes.weight],
            lr=training_config.learning_rate,
        )
        if codes.block_codes is not None
        else None
    )
    generator = torch.Generator(device="cpu").manual_seed(training_config.seed + 1)
    metrics = []
    decoder.train()
    for update in range(1, training_config.updates + 1):
        batch = sampler.make_batch(
            batch_size=training_config.batch_size,
            generator=generator,
            device=device,
        )
        decoder_inputs = _decoder_inputs(codes, batch)
        predictions = decoder(**decoder_inputs)
        reconstruction = masked_block_mse(
            predictions,
            batch.targets,
            batch.valid_masks,
        )
        code_terms = [
            codes.global_codes.square().mean(),
            codes.layer_codes.square().mean(),
            codes.tensor_codes.square().mean(),
        ]
        if "block_codes" in decoder_inputs:
            code_terms.append(decoder_inputs["block_codes"].square().mean())
        code_rms = torch.stack(code_terms).mean()
        loss = reconstruction + training_config.code_weight_decay * code_rms
        optimizer.zero_grad(set_to_none=True)
        if block_optimizer is not None:
            block_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [*decoder.parameters(), *dense_code_parameters],
            training_config.grad_clip_norm,
        )
        optimizer.step()
        if block_optimizer is not None:
            block_optimizer.step()
        if (
            update == 1
            or update % training_config.log_every == 0
            or update == training_config.updates
        ):
            metrics.append(
                {
                    "update": update,
                    "normalized_block_mse": float(reconstruction.detach().item()),
                    "code_rms": float(code_rms.detach().sqrt().item()),
                }
            )
    decoder.eval()

    temp = temporary_directory(destination.parent, f".{destination.name}.building.")
    try:
        interpreter_path = temp / "interpreter"
        info = save_interpreter(
            decoder,
            role_to_id=sampler.role_to_id,
            layer_to_slot=sampler.layer_to_slot,
            role_scales=sampler.role_scales,
            path=interpreter_path,
            training_metadata={
                "method": (
                    "structured_decoder_with_canonical_residual_codes"
                    if decoder_config.block_code_mode == "residual"
                    else "joint_multi_life_autodecoder"
                ),
                "training_run_ids": [life.run_id for life in lives],
                "training_config": asdict(training_config),
                "target_endpoints_seen": True,
                "hidden_endpoints_seen": False,
            },
        )
        code_root = temp / "codes"
        genome_root = temp / "genomes"
        code_root.mkdir()
        genome_root.mkdir()
        code_records = []
        for index, life in enumerate(lives):
            life_codes = {
                name: value.detach().to(torch.float32).cpu()
                for name, value in codes.for_life(index).items()
            }
            if decoder_config.block_code_mode == "residual":
                life_codes["block_codes"] = materialize_residual_block_codes(
                    decoder=decoder,
                    codes=codes,
                    sampler=sampler,
                    life_index=index,
                    device=device,
                    batch_size=training_config.batch_size,
                )
            code_path = code_root / f"{life.run_id}.safetensors"
            save_file(life_codes, str(code_path))
            program = genome_program_from_codes(
                codes=life_codes,
                tensor_specs=tensor_specs,
                tied_groups=tied_groups,
                decoder_config=decoder_config,
                role_to_id=sampler.role_to_id,
                layer_to_slot=sampler.layer_to_slot,
                role_scales=sampler.role_scales,
                tensor_scales=sampler.tensor_scales,
                interpreter_info=info,
                candidate_id=f"{life.run_id}-fitted-genome",
                manifest_metadata={
                    "research_level": "G0",
                    "run_id": life.run_id,
                    "base_state_sha256": life.manifest["W0"]["canonical_state_sha256"],
                    "target_endpoint_seen_during_fit": True,
                },
            )
            program_path = genome_root / f"{life.run_id}.mgp"
            save_program(program, program_path)
            code_records.append(
                {
                    "run_id": life.run_id,
                    "code_file": str(code_path.relative_to(temp)),
                    "code_sha256": sha256_file(code_path),
                    "genome_path": str(program_path.relative_to(temp)),
                    "genome_bytes": directory_size(program_path),
                    "genome_sha256": sha256_directory(program_path),
                }
            )
        write_json(temp / "training_metrics.json", metrics, canonical=True)
        layout = GenomeCodeLayout(
            global_code_dim=decoder_config.global_code_dim,
            n_layers=len(sampler.layer_to_slot),
            layer_code_dim=decoder_config.layer_code_dim,
            n_tensors=len(tensor_specs),
            tensor_code_dim=decoder_config.tensor_code_dim,
        )
        manifest = {
            "format": "GENOME_SHARED_NEURAL_DECODER",
            "version": "0.1.0",
            "training_run_ids": [life.run_id for life in lives],
            "training_life_count": len(lives),
            "hidden_run_ids": [],
            "decoder_config": decoder_config.to_dict(),
            "training_config": asdict(training_config),
            "layout": layout.to_dict(),
            "block_layout": {
                "block_count": len(sampler.references),
                "block_code_dim": decoder_config.block_code_dim,
                "block_code_mode": decoder_config.block_code_mode,
                "block_code_storage_dtype": decoder_config.block_code_storage_dtype,
            },
            "role_to_id": sampler.role_to_id,
            "layer_to_slot": {
                "none" if key is None else str(key): value
                for key, value in sampler.layer_to_slot.items()
            },
            "role_scales": sampler.role_scales,
            "tensor_scales": sampler.tensor_scales,
            "interpreter": {
                "path": "interpreter",
                "manifest_sha256": info["manifest_sha256"],
                "decoder_sha256": info["decoder_sha256"],
                "bytes": info["bytes"],
            },
            "codes": code_records,
            "metrics_file": "training_metrics.json",
            "metrics_sha256": sha256_file(temp / "training_metrics.json"),
        }
        manifest["content_sha256"] = sha256_json(manifest)
        write_json(temp / "manifest.json", manifest, canonical=True)
        replace_directory_atomic(temp, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def load_shared_decoder(
    path: str | Path,
    *,
    device: str | torch.device = "cpu",
) -> tuple[NeuralBlockInterpreter, dict[str, Any], dict[int | None, int]]:
    root = Path(path).expanduser().resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    if not isinstance(manifest, dict):
        raise TypeError("shared decoder manifest must be an object")
    if (
        manifest.get("format") != "GENOME_SHARED_NEURAL_DECODER"
        or manifest.get("version") != "0.1.0"
    ):
        raise ValueError("unsupported shared neural decoder")
    content = dict(manifest)
    declared = content.pop("content_sha256", None)
    if sha256_json(content) != declared:
        raise ValueError("shared decoder manifest hash mismatch")
    interpreter_root = resolve_artifact_directory(
        root,
        manifest["interpreter"]["path"],
        field="interpreter.path",
    )
    interpreter = load_interpreter(interpreter_root, device=device)
    if (
        sha256_file(interpreter_root / "manifest.json")
        != manifest["interpreter"]["manifest_sha256"]
    ):
        raise ValueError("shared decoder interpreter manifest hash mismatch")
    raw_slots = manifest["layer_to_slot"]
    layer_to_slot = {
        None if key == "none" else int(key): int(value) for key, value in raw_slots.items()
    }
    return interpreter, manifest, layer_to_slot


def fitted_code_path(decoder_root: str | Path, run_id: str) -> Path:
    root = Path(decoder_root).expanduser().resolve(strict=True)
    manifest = read_json(root / "manifest.json")
    records = [record for record in manifest["codes"] if record["run_id"] == run_id]
    if len(records) != 1:
        raise KeyError(f"shared decoder has {len(records)} fitted codes for {run_id}")
    path = resolve_artifact_relative_file(
        root,
        records[0]["code_file"],
        field=f"codes.{run_id}.code_file",
    )
    if sha256_file(path) != records[0]["code_sha256"]:
        raise ValueError(f"fitted code artifact is invalid for {run_id}")
    return path


def decoder_tensor_scales(
    decoder_manifest: Mapping[str, Any],
    tensor_specs: Sequence[TensorSpec],
    tied_groups: Sequence[Sequence[str]],
    role_scales: Mapping[str, float],
) -> dict[str, float]:
    aliases = tied_owner_map(tied_groups)
    untied_specs = [spec for spec in tensor_specs if spec.name not in aliases]
    stored = decoder_manifest.get("tensor_scales")
    if stored is None:
        return {spec.name: float(role_scales[spec.role]) for spec in untied_specs}
    if not isinstance(stored, Mapping):
        raise TypeError("shared decoder tensor_scales must be an object")
    result = {str(name): float(scale) for name, scale in stored.items()}
    expected = {spec.name for spec in untied_specs}
    if set(result) != expected:
        raise ValueError("shared decoder tensor scales do not match untied tensor inventory")
    if any(not math.isfinite(scale) or scale <= 0 for scale in result.values()):
        raise ValueError("shared decoder tensor scales must be finite and positive")
    return result


def fit_genome_code_with_frozen_decoder(
    life: CanonicalModelLife,
    *,
    tensor_specs: Sequence[TensorSpec],
    tied_groups: Sequence[Sequence[str]],
    shared_decoder_path: str | Path,
    output_path: str | Path,
    config: LatentCodeFitConfig,
) -> dict[str, Any]:
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    interpreter, decoder_manifest, layer_to_slot = load_shared_decoder(
        shared_decoder_path,
        device=config.device,
    )
    decoder = interpreter.decoder
    decoder.requires_grad_(False)
    device = torch.device(config.device)
    tensor_scales = decoder_tensor_scales(
        decoder_manifest,
        tensor_specs,
        tied_groups,
        interpreter.role_scales,
    )
    sampler = MultiLifeBlockSampler(
        base_states=[life.load_base()],
        target_states=[life.load_target()],
        tensor_specs=tensor_specs,
        tied_groups=tied_groups,
        decoder_config=interpreter.config,
        role_to_id=interpreter.role_to_id,
        layer_to_slot=layer_to_slot,
        role_scales=interpreter.role_scales,
        tensor_scales=tensor_scales,
    )
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    codes = MultiLifeGenomeCodes(
        life_count=1,
        layer_count=len(layer_to_slot),
        tensor_count=len(tensor_specs),
        block_count=len(sampler.references),
        config=interpreter.config,
    ).to(device)
    dense_code_parameters = [
        codes.global_codes,
        codes.layer_codes,
        codes.tensor_codes,
    ]
    optimizer = torch.optim.AdamW(
        dense_code_parameters,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    block_optimizer = (
        torch.optim.SparseAdam(
            [codes.block_codes.weight],
            lr=config.learning_rate,
        )
        if codes.block_codes is not None
        else None
    )
    generator = torch.Generator(device="cpu").manual_seed(config.seed + 1)
    metrics = []
    for update in range(1, config.updates + 1):
        batch = sampler.make_batch(
            batch_size=config.batch_size,
            generator=generator,
            device=device,
        )
        prediction = decoder(**_decoder_inputs(codes, batch))
        loss = masked_block_mse(
            prediction,
            batch.targets,
            batch.valid_masks,
        )
        optimizer.zero_grad(set_to_none=True)
        if block_optimizer is not None:
            block_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dense_code_parameters, config.grad_clip_norm)
        optimizer.step()
        if block_optimizer is not None:
            block_optimizer.step()
        if update == 1 or update % config.log_every == 0 or update == config.updates:
            metrics.append(
                {
                    "update": update,
                    "normalized_block_mse": float(loss.detach().item()),
                }
            )
    life_codes = {
        name: value.detach().to(torch.float32).cpu() for name, value in codes.for_life(0).items()
    }
    if interpreter.config.block_code_mode == "residual":
        life_codes["block_codes"] = materialize_residual_block_codes(
            decoder=decoder,
            codes=codes,
            sampler=sampler,
            life_index=0,
            device=device,
            batch_size=config.batch_size,
        )
    decoder_root = Path(shared_decoder_path).expanduser().resolve(strict=True)
    interpreter_path = resolve_artifact_directory(
        decoder_root,
        decoder_manifest["interpreter"]["path"],
        field="interpreter.path",
    )
    info = interpreter_artifact_info(interpreter_path)
    program = genome_program_from_codes(
        codes=life_codes,
        tensor_specs=tensor_specs,
        tied_groups=tied_groups,
        decoder_config=interpreter.config,
        role_to_id=interpreter.role_to_id,
        layer_to_slot=layer_to_slot,
        role_scales=interpreter.role_scales,
        tensor_scales=tensor_scales,
        interpreter_info=info,
        candidate_id=f"{life.run_id}-frozen-decoder-fitted-genome",
        manifest_metadata={
            "research_level": "G0",
            "run_id": life.run_id,
            "base_state_sha256": life.manifest["W0"]["canonical_state_sha256"],
            "target_endpoint_seen_during_fit": True,
            "shared_decoder_frozen": True,
        },
    )
    temp = temporary_directory(destination.parent, f".{destination.name}.building.")
    try:
        code_path = temp / "genome_code.safetensors"
        save_file(life_codes, str(code_path))
        program_path = temp / "genome.mgp"
        save_program(program, program_path)
        write_json(temp / "training_metrics.json", metrics, canonical=True)
        manifest = {
            "format": "GENOME_FITTED_LIFE_CODE",
            "version": "0.1.0",
            "run_id": life.run_id,
            "split": life.split,
            "research_level": "G0",
            "shared_decoder_manifest_sha256": sha256_file(decoder_root / "manifest.json"),
            "target_endpoint_seen_during_fit": True,
            "code_file": "genome_code.safetensors",
            "code_sha256": sha256_file(code_path),
            "genome_path": "genome.mgp",
            "genome_bytes": directory_size(program_path),
            "genome_sha256": sha256_directory(program_path),
            "training_config": asdict(config),
            "metrics_file": "training_metrics.json",
            "metrics_sha256": sha256_file(temp / "training_metrics.json"),
        }
        manifest["content_sha256"] = sha256_json(manifest)
        write_json(temp / "manifest.json", manifest, canonical=True)
        replace_directory_atomic(temp, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise
