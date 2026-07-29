from __future__ import annotations

import math
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from torch import nn

from ..hashing import sha256_file, sha256_json
from ..io import (
    read_json,
    replace_directory_atomic,
    resolve_artifact_member,
    temporary_directory,
    write_json,
)
from ..types import GenomeComponent, GenomeProgram, TensorSpec


@dataclass(frozen=True)
class BlockDecoderConfig:
    block_rows: int = 16
    block_cols: int = 16
    global_code_dim: int = 64
    layer_code_dim: int = 32
    tensor_code_dim: int = 32
    role_embedding_dim: int = 16
    feature_dim: int = 7
    hidden_dim: int = 256
    depth: int = 4
    output_scale_clip: float = 8.0

    def __post_init__(self) -> None:
        positive_integer_fields = (
            "block_rows",
            "block_cols",
            "global_code_dim",
            "layer_code_dim",
            "tensor_code_dim",
            "role_embedding_dim",
            "feature_dim",
            "hidden_dim",
        )
        for name in positive_integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.depth, bool) or not isinstance(self.depth, int) or self.depth < 0:
            raise ValueError("depth must be a non-negative integer")
        if not math.isfinite(self.output_scale_clip) or self.output_scale_clip <= 0:
            raise ValueError("output_scale_clip must be a finite positive number")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> BlockDecoderConfig:
        expected = set(cls.__dataclass_fields__)
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise ValueError(f"block decoder config mismatch; missing={missing}, extra={extra}")
        return cls(**dict(value))


class GenomeCodeBank(nn.Module):
    def __init__(self, n_layers: int, n_tensors: int, config: BlockDecoderConfig) -> None:
        super().__init__()
        self.global_code = nn.Parameter(torch.zeros(config.global_code_dim))
        self.layer_codes = nn.Parameter(torch.zeros(max(n_layers, 1), config.layer_code_dim))
        self.tensor_codes = nn.Parameter(torch.zeros(n_tensors, config.tensor_code_dim))
        nn.init.normal_(self.global_code, std=0.02)
        nn.init.normal_(self.layer_codes, std=0.02)
        nn.init.normal_(self.tensor_codes, std=0.02)


class RoleConditionedBlockDecoder(nn.Module):
    def __init__(self, n_roles: int, config: BlockDecoderConfig) -> None:
        super().__init__()
        if isinstance(n_roles, bool) or not isinstance(n_roles, int) or n_roles < 1:
            raise ValueError("n_roles must be a positive integer")
        self.config = config
        self.role_embedding = nn.Embedding(n_roles, config.role_embedding_dim)
        input_dim = (
            config.global_code_dim
            + config.layer_code_dim
            + config.tensor_code_dim
            + config.role_embedding_dim
            + config.feature_dim
        )
        layers: list[nn.Module] = []
        dimension = input_dim
        for _ in range(config.depth):
            layers.extend([nn.Linear(dimension, config.hidden_dim), nn.SiLU()])
            dimension = config.hidden_dim
        layers.append(nn.Linear(dimension, config.block_rows * config.block_cols))
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        global_codes: torch.Tensor,
        layer_codes: torch.Tensor,
        tensor_codes: torch.Tensor,
        role_ids: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if global_codes.ndim == 1:
            global_codes = global_codes.unsqueeze(0)
        if layer_codes.ndim == 1:
            layer_codes = layer_codes.unsqueeze(0)
        if tensor_codes.ndim == 1:
            tensor_codes = tensor_codes.unsqueeze(0)
        role = self.role_embedding(role_ids)
        value = torch.cat([global_codes, layer_codes, tensor_codes, role, features], dim=-1)
        output = self.network(value)
        output = torch.tanh(output / self.config.output_scale_clip) * self.config.output_scale_clip
        return output.view(-1, self.config.block_rows, self.config.block_cols)


@dataclass(frozen=True)
class BlockReference:
    tensor_name: str
    tensor_index: int
    role_id: int
    layer_slot: int
    block_row: int
    block_col: int
    row_start: int
    row_end: int
    col_start: int
    col_end: int
    rows: int
    cols: int


class LazyDeltaBlockDataset:
    def __init__(
        self,
        base_state: Mapping[str, torch.Tensor],
        delta_state: Mapping[str, torch.Tensor],
        tensor_specs: Sequence[TensorSpec],
        *,
        tied_aliases: Mapping[str, str],
        role_to_id: Mapping[str, int],
        layer_to_slot: Mapping[int | None, int],
        role_scales: Mapping[str, float],
        config: BlockDecoderConfig,
    ) -> None:
        self.base_state = base_state
        self.delta_state = delta_state
        self.specs = list(tensor_specs)
        self.role_to_id = dict(role_to_id)
        self.layer_to_slot = dict(layer_to_slot)
        self.role_scales = dict(role_scales)
        self.config = config
        self.references: list[BlockReference] = []
        self.spec_by_name = {spec.name: spec for spec in tensor_specs}
        for spec in tensor_specs:
            if spec.name in tied_aliases:
                continue
            rows, cols = tensor_matrix_shape(spec.shape)
            for row_start in range(0, rows, config.block_rows):
                for col_start in range(0, cols, config.block_cols):
                    self.references.append(
                        BlockReference(
                            tensor_name=spec.name,
                            tensor_index=spec.canonical_index,
                            role_id=self.role_to_id[spec.role],
                            layer_slot=self.layer_to_slot[spec.layer_index],
                            block_row=row_start // config.block_rows,
                            block_col=col_start // config.block_cols,
                            row_start=row_start,
                            row_end=min(row_start + config.block_rows, rows),
                            col_start=col_start,
                            col_end=min(col_start + config.block_cols, cols),
                            rows=rows,
                            cols=cols,
                        )
                    )
        if not self.references:
            raise ValueError("neural block dataset contains no matrix tensors")

    def __len__(self) -> int:
        return len(self.references)

    def _features(self, reference: BlockReference) -> torch.Tensor:
        base = tensor_matrix_view(self.base_state[reference.tensor_name].to(torch.float32))
        block = base[
            reference.row_start : reference.row_end,
            reference.col_start : reference.col_end,
        ]
        row_blocks = max(math.ceil(reference.rows / self.config.block_rows), 1)
        col_blocks = max(math.ceil(reference.cols / self.config.block_cols), 1)
        layer_count = max(len(self.layer_to_slot) - (1 if None in self.layer_to_slot else 0), 1)
        layer_value = (
            -1.0
            if self.spec_by_name[reference.tensor_name].layer_index is None
            else float(self.spec_by_name[reference.tensor_name].layer_index)
            / max(layer_count - 1, 1)
        )
        metadata = torch.tensor(
            [
                layer_value,
                reference.block_row / max(row_blocks - 1, 1),
                reference.block_col / max(col_blocks - 1, 1),
                math.log1p(reference.rows) / 16.0,
                math.log1p(reference.cols) / 16.0,
                float(block.mean().item()),
                float(block.std(unbiased=False).item()) if block.numel() > 1 else 0.0,
            ],
            dtype=torch.float32,
        )
        return make_block_features(metadata, block, self.config)

    def make_batch(
        self,
        indices: torch.Tensor,
        code_bank: GenomeCodeBank,
        *,
        device: torch.device,
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        references = [self.references[int(index)] for index in indices.tolist()]
        features = torch.stack([self._features(reference) for reference in references]).to(device)
        role_ids = torch.tensor(
            [reference.role_id for reference in references], dtype=torch.long, device=device
        )
        layer_slots = torch.tensor(
            [reference.layer_slot for reference in references], dtype=torch.long, device=device
        )
        tensor_indices = torch.tensor(
            [reference.tensor_index for reference in references], dtype=torch.long, device=device
        )
        targets = torch.zeros(
            len(references), self.config.block_rows, self.config.block_cols, device=device
        )
        for batch_index, reference in enumerate(references):
            spec = self.spec_by_name[reference.tensor_name]
            scale = max(float(self.role_scales[spec.role]), 1e-12)
            block = tensor_matrix_view(self.delta_state[reference.tensor_name])[
                reference.row_start : reference.row_end,
                reference.col_start : reference.col_end,
            ].to(torch.float32)
            targets[
                batch_index,
                : reference.row_end - reference.row_start,
                : reference.col_end - reference.col_start,
            ] = block.to(device) / scale
        inputs = {
            "global_codes": code_bank.global_code.unsqueeze(0).expand(len(references), -1),
            "layer_codes": code_bank.layer_codes[layer_slots],
            "tensor_codes": code_bank.tensor_codes[tensor_indices],
            "role_ids": role_ids,
            "features": features,
        }
        return inputs, targets


@dataclass(frozen=True)
class NeuralInterpreterManifest:
    format: str
    version: str
    config: dict[str, Any]
    role_to_id: dict[str, int]
    layer_to_slot: dict[str, int]
    role_scales: dict[str, float]
    decoder_file: str
    decoder_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class NeuralBlockInterpreter:
    def __init__(
        self,
        decoder: RoleConditionedBlockDecoder,
        manifest: Mapping[str, Any],
        *,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.decoder = decoder.to(self.device).eval()
        self.manifest = dict(manifest)
        self.config = BlockDecoderConfig.from_dict(self.manifest["config"])
        self.role_to_id = {str(k): int(v) for k, v in self.manifest["role_to_id"].items()}
        self.role_scales = {str(k): float(v) for k, v in self.manifest["role_scales"].items()}

    def decode_tensor_from_component(
        self,
        record_name: str,
        component: GenomeComponent,
        program: GenomeProgram,
        *,
        base_tensor: torch.Tensor,
    ) -> torch.Tensor:
        args = component.arguments
        role = str(args["role"])
        role_id = self.role_to_id[role]
        tensor_code = program.payload_tensors[component.payload_keys[0]].to(self.device)
        global_code = program.payload_tensors[str(args["global_code_key"])].to(self.device)
        layer_codes = program.payload_tensors[str(args["layer_codes_key"])].to(self.device)
        layer_slot = int(args["layer_slot"])
        layer_code = layer_codes[layer_slot]
        original_shape = tuple(int(x) for x in args["shape"])
        raw_matrix_shape = args.get("matrix_shape", args["shape"])
        rows, cols = (int(x) for x in raw_matrix_shape)
        if rows * cols != math.prod(original_shape):
            raise ValueError(f"invalid neural matrix shape for {record_name}")
        base_matrix = base_tensor.reshape(rows, cols)
        result = torch.zeros(rows, cols, dtype=torch.float32, device=self.device)
        items: list[tuple[int, int, int, int, torch.Tensor]] = []
        for row_start in range(0, rows, self.config.block_rows):
            for col_start in range(0, cols, self.config.block_cols):
                row_end = min(row_start + self.config.block_rows, rows)
                col_end = min(col_start + self.config.block_cols, cols)
                base_block = base_matrix[row_start:row_end, col_start:col_end].to(torch.float32)
                row_blocks = max(math.ceil(rows / self.config.block_rows), 1)
                col_blocks = max(math.ceil(cols / self.config.block_cols), 1)
                metadata = torch.tensor(
                    [
                        float(args["normalized_layer"]),
                        (row_start // self.config.block_rows) / max(row_blocks - 1, 1),
                        (col_start // self.config.block_cols) / max(col_blocks - 1, 1),
                        math.log1p(rows) / 16.0,
                        math.log1p(cols) / 16.0,
                        float(base_block.mean().item()),
                        float(base_block.std(unbiased=False).item())
                        if base_block.numel() > 1
                        else 0.0,
                    ],
                    dtype=torch.float32,
                )
                feature = make_block_features(metadata, base_block, self.config)
                items.append((row_start, row_end, col_start, col_end, feature))

        batch_size = int(args.get("decode_batch_size", 256))
        scale = float(args.get("scale", self.role_scales[role]))
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError(f"invalid neural block scale for {record_name}")
        for start in range(0, len(items), batch_size):
            chunk = items[start : start + batch_size]
            features = torch.stack([item[4] for item in chunk]).to(self.device)
            count = len(chunk)
            predictions = (
                self.decoder(
                    global_code.unsqueeze(0).expand(count, -1),
                    layer_code.unsqueeze(0).expand(count, -1),
                    tensor_code.unsqueeze(0).expand(count, -1),
                    torch.full((count,), role_id, dtype=torch.long, device=self.device),
                    features,
                )
                * scale
            )
            for prediction, (row_start, row_end, col_start, col_end, _) in zip(
                predictions, chunk, strict=True
            ):
                result[row_start:row_end, col_start:col_end] = prediction[
                    : row_end - row_start, : col_end - col_start
                ]
        return result.reshape(original_shape)


def tensor_matrix_shape(shape: Sequence[int]) -> tuple[int, int]:
    dimensions = tuple(int(value) for value in shape)
    if not dimensions:
        return 1, 1
    if len(dimensions) == 1:
        return 1, dimensions[0]
    return dimensions[0], math.prod(dimensions[1:])


def tensor_matrix_view(tensor: torch.Tensor) -> torch.Tensor:
    rows, cols = tensor_matrix_shape(tuple(tensor.shape))
    return tensor.reshape(rows, cols)


def append_base_block_features(
    metadata: torch.Tensor,
    block: torch.Tensor,
    config: BlockDecoderConfig,
) -> torch.Tensor:
    if config.feature_dim == metadata.numel():
        return metadata
    expected = metadata.numel() + config.block_rows * config.block_cols
    if config.feature_dim != expected:
        raise ValueError(
            f"feature_dim must be {metadata.numel()} or {expected}, got {config.feature_dim}"
        )
    padded = torch.zeros(config.block_rows, config.block_cols, dtype=torch.float32)
    rows, cols = block.shape
    padded[:rows, :cols] = block.detach().to(dtype=torch.float32, device="cpu")
    return torch.cat([metadata, padded.flatten()])


def make_block_features(
    metadata: torch.Tensor,
    block: torch.Tensor,
    config: BlockDecoderConfig,
) -> torch.Tensor:
    if metadata.numel() != 7:
        raise ValueError(f"block metadata must contain seven values, got {metadata.numel()}")
    block_values = config.block_rows * config.block_cols
    if config.feature_dim in {7, 7 + block_values}:
        return append_base_block_features(metadata, block, config)
    include_base_block = config.feature_dim > 7 + block_values
    fixed_values = 7 + (block_values if include_base_block else 0)
    coordinate_values = config.feature_dim - fixed_values
    if coordinate_values < 4 or coordinate_values % 4:
        raise ValueError("feature_dim must add four Fourier values per coordinate frequency")
    frequency_count = coordinate_values // 4
    row_coordinate = float(metadata[1].item())
    column_coordinate = float(metadata[2].item())
    fourier = []
    for index in range(frequency_count):
        frequency = math.pi * (2**index)
        fourier.extend(
            [
                math.sin(frequency * row_coordinate),
                math.cos(frequency * row_coordinate),
                math.sin(frequency * column_coordinate),
                math.cos(frequency * column_coordinate),
            ]
        )
    expanded = torch.cat(
        [
            metadata,
            torch.tensor(fourier, dtype=torch.float32),
        ]
    )
    if not include_base_block:
        return expanded
    padded = torch.zeros(config.block_rows, config.block_cols, dtype=torch.float32)
    rows, cols = block.shape
    padded[:rows, :cols] = block.detach().to(dtype=torch.float32, device="cpu")
    return torch.cat([expanded, padded.flatten()])


def save_interpreter(
    decoder: RoleConditionedBlockDecoder,
    *,
    role_to_id: Mapping[str, int],
    layer_to_slot: Mapping[int | None, int],
    role_scales: Mapping[str, float],
    path: str | Path,
    training_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    destination = Path(path)
    if destination.exists():
        raise FileExistsError(destination)
    temp = temporary_directory(destination.parent, f".{destination.name}.building.")
    try:
        decoder_file = temp / "decoder.safetensors"
        save_file(
            {
                name: tensor.detach().contiguous().cpu()
                for name, tensor in decoder.state_dict().items()
            },
            str(decoder_file),
        )
        manifest = NeuralInterpreterManifest(
            format="GENOME_NEURAL_INTERPRETER",
            version="0.1.0",
            config=decoder.config.to_dict(),
            role_to_id={str(k): int(v) for k, v in role_to_id.items()},
            layer_to_slot={
                "none" if k is None else str(k): int(v) for k, v in layer_to_slot.items()
            },
            role_scales={str(k): float(v) for k, v in role_scales.items()},
            decoder_file="decoder.safetensors",
            decoder_sha256=sha256_file(decoder_file),
        ).to_dict()
        if training_metadata is not None:
            manifest["training_metadata"] = dict(training_metadata)
        manifest["manifest_content_sha256"] = sha256_json(manifest)
        write_json(temp / "manifest.json", manifest, canonical=True)
        replace_directory_atomic(temp, destination)
        return {
            "path": str(destination),
            "manifest_sha256": sha256_file(destination / "manifest.json"),
            "decoder_sha256": manifest["decoder_sha256"],
            "bytes": sum(item.stat().st_size for item in destination.rglob("*") if item.is_file()),
        }
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _lower_sha256(value: object, *, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _string_int_mapping(value: object, *, field: str) -> dict[str, int]:
    if not isinstance(value, dict) or any(
        not isinstance(key, str) or isinstance(item, bool) or not isinstance(item, int)
        for key, item in value.items()
    ):
        raise TypeError(f"{field} must be an object mapping strings to integers")
    return {str(key): int(item) for key, item in value.items()}


def load_interpreter(
    path: str | Path, *, device: str | torch.device = "cpu"
) -> NeuralBlockInterpreter:
    root = Path(path).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise ValueError(f"neural interpreter path is not a directory: {root}")
    manifest_path = resolve_artifact_member(root, "manifest.json", field="manifest_file")
    raw_manifest = read_json(manifest_path)
    if not isinstance(raw_manifest, dict) or any(not isinstance(key, str) for key in raw_manifest):
        raise TypeError("neural interpreter manifest must be an object with string keys")
    manifest: dict[str, Any] = raw_manifest
    if manifest.get("format") != "GENOME_NEURAL_INTERPRETER":
        raise ValueError("not a GENOME neural interpreter")
    if manifest.get("version") != "0.1.0":
        raise ValueError(f"unsupported neural interpreter version: {manifest.get('version')!r}")

    decoder_file = resolve_artifact_member(root, manifest.get("decoder_file"), field="decoder_file")
    if sha256_file(decoder_file) != _lower_sha256(
        manifest.get("decoder_sha256"), field="decoder_sha256"
    ):
        raise ValueError("neural interpreter decoder hash mismatch")
    declared_manifest_hash = _lower_sha256(
        manifest.get("manifest_content_sha256"), field="manifest_content_sha256"
    )
    content = dict(manifest)
    content.pop("manifest_content_sha256", None)
    if sha256_json(content) != declared_manifest_hash:
        raise ValueError("neural interpreter manifest content hash mismatch")

    raw_config = manifest.get("config")
    if not isinstance(raw_config, dict):
        raise TypeError("neural interpreter config must be an object")
    config = BlockDecoderConfig.from_dict(raw_config)
    role_to_id = _string_int_mapping(manifest.get("role_to_id"), field="role_to_id")
    if not role_to_id or sorted(role_to_id.values()) != list(range(len(role_to_id))):
        raise ValueError("role_to_id values must be contiguous from zero")
    layer_to_slot = _string_int_mapping(manifest.get("layer_to_slot"), field="layer_to_slot")
    if (
        not layer_to_slot
        or len(set(layer_to_slot.values())) != len(layer_to_slot)
        or sorted(layer_to_slot.values()) != list(range(len(layer_to_slot)))
    ):
        raise ValueError("layer_to_slot values must be contiguous from zero")
    raw_scales = manifest.get("role_scales")
    if not isinstance(raw_scales, dict) or set(raw_scales) != set(role_to_id):
        raise ValueError("role_scales must contain exactly the declared roles")
    for role, scale in raw_scales.items():
        if isinstance(scale, bool) or not isinstance(scale, (int, float)):
            raise TypeError(f"role scale must be numeric: {role}")
        if not math.isfinite(float(scale)) or float(scale) <= 0:
            raise ValueError(f"role scale must be finite and positive: {role}")

    decoder = RoleConditionedBlockDecoder(len(role_to_id), config)
    decoder.load_state_dict(load_file(str(decoder_file), device=str(device)), strict=True)
    return NeuralBlockInterpreter(decoder, manifest, device=device)
