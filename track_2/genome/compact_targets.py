from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from .codecs.common import make_manifest, make_records
from .io import write_json
from .mgp.opcodes import COPY_FROM_TIED, LOW_RANK, QUANTIZED_DELTA
from .mgp.policy import CompilerTargetAudit, CompilerTargetPolicy, audit_compiler_target
from .mgp.serializer import save_program
from .state import compute_delta
from .types import GenomeComponent, GenomeProgram, TensorSpec


@dataclass(frozen=True)
class CompactTargetConfig:
    """Transparent first target language for compiler supervision.

    The fitter never emits dense matrix deltas or exact residuals. Matrix information is carried by
    canonical low-rank factors; small vectors may be int8 because their aggregate byte share is
    bounded by the compiler-target policy.
    """

    target_fraction_of_fp16_delta: float = 0.10
    max_rank: int = 64
    factor_dtype: str = "float16"
    vector_bits: int = 8
    direct_vector_numel_limit: int = 4096

    def __post_init__(self) -> None:
        if not 0.0 < self.target_fraction_of_fp16_delta < 1.0:
            raise ValueError("target_fraction_of_fp16_delta must lie in (0, 1)")
        if isinstance(self.max_rank, bool) or not isinstance(self.max_rank, int) or self.max_rank < 0:
            raise ValueError("max_rank must be a non-negative integer")
        if self.factor_dtype not in {"float16", "float32"}:
            raise ValueError("factor_dtype must be float16 or float32")
        if self.vector_bits != 8:
            raise ValueError("the compact target fitter currently supports int8 vectors only")
        if (
            isinstance(self.direct_vector_numel_limit, bool)
            or not isinstance(self.direct_vector_numel_limit, int)
            or self.direct_vector_numel_limit < 0
        ):
            raise ValueError("direct_vector_numel_limit must be non-negative")

    @property
    def torch_factor_dtype(self) -> torch.dtype:
        return torch.float16 if self.factor_dtype == "float16" else torch.float32


@dataclass(frozen=True)
class CompactTargetResult:
    program: GenomeProgram
    audit: CompilerTargetAudit
    allocated_ranks: dict[str, int]
    logical_factor_bytes: int
    logical_vector_bytes: int


@dataclass(frozen=True)
class SerializedCompactTarget:
    path: Path
    audit: CompilerTargetAudit
    artifact_sizes: dict[str, Any]


@dataclass(frozen=True)
class _SVD:
    u: torch.Tensor
    s: torch.Tensor
    vh: torch.Tensor


def canonicalize_svd(
    u: torch.Tensor,
    s: torch.Tensor,
    vh: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Remove the per-component sign ambiguity from an SVD deterministically."""

    if u.ndim != 2 or s.ndim != 1 or vh.ndim != 2:
        raise ValueError("canonicalize_svd expects U, S, and Vh")
    if u.shape[1] != s.numel() or vh.shape[0] != s.numel():
        raise ValueError("SVD factor dimensions differ")
    u = u.clone()
    vh = vh.clone()
    for index in range(s.numel()):
        column = u[:, index]
        pivot = int(torch.argmax(column.abs()).item())
        if float(column[pivot].item()) < 0.0:
            u[:, index].neg_()
            vh[index].neg_()
    return u, s, vh


def _factorize(delta: torch.Tensor) -> _SVD:
    if delta.ndim != 2:
        raise ValueError("only matrix deltas can be factorized")
    u, s, vh = torch.linalg.svd(delta.to(torch.float32), full_matrices=False)
    u, s, vh = canonicalize_svd(u, s, vh)
    return _SVD(u=u, s=s, vh=vh)


def _quantize_int8(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = float(value.detach().abs().max().item()) if value.numel() else 0.0
    scale = maximum / 127.0 if maximum > 0.0 else 1.0
    quantized = torch.round(value.to(torch.float32) / scale).clamp(-127, 127).to(torch.int8)
    return quantized.contiguous(), torch.tensor(scale, dtype=torch.float32)


def _allocate_ranks(
    factors: Mapping[str, _SVD],
    specs: Mapping[str, TensorSpec],
    *,
    byte_budget: int,
    factor_dtype: torch.dtype,
    max_rank: int,
) -> tuple[dict[str, int], int]:
    if byte_budget < 0:
        raise ValueError("byte budget must be non-negative")
    element_size = torch.empty((), dtype=factor_dtype).element_size()
    candidates: list[tuple[float, str, int, int]] = []
    for name, factor in factors.items():
        spec = specs[name]
        per_component_bytes = (spec.shape[0] + spec.shape[1] + 1) * element_size
        component_limit = min(max_rank, factor.s.numel())
        for component_index in range(component_limit):
            score = float(factor.s[component_index].square().item()) / max(
                per_component_bytes, 1
            )
            candidates.append((score, name, component_index, per_component_bytes))
    candidates.sort(key=lambda item: (-item[0], item[1], item[2]))

    ranks = {name: 0 for name in factors}
    used_bytes = 0
    for _score, name, component_index, cost in candidates:
        if component_index != ranks[name]:
            continue
        if used_bytes + cost > byte_budget:
            continue
        ranks[name] += 1
        used_bytes += cost
    return ranks, used_bytes


def fit_compact_svd_target(
    base_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    tensor_specs: Sequence[TensorSpec],
    *,
    tied_groups: Sequence[Sequence[str]] = (),
    config: CompactTargetConfig | None = None,
    policy: CompilerTargetPolicy | None = None,
    candidate_id: str = "compact-svd-target",
    manifest_metadata: Mapping[str, Any] | None = None,
) -> CompactTargetResult:
    """Fit one canonical compact-target candidate.

    The in-memory audit is an estimate. It cannot approve training supervision. Call
    ``serialize_and_audit_compiler_target`` and then run the functional Genome Gate before using
    the program as a label.
    """

    config = config or CompactTargetConfig()
    policy = policy or CompilerTargetPolicy(
        max_target_fraction_of_fp16_delta=config.target_fraction_of_fp16_delta,
        direct_tensor_numel_limit=config.direct_vector_numel_limit,
    )
    delta = compute_delta(base_state, target_state, tensor_specs)
    records, aliases = make_records(tensor_specs, tied_groups)
    specs_by_name = {spec.name: spec for spec in tensor_specs}
    unique_specs = [spec for spec in tensor_specs if spec.name not in aliases and not spec.is_buffer]
    fp16_delta_bytes = sum(spec.numel * 2 for spec in unique_specs)
    if fp16_delta_bytes < 1:
        raise ValueError("target contains no trainable tensor values")

    vector_payload_estimate = sum(
        spec.numel + 4
        for spec in unique_specs
        if len(spec.shape) != 2 and spec.numel <= config.direct_vector_numel_limit
    )
    target_budget = int(fp16_delta_bytes * config.target_fraction_of_fp16_delta)
    factor_budget = max(target_budget - vector_payload_estimate, 0)
    factors = {
        spec.name: _factorize(delta[spec.name])
        for spec in unique_specs
        if len(spec.shape) == 2
    }
    ranks, logical_factor_bytes = _allocate_ranks(
        factors,
        specs_by_name,
        byte_budget=factor_budget,
        factor_dtype=config.torch_factor_dtype,
        max_rank=config.max_rank,
    )

    payload: dict[str, torch.Tensor] = {}
    logical_vector_bytes = 0
    for record in records:
        name = record.tensor_name
        if name in aliases:
            record.components.append(
                GenomeComponent(COPY_FROM_TIED, arguments={"owner": aliases[name]})
            )
            continue
        spec = specs_by_name[name]
        if spec.is_buffer:
            continue
        source = delta[name]
        if source.ndim == 2:
            rank = ranks.get(name, 0)
            if rank < 1:
                continue
            factor = factors[name]
            prefix = f"t{record.canonical_index:05d}.low_rank"
            keys = [f"{prefix}.u", f"{prefix}.s", f"{prefix}.vh"]
            payload[keys[0]] = factor.u[:, :rank].to(config.torch_factor_dtype).contiguous()
            payload[keys[1]] = factor.s[:rank].to(config.torch_factor_dtype).contiguous()
            payload[keys[2]] = factor.vh[:rank].to(config.torch_factor_dtype).contiguous()
            record.components.append(
                GenomeComponent(
                    LOW_RANK,
                    payload_keys=keys,
                    arguments={
                        "rank": rank,
                        "canonical_sign": "max_abs_u_pivot_positive",
                        "factor_dtype": config.factor_dtype,
                    },
                )
            )
        elif spec.numel <= config.direct_vector_numel_limit:
            quantized, scale = _quantize_int8(source)
            prefix = f"t{record.canonical_index:05d}.vector_q8"
            keys = [f"{prefix}.values", f"{prefix}.scale"]
            payload[keys[0]] = quantized
            payload[keys[1]] = scale
            logical_vector_bytes += quantized.numel() + scale.numel() * scale.element_size()
            record.components.append(
                GenomeComponent(
                    QUANTIZED_DELTA,
                    payload_keys=keys,
                    arguments={"bits": 8, "shape": list(record.shape), "small_vector": True},
                )
            )

    metadata = dict(manifest_metadata or {})
    metadata.update(
        {
            # Target labels must have a stable serialized identity. make_manifest permits
            # metadata to override its wall-clock field, so compiler labels pin it to zero.
            "created_unix": 0.0,
            "compiler_target": True,
            "contains_exact_residual": False,
            "contains_dense_matrix_delta": False,
            "target_language": "canonical_low_rank_plus_small_int8_vectors",
        }
    )
    manifest = make_manifest(
        candidate_id=candidate_id,
        codec="compact_svd_target_v2",
        metadata=metadata,
    )
    manifest["codec_config"] = {
        **asdict(config),
        "allocated_ranks": ranks,
        "fp16_delta_bytes": fp16_delta_bytes,
    }
    program = GenomeProgram(manifest=manifest, records=records, payload_tensors=payload)
    audit = audit_compiler_target(
        program,
        tensor_specs,
        fp16_delta_bytes=fp16_delta_bytes,
        policy=policy,
    )
    return CompactTargetResult(
        program=program,
        audit=audit,
        allocated_ranks=ranks,
        logical_factor_bytes=logical_factor_bytes,
        logical_vector_bytes=logical_vector_bytes,
    )


def serialize_and_audit_compiler_target(
    result: CompactTargetResult,
    inventory: Sequence[TensorSpec],
    path: str | Path,
    *,
    policy: CompilerTargetPolicy | None = None,
) -> SerializedCompactTarget:
    """Serialize a target candidate and repeat the policy audit with real file bytes."""

    destination = Path(path)
    artifact_sizes = save_program(result.program, destination)
    effective_policy = policy or CompilerTargetPolicy(**result.audit.policy)
    audit = audit_compiler_target(
        result.program,
        inventory,
        fp16_delta_bytes=result.audit.fp16_delta_bytes,
        policy=effective_policy,
        actual_mgp_bytes=int(artifact_sizes["mgp_bytes"]),
    )
    write_json(destination / "compiler_target_audit.json", audit.to_dict(), canonical=True)
    return SerializedCompactTarget(
        path=destination,
        audit=audit,
        artifact_sizes=artifact_sizes,
    )


def fit_compact_svd_frontier(
    base_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    tensor_specs: Sequence[TensorSpec],
    *,
    fractions: Sequence[float] = (0.01, 0.025, 0.05, 0.10, 0.20),
    tied_groups: Sequence[Sequence[str]] = (),
    max_rank: int = 64,
) -> tuple[CompactTargetResult, ...]:
    if not fractions:
        raise ValueError("frontier requires at least one byte fraction")
    if tuple(sorted(set(float(value) for value in fractions))) != tuple(float(value) for value in fractions):
        raise ValueError("frontier fractions must be unique and increasing")
    results = []
    for fraction in fractions:
        exploratory = fraction > 0.10
        config = CompactTargetConfig(
            target_fraction_of_fp16_delta=float(fraction),
            max_rank=max_rank,
        )
        policy = CompilerTargetPolicy(
            max_target_fraction_of_fp16_delta=min(float(fraction), 0.10),
            exploratory_max_fraction_of_fp16_delta=max(float(fraction), 0.10),
            direct_tensor_numel_limit=config.direct_vector_numel_limit,
            allow_exploratory_band=exploratory,
        )
        results.append(
            fit_compact_svd_target(
                base_state,
                target_state,
                tensor_specs,
                tied_groups=tied_groups,
                config=config,
                policy=policy,
                candidate_id=f"compact-svd-{fraction:.4f}",
            )
        )
    return tuple(results)
