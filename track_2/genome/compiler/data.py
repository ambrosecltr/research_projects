from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file

from ..architecture import ArchitectureGraph
from ..fingerprint import FingerprintBundle
from ..hashing import sha256_file, stable_u64
from ..io import load_json
from ..mgp.runtime import execute_program
from ..mgp.schema import ModelGenomeProgram
from ..mgp.serialize import load_program
from ..protocol import ArtifactBinding, TargetFormula, require_matching_bindings
from ..sources import SourcePlan
from ..state import load_state, state_id
from .model import CompilerExample, TensorEvidence

NON_SEMANTIC_RECIPE_KEYS = frozenset(
    {
        "commit",
        "licence",
        "license",
        "order_files",
        "order_id",
        "order_repository",
        "order_revision",
        "path",
        "provenance",
        "repository",
        "revision",
        "run_id",
        "sha256",
        "source_plan_id",
        "uri",
    }
)


def _count_sketch(values: torch.Tensor, dim: int, *, seed: int) -> torch.Tensor:
    flat = values.detach().float().reshape(-1).cpu()
    result = torch.zeros(dim, dtype=torch.float32)
    for offset, value in enumerate(flat.tolist()):
        hashed = stable_u64(f"{seed}:{offset}")
        index = int(hashed % dim)
        sign = 1.0 if (hashed >> 63) == 0 else -1.0
        result[index] += sign * float(value)
    return result / max(1.0, math.sqrt(flat.numel()))


def _fixed_vector(parts: Sequence[torch.Tensor], dim: int, *, seed: int) -> torch.Tensor:
    if not parts:
        return torch.zeros(dim, dtype=torch.float32)
    flat = torch.cat([item.detach().float().reshape(-1).cpu() for item in parts])
    return _count_sketch(flat, dim, seed=seed)


def recipe_vector(recipe: Mapping[str, Any], dim: int, *, seed: int = 3107) -> torch.Tensor:
    """Encode numeric recipe values and categorical strings without treating hashes as semantics.

    Numeric values are inserted directly. Categorical values select deterministic feature slots, but
    the digest itself is never exposed as a model input. This is conventional feature hashing, not
    a cryptographic dataset fingerprint.
    """

    output = torch.zeros(dim, dtype=torch.float32)

    def visit(prefix: str, value: Any) -> None:
        if isinstance(value, bool):
            index = stable_u64(f"{seed}:{prefix}:bool") % dim
            output[index] += 1.0 if value else -1.0
        elif isinstance(value, (int, float)) and math.isfinite(float(value)):
            index = stable_u64(f"{seed}:{prefix}:number") % dim
            output[index] += math.copysign(math.log1p(abs(float(value))), float(value))
        elif isinstance(value, str):
            index = stable_u64(f"{seed}:{prefix}:{value}") % dim
            output[index] += 1.0
        elif isinstance(value, Mapping):
            for key in sorted(value):
                if str(key).lower() in NON_SEMANTIC_RECIPE_KEYS:
                    continue
                visit(f"{prefix}.{key}", value[key])
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
            for index, item in enumerate(value):
                visit(f"{prefix}[{index}]", item)

    visit("recipe", recipe)
    norm = output.norm()
    return output if norm == 0 else output / norm


def _tensor_features(
    tensor: torch.Tensor,
    *,
    layer: int | None,
    role: str,
    feature_dim: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
    values = tensor.detach().float().cpu()
    shape = tuple(values.shape)
    base = [
        math.log1p(values.numel()),
        float(values.ndim),
        math.log1p(shape[0]) if shape else 0.0,
        math.log1p(shape[1]) if len(shape) > 1 else 0.0,
        float(values.mean()),
        float(values.std(unbiased=False)),
        float(values.abs().mean()),
        math.log1p(float(values.norm())),
        -1.0 if layer is None else math.log1p(layer),
        float(shape[0] / max(1, shape[1])) if len(shape) > 1 else 0.0,
    ]
    features = torch.tensor(base, dtype=torch.float32)
    if features.numel() < feature_dim:
        features = torch.nn.functional.pad(features, (0, feature_dim - features.numel()))
    else:
        features = features[:feature_dim]

    row_features: torch.Tensor | None = None
    col_features: torch.Tensor | None = None
    if values.ndim == 2:
        row_features = torch.stack(
            [
                values.mean(dim=1),
                values.std(dim=1, unbiased=False),
                values.norm(dim=1) / max(1.0, math.sqrt(values.shape[1])),
            ],
            dim=1,
        )
        col_features = torch.stack(
            [
                values.mean(dim=0),
                values.std(dim=0, unbiased=False),
                values.norm(dim=0) / max(1.0, math.sqrt(values.shape[0])),
            ],
            dim=1,
        )
    elif values.ndim == 1:
        row_features = torch.stack(
            [
                values,
                values.abs(),
                torch.full_like(values, float(values.std(unbiased=False))),
            ],
            dim=1,
        )
    return features, row_features, col_features


def build_compiler_example(
    graph: ArchitectureGraph,
    w0: Mapping[str, torch.Tensor],
    fingerprint: FingerprintBundle,
    recipe: Mapping[str, Any],
    *,
    global_feature_dim: int,
    tensor_feature_dim: int,
    base_state_id: str,
) -> CompilerExample:
    if {node.name for node in graph.tensors} != set(w0):
        raise ValueError("architecture graph and W0 state have different tensor names")
    fingerprint_vector = _fixed_vector(
        [fingerprint.tensors[name] for name in sorted(fingerprint.tensors)],
        global_feature_dim,
        seed=1977,
    )
    recipe_features = recipe_vector(recipe, global_feature_dim, seed=3107)
    global_features = torch.nn.functional.normalize(fingerprint_vector + recipe_features, dim=0)
    tensors: list[TensorEvidence] = []
    for node in graph.tensors:
        features, rows, cols = _tensor_features(
            w0[node.name],
            layer=node.layer,
            role=node.role,
            feature_dim=tensor_feature_dim,
        )
        tensors.append(
            TensorEvidence(
                name=node.name,
                role=node.role,
                shape=node.shape,
                tied_to=node.tied_to,
                features=features,
                row_features=rows,
                col_features=cols,
            )
        )
    return CompilerExample(
        architecture=graph,
        global_features=global_features,
        tensors=tensors,
        base_state_id=base_state_id,
    )


@dataclass(frozen=True)
class CompilerRecord:
    run_id: str
    split: str
    graph_path: str
    w0_path: str
    wt_path: str
    fingerprint_path: str
    recipe_path: str
    program_path: str
    evaluation_report_path: str
    evaluation_jsonl: str
    model_config_path: str | None = None
    probe_jsonl: str | None = None

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> CompilerRecord:
        required = {
            "run_id",
            "split",
            "graph_path",
            "w0_path",
            "wt_path",
            "fingerprint_path",
            "recipe_path",
            "program_path",
            "evaluation_report_path",
            "evaluation_jsonl",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(f"compiler record is missing {sorted(missing)}")
        data = {key: (None if item is None else str(item)) for key, item in value.items()}
        return cls(**data)


@dataclass(frozen=True)
class CompilerCorpus:
    records: tuple[CompilerRecord, ...]
    formula_id: str
    source_plan_id: str
    format: str = "GENOME_COMPILER_CORPUS"
    version: str = "2.0.0"

    def __post_init__(self) -> None:
        ids = [item.run_id for item in self.records]
        if len(ids) != len(set(ids)):
            raise ValueError("compiler corpus run IDs must be unique")
        if any(item.split not in {"training", "development"} for item in self.records):
            raise ValueError("compiler corpus may contain only training and development lives")
        if not any(item.split == "training" for item in self.records):
            raise ValueError("compiler corpus requires training lives")
        if not any(item.split == "development" for item in self.records):
            raise ValueError("compiler corpus requires development lives")
        for item in self.records:
            verify_compiler_record(
                item,
                formula_id=self.formula_id,
                source_plan_id=self.source_plan_id,
            )

    @classmethod
    def load(cls, path: str | Path) -> CompilerCorpus:
        value = load_json(path)
        return cls(
            records=tuple(CompilerRecord.from_dict(item) for item in value["records"]),
            formula_id=str(value["formula_id"]),
            source_plan_id=str(value["source_plan_id"]),
            format=str(value.get("format", "GENOME_COMPILER_CORPUS")),
            version=str(value.get("version", "2.0.0")),
        )

    def to_dict(self) -> dict[str, Any]:
        training = sum(item.split == "training" for item in self.records)
        development = sum(item.split == "development" for item in self.records)
        return {
            "format": self.format,
            "version": self.version,
            "expected_records": {
                "training": training,
                "development": development,
                "total": len(self.records),
            },
            "formula_id": self.formula_id,
            "source_plan_id": self.source_plan_id,
            "records": [asdict(item) for item in self.records],
        }


def verify_compiler_record(
    record: CompilerRecord,
    *,
    formula_id: str,
    source_plan_id: str,
) -> ArtifactBinding:
    program_root = Path(record.program_path)
    acceptance_path = program_root / "acceptance.json"
    if not acceptance_path.is_file():
        raise ValueError(f"compiler target for {record.run_id} has no acceptance.json")
    acceptance = load_json(acceptance_path)
    if acceptance.get("accepted") is not True:
        raise ValueError(f"compiler target for {record.run_id} did not pass the Genome Gate")
    evaluation_path = Path(record.evaluation_report_path)
    evaluation = load_json(evaluation_path)
    if evaluation.get("format") != "GENOME_TARGET_EVALUATION":
        raise ValueError(f"compiler target for {record.run_id} has an invalid evaluation report")
    if acceptance.get("format") != "GENOME_TARGET_ACCEPTANCE":
        raise ValueError(f"compiler target for {record.run_id} has an invalid acceptance report")
    evaluation_binding = ArtifactBinding.from_dict(evaluation["binding"])
    acceptance_binding = ArtifactBinding.from_dict(acceptance["binding"])
    require_matching_bindings(
        evaluation_binding,
        acceptance_binding,
        context=f"compiler target {record.run_id}",
    )
    binding = acceptance_binding
    if binding.run_id != record.run_id:
        raise ValueError(f"compiler target binding has the wrong run_id for {record.run_id}")
    if binding.formula_id != formula_id:
        raise ValueError(f"compiler target {record.run_id} has the wrong formula_id")
    if binding.source_plan_id != source_plan_id:
        raise ValueError(f"compiler target {record.run_id} has the wrong source-plan ID")
    _, _, manifest = load_program(program_root)
    checks = {
        "program_id": str(manifest["program_id"]),
        "program_manifest_sha256": sha256_file(program_root / "manifest.json"),
        "payload_sha256": sha256_file(program_root / "payload.safetensors"),
        "w0_state_id": state_id(load_state(record.w0_path)),
        "wt_state_id": state_id(load_state(record.wt_path)),
        "evaluation_jsonl_sha256": sha256_file(record.evaluation_jsonl),
    }
    for name, actual in checks.items():
        if getattr(binding, name) != actual:
            raise ValueError(f"compiler target {record.run_id} has a stale {name} binding")
    if sha256_file(evaluation_path) != acceptance["evaluation_report_sha256"]:
        raise ValueError(f"compiler target {record.run_id} acceptance points to another evaluation")
    return binding


def build_compiler_corpus(
    plan: SourcePlan,
    *,
    workspace: str | Path,
    program_root: str | Path,
    formula_path: str | Path,
) -> CompilerCorpus:
    root = Path(workspace)
    accepted = Path(program_root)
    formula = TargetFormula.load(formula_path)
    if formula.status != "frozen":
        raise ValueError("compiler corpus construction requires the global formula to be frozen")
    probe = Path(formula.data["refinement"])
    rejected = {str(item) for item in formula.corpus["rejected_training_lives"]}
    records: list[CompilerRecord] = []
    for life in plan.lives:
        if life.split not in {"training", "development"}:
            continue
        program_path = accepted / life.run_id
        acceptance_path = program_path / "acceptance.json"
        if not acceptance_path.is_file():
            raise ValueError(f"target report is missing for {life.run_id}")
        acceptance = load_json(acceptance_path)
        should_be_rejected = life.run_id in rejected
        if should_be_rejected:
            if life.split != "training" or acceptance.get("accepted") is not False:
                raise ValueError(f"declared rejected training life {life.run_id} is not rejected")
            binding = ArtifactBinding.from_dict(acceptance["binding"])
            if binding.formula_id != formula.formula_id or binding.source_plan_id != plan.plan_id:
                raise ValueError(f"rejection report for {life.run_id} has stale bindings")
            continue
        if acceptance.get("accepted") is not True:
            raise ValueError(f"required compiler target {life.run_id} was not accepted")
        records.append(
            CompilerRecord(
                run_id=life.run_id,
                split=life.split,
                graph_path=str(
                    root / "canonical" / "lives" / life.run_id / "architecture.json"
                ),
                w0_path=str(root / "canonical" / "lives" / life.run_id / "w0.safetensors"),
                wt_path=str(root / "canonical" / "lives" / life.run_id / "wt.safetensors"),
                fingerprint_path=str(root / "evidence" / life.run_id),
                recipe_path=str(root / "canonical" / "lives" / life.run_id / "recipe.json"),
                program_path=str(program_path),
                evaluation_report_path=str(program_path / "evaluation.json"),
                evaluation_jsonl=str(
                    formula.data[
                        "development_verifier"
                        if life.split == "development"
                        else "formula_tuning"
                    ]
                ),
                model_config_path=str(
                    root / "canonical" / "lives" / life.run_id / "model_config.json"
                ),
                probe_jsonl=str(probe),
            )
        )
    training_count = sum(record.split == "training" for record in records)
    development_count = sum(record.split == "development" for record in records)
    expected_training = int(formula.corpus["expected_training_records"])
    expected_development = int(formula.corpus["expected_development_records"])
    if (training_count, development_count) != (expected_training, expected_development):
        raise ValueError(
            "compiler corpus count differs from the declared protocol: "
            f"{training_count} training, {development_count} development"
        )
    return CompilerCorpus(
        records=tuple(records),
        formula_id=formula.formula_id,
        source_plan_id=plan.plan_id,
    )


def load_record(
    record: CompilerRecord,
    *,
    global_feature_dim: int,
    tensor_feature_dim: int,
) -> tuple[
    CompilerExample,
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    ModelGenomeProgram,
    dict[str, torch.Tensor],
]:
    graph = ArchitectureGraph.from_dict(load_json(record.graph_path))
    w0 = dict(load_file(record.w0_path, device="cpu"))
    fingerprint = FingerprintBundle.load(record.fingerprint_path)
    recipe = load_json(record.recipe_path)
    program, payloads, _ = load_program(record.program_path)
    base_state_id = str(program.base_state_id)
    example = build_compiler_example(
        graph,
        w0,
        fingerprint,
        recipe,
        global_feature_dim=global_feature_dim,
        tensor_feature_dim=tensor_feature_dim,
        base_state_id=base_state_id,
    )
    target_state = execute_program(w0, program, payloads)
    target_deltas = {name: target_state[name].float() - w0[name].float() for name in w0}
    return example, w0, target_deltas, program, payloads
