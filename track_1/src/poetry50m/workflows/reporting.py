"""Held-out loss, blind-judgment, and cost reporting commands."""

from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol, cast

import torch

from poetry50m.data import PreparedBatchStream
from poetry50m.data.artifacts import read_packed_sequences
from poetry50m.evaluation import BlindJudgment, CostRecord, aggregate_blind_judgments
from poetry50m.evaluation.schema import BlindComparison, BlindComparisonPack
from poetry50m.training import Trainer
from poetry50m.trajectory.ledgers import CostLedger, CpuCost

JsonWriter = Callable[[Path, object], None]
_COST_COMPONENT_NAMES = frozenset({"analysis", "checkpoint_io", "verification_per_replay"})
_COST_COMPONENT_FIELDS = frozenset(
    {
        "steps",
        "tokens",
        "wall_seconds",
        "process_cpu_seconds",
        "accelerator_seconds",
        "device_active_wall_seconds",
        "timing_scope",
    }
)


@dataclass(frozen=True, slots=True)
class _ReceiptCostComponent:
    steps: int
    tokens: int
    wall_seconds: float
    process_cpu_seconds: float
    accelerator_seconds: float | None
    device_active_wall_seconds: float | None


class TrainerFactory(Protocol):
    def __call__(
        self,
        args: argparse.Namespace,
        *,
        resume: Path | None = None,
        read_only: bool = False,
    ) -> tuple[Trainer, PreparedBatchStream]: ...


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def eval_loss_command(
    args: argparse.Namespace, *, trainer_factory: TrainerFactory, write_json: JsonWriter
) -> int:
    command_started = time.perf_counter()
    process_started = time.process_time()
    trainer, _ = trainer_factory(args, resume=Path(args.checkpoint), read_only=True)
    packs = read_packed_sequences(Path(args.prepared) / f"{args.split}.packed.jsonl")
    if not packs:
        raise ValueError(f"prepared artifact has no {args.split} packs")
    _synchronize(trainer.device)
    accelerator_start = (
        torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        if trainer.device.type == "cuda"
        else None
    )
    accelerator_end = (
        torch.cuda.Event(enable_timing=True)  # type: ignore[no-untyped-call]
        if trainer.device.type == "cuda"
        else None
    )
    if accelerator_start is not None:
        accelerator_start.record()
    started = time.perf_counter()
    weighted_loss = 0.0
    token_count = 0
    trainer.model.eval()
    with torch.inference_mode():
        for pack in packs:
            inputs = torch.tensor([pack.input_ids[:-1]], dtype=torch.long, device=trainer.device)
            targets = torch.tensor([pack.input_ids[1:]], dtype=torch.long, device=trainer.device)
            mask = torch.tensor([pack.loss_mask[1:]], dtype=torch.bool, device=trainer.device)
            model_output = trainer.model(inputs, targets=targets, loss_mask=mask)
            if model_output.loss is None or model_output.token_count < 1:
                raise ValueError(f"{args.split} pack {pack.pack_id} has no supervised targets")
            weighted_loss += float(model_output.loss.item()) * model_output.token_count
            token_count += model_output.token_count
    if accelerator_end is not None:
        accelerator_end.record()
    _synchronize(trainer.device)
    evaluation_wall_seconds = time.perf_counter() - started
    command_wall_seconds = time.perf_counter() - command_started
    process_cpu_seconds = time.process_time() - process_started
    accelerator_seconds = (
        accelerator_start.elapsed_time(accelerator_end) / 1000.0
        if accelerator_start is not None and accelerator_end is not None
        else None
    )
    write_json(
        Path(args.output),
        {
            "format_version": 1,
            "split": args.split,
            "mean_ntp_loss": weighted_loss / token_count,
            "token_count": token_count,
            "steps": 0,
            "tokens": token_count,
            "step": trainer.state.global_step,
            "run_id": trainer.trajectory_metadata.run_id
            if trainer.trajectory_metadata is not None
            else trainer.run_identity,
            "checkpoint_hash": _file_hash(Path(args.checkpoint)),
            "checkpoint_sha256": _file_hash(Path(args.checkpoint)),
            "wall_seconds": command_wall_seconds,
            "evaluation_wall_seconds": evaluation_wall_seconds,
            "process_cpu_seconds": process_cpu_seconds,
            "accelerator_seconds": accelerator_seconds,
            "device_active_wall_seconds": (
                evaluation_wall_seconds if trainer.device.type in {"cuda", "mps"} else None
            ),
            "device": trainer.device.type,
            "use": "reporting_only" if args.split == "test" else "validation_reporting",
        },
    )
    return 0


def _file_hash(path: Path) -> str:
    from poetry50m.config import file_hash

    return file_hash(path)


def _read_jsonl_objects(path: Path) -> tuple[dict[str, object], ...]:
    values: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
                raise ValueError(f"JSONL record must be an object at {path}:{line_number}")
            values.append(value)
    return tuple(values)


def blind_aggregate_command(
    args: argparse.Namespace,
    *,
    load_mapping: Callable[[Path], dict[str, object]],
    write_json: JsonWriter,
) -> int:
    comparisons: list[BlindComparison] = []
    comparison_keys = {
        "comparison_id",
        "request_id",
        "case_id",
        "seed",
        "left_label",
        "right_label",
        "left_text",
        "right_text",
    }
    for value in _read_jsonl_objects(Path(args.blind_pack)):
        if set(value) != comparison_keys:
            raise ValueError("blind comparison row has unknown or missing fields")
        strings = tuple(
            value[name]
            for name in (
                "comparison_id",
                "request_id",
                "case_id",
                "left_label",
                "right_label",
                "left_text",
                "right_text",
            )
        )
        seed = value["seed"]
        if (
            any(not isinstance(item, str) for item in strings)
            or isinstance(seed, bool)
            or not isinstance(seed, int)
        ):
            raise TypeError("blind comparison fields have invalid types")
        comparisons.append(
            BlindComparison(
                cast(str, strings[0]),
                cast(str, strings[1]),
                cast(str, strings[2]),
                seed,
                cast(str, strings[3]),
                cast(str, strings[4]),
                cast(str, strings[5]),
                cast(str, strings[6]),
            )
        )
    comparison_ids = [comparison.comparison_id for comparison in comparisons]
    if len(set(comparison_ids)) != len(comparison_ids):
        raise ValueError("blind pack comparison IDs must be unique")
    if len({comparison.request_id for comparison in comparisons}) != len(comparisons):
        raise ValueError("blind pack request IDs must be unique")
    raw_key = load_mapping(Path(args.key))
    key: dict[str, dict[str, str]] = {}
    for comparison_id, labels in raw_key.items():
        if (
            not isinstance(labels, dict)
            or set(labels) != {"A", "B"}
            or not all(isinstance(value, str) and value for value in labels.values())
        ):
            raise ValueError("blind key entries must map A and B to candidate identities")
        key[comparison_id] = {"A": labels["A"], "B": labels["B"]}
    if set(key) != set(comparison_ids):
        raise ValueError("unblinding key must exactly cover the blind comparison pack")
    expected_candidates = {args.candidate_a, args.candidate_b}
    if any(set(labels.values()) != expected_candidates for labels in key.values()):
        raise ValueError("unblinding key candidate identities do not match the command")
    judgments: list[BlindJudgment] = []
    judgment_keys = {
        "comparison_id",
        "prompt_relevance",
        "poetic_quality",
        "image_music",
        "degeneration",
        "notes",
    }
    for value in _read_jsonl_objects(Path(args.judgments)):
        if set(value) != judgment_keys or any(
            not isinstance(value[name], str) for name in judgment_keys
        ):
            raise ValueError("blind judgment row has unknown, missing, or non-string fields")
        choices = tuple(
            value[name]
            for name in ("prompt_relevance", "poetic_quality", "image_music", "degeneration")
        )
        if any(choice not in {"A", "B", "tie"} for choice in choices):
            raise ValueError("blind judgment choices must be A, B, or tie")
        judgments.append(
            BlindJudgment(
                cast(str, value["comparison_id"]),
                cast(Literal["A", "B", "tie"], choices[0]),
                cast(Literal["A", "B", "tie"], choices[1]),
                cast(Literal["A", "B", "tie"], choices[2]),
                cast(Literal["A", "B", "tie"], choices[3]),
                cast(str, value["notes"]),
            )
        )
    if len({judgment.comparison_id for judgment in judgments}) != len(judgments):
        raise ValueError("blind judgments must contain unique comparison IDs")
    pack = BlindComparisonPack(
        tuple(comparisons),
        key,
        candidate_a_id=args.candidate_a,
        candidate_b_id=args.candidate_b,
    )
    tallies = aggregate_blind_judgments(pack, tuple(judgments))
    write_json(
        Path(args.output),
        {
            "comparison_count": len(comparisons),
            "candidate_a_id": args.candidate_a,
            "candidate_b_id": args.candidate_b,
            "tallies": [asdict(tally) for tally in tallies],
        },
    )
    return 0


def _string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: object, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_integer(value: object, *, name: str) -> int | None:
    return None if value is None else _integer(value, name=name)


def _number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    result = float(value)
    if not torch.isfinite(torch.tensor(result)).item() or result < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return result


def _optional_number(value: object, *, name: str) -> float | None:
    return None if value is None else _number(value, name=name)


def _optional_components_total(
    components: tuple[_ReceiptCostComponent, ...],
    field: Literal["accelerator_seconds", "device_active_wall_seconds"],
) -> float | None:
    values = tuple(getattr(component, field) for component in components)
    return (
        None
        if any(value is None for value in values)
        else sum(value for value in values if value is not None)
    )


def _cost_components(receipt: dict[str, object]) -> dict[str, _ReceiptCostComponent]:
    raw_components = receipt.get("cost_components")
    if not isinstance(raw_components, dict) or set(raw_components) != _COST_COMPONENT_NAMES:
        raise ValueError(
            f"analysis receipt cost_components must cover {sorted(_COST_COMPONENT_NAMES)} exactly"
        )
    components: dict[str, _ReceiptCostComponent] = {}
    for name in sorted(_COST_COMPONENT_NAMES):
        raw_component = raw_components[name]
        if not isinstance(raw_component, dict) or set(raw_component) != _COST_COMPONENT_FIELDS:
            raise ValueError(
                f"{name} cost component must contain exactly {sorted(_COST_COMPONENT_FIELDS)}"
            )
        _string(raw_component["timing_scope"], name=f"{name}.timing_scope")
        components[name] = _ReceiptCostComponent(
            _integer(raw_component["steps"], name=f"{name}.steps"),
            _integer(raw_component["tokens"], name=f"{name}.tokens"),
            _number(raw_component["wall_seconds"], name=f"{name}.wall_seconds"),
            _number(
                raw_component["process_cpu_seconds"],
                name=f"{name}.process_cpu_seconds",
            ),
            _optional_number(
                raw_component["accelerator_seconds"],
                name=f"{name}.accelerator_seconds",
            ),
            _optional_number(
                raw_component["device_active_wall_seconds"],
                name=f"{name}.device_active_wall_seconds",
            ),
        )
    ordered = tuple(components[name] for name in sorted(_COST_COMPONENT_NAMES))
    expected_steps = _integer(receipt.get("steps"), name="analysis.steps")
    expected_tokens = _integer(receipt.get("tokens"), name="analysis.tokens")
    expected_wall = _number(receipt.get("wall_seconds"), name="analysis.wall_seconds")
    expected_cpu = _number(
        receipt.get("process_cpu_seconds"),
        name="analysis.process_cpu_seconds",
    )
    expected_accelerator = _optional_number(
        receipt.get("accelerator_seconds"),
        name="analysis.accelerator_seconds",
    )
    expected_device_active = _optional_number(
        receipt.get("device_active_wall_seconds"),
        name="analysis.device_active_wall_seconds",
    )
    component_accelerator = _optional_components_total(ordered, "accelerator_seconds")
    component_device_active = _optional_components_total(ordered, "device_active_wall_seconds")
    if (
        sum(component.steps for component in ordered) != expected_steps
        or sum(component.tokens for component in ordered) != expected_tokens
        or not math.isclose(
            sum(component.wall_seconds for component in ordered),
            expected_wall,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or not math.isclose(
            sum(component.process_cpu_seconds for component in ordered),
            expected_cpu,
            rel_tol=1e-9,
            abs_tol=1e-9,
        )
        or component_accelerator != expected_accelerator
        or component_device_active != expected_device_active
    ):
        raise ValueError("analysis cost components do not reconcile to the full receipt")
    return components


def _receipt_reference(
    value: object,
    *,
    name: str,
    base_directory: Path,
    load_mapping: Callable[[Path], dict[str, object]],
) -> tuple[Path, str, float | None, dict[str, object]]:
    if not isinstance(value, dict) or set(value) != {
        "receipt",
        "sha256",
        "estimated_cost_usd",
    }:
        raise ValueError(f"{name} must contain receipt, sha256, and estimated_cost_usd")
    raw_path = _string(value["receipt"], name=f"{name}.receipt")
    expected_hash = _string(value["sha256"], name=f"{name}.sha256")
    path = Path(raw_path)
    if not path.is_absolute():
        path = base_directory / path
    if _file_hash(path) != expected_hash:
        raise ValueError(f"{name} receipt does not match its immutable SHA-256")
    raw_usd = value["estimated_cost_usd"]
    usd = None if raw_usd is None else _number(raw_usd, name=f"{name}.estimated_cost_usd")
    receipt = load_mapping(path)
    version = receipt.get("format_version")
    if isinstance(version, bool) or version != 1:
        raise ValueError(f"{name} receipt must use format_version 1")
    return path, expected_hash, usd, receipt


def _receipt_cost_record(
    receipt: dict[str, object],
    *,
    name: str,
    estimated_cost_usd: float | None,
) -> tuple[CostRecord, CpuCost]:
    run_id = _string(receipt.get("run_id"), name=f"{name}.run_id")
    checkpoint_id = _string(
        receipt.get("checkpoint_sha256", receipt.get("checkpoint_id")),
        name=f"{name}.checkpoint_id",
    )
    if name in _COST_COMPONENT_NAMES:
        component = _cost_components(receipt)[name]
        steps = component.steps
        tokens = component.tokens
        wall_seconds = component.wall_seconds
        process_cpu_seconds = component.process_cpu_seconds
        accelerator_seconds = component.accelerator_seconds
        device_active_wall_seconds = component.device_active_wall_seconds
    elif "optimizer_steps_executed_this_command" in receipt:
        steps = _integer(
            receipt["optimizer_steps_executed_this_command"],
            name=f"{name}.optimizer_steps_executed_this_command",
        )
        tokens = _integer(
            receipt.get("data_tokens_processed_this_command"),
            name=f"{name}.data_tokens_processed_this_command",
        )
        wall_seconds = _number(
            receipt.get("command_wall_seconds"),
            name=f"{name}.command_wall_seconds",
        )
        process_cpu_seconds = _number(
            receipt.get("process_cpu_seconds"),
            name=f"{name}.process_cpu_seconds",
        )
        accelerator_seconds = _optional_number(
            receipt.get("accelerator_seconds"),
            name=f"{name}.accelerator_seconds",
        )
        device_active_wall_seconds = _optional_number(
            receipt.get("device_active_wall_seconds"),
            name=f"{name}.device_active_wall_seconds",
        )
    else:
        steps = _integer(receipt.get("steps"), name=f"{name}.steps")
        tokens = _integer(receipt.get("tokens"), name=f"{name}.tokens")
        wall_seconds = _number(
            receipt.get("wall_seconds"),
            name=f"{name}.wall_seconds",
        )
        process_cpu_seconds = _number(
            receipt.get("process_cpu_seconds"),
            name=f"{name}.process_cpu_seconds",
        )
        accelerator_seconds = _optional_number(
            receipt.get("accelerator_seconds"),
            name=f"{name}.accelerator_seconds",
        )
        device_active_wall_seconds = _optional_number(
            receipt.get("device_active_wall_seconds"),
            name=f"{name}.device_active_wall_seconds",
        )
    _string(receipt.get("device"), name=f"{name}.device")
    return (
        CostRecord(
            run_id,
            checkpoint_id,
            steps,
            tokens,
            wall_seconds,
            accelerator_seconds,
            estimated_cost_usd,
            device_active_wall_seconds,
        ),
        CpuCost(process_cpu_seconds),
    )


def cost_report_command(
    args: argparse.Namespace,
    *,
    load_mapping: Callable[[Path], dict[str, object]],
    write_json: JsonWriter,
) -> int:
    input_path = Path(args.input)
    value = load_mapping(input_path)
    expected = {"format_version", "records", "resource_receipt", "amortized_uses"}
    if (
        set(value) != expected
        or isinstance(value["format_version"], bool)
        or not isinstance(value["format_version"], int)
        or value["format_version"] != 1
    ):
        raise ValueError(f"cost assembly must contain exactly {sorted(expected)}")
    record_names = (
        "reference",
        "analysis",
        "checkpoint_io",
        "verification_per_replay",
        "replay",
        "baseline_replay",
    )
    raw_records = value["records"]
    if not isinstance(raw_records, dict) or set(raw_records) != set(record_names):
        raise ValueError("cost assembly records must cover every CostLedger role exactly")
    base_directory = input_path.parent
    records: dict[str, CostRecord] = {}
    cpus: dict[str, CpuCost] = {}
    evidence: dict[str, dict[str, str]] = {}
    for name in record_names:
        path, receipt_hash, usd, receipt = _receipt_reference(
            raw_records[name],
            name=name,
            base_directory=base_directory,
            load_mapping=load_mapping,
        )
        records[name], cpus[name] = _receipt_cost_record(
            receipt,
            name=name,
            estimated_cost_usd=usd,
        )
        evidence[name] = {"receipt": str(path.resolve()), "sha256": receipt_hash}
    resource_path, resource_hash, _usd, resources = _receipt_reference(
        value["resource_receipt"],
        name="resource_receipt",
        base_directory=base_directory,
        load_mapping=load_mapping,
    )
    peak = _optional_integer(
        resources.get("actual_peak_working_memory_bytes"),
        name="resource_receipt.actual_peak_working_memory_bytes",
    )
    current_memory = _optional_integer(
        resources.get("current_working_memory_bytes"),
        name="resource_receipt.current_working_memory_bytes",
    )
    peak_memory_semantics = _string(
        resources.get("peak_memory_semantics"),
        name="resource_receipt.peak_memory_semantics",
    )
    checkpoint_io_wall = _number(
        resources.get("checkpoint_io_wall_seconds"),
        name="resource_receipt.checkpoint_io_wall_seconds",
    )
    bytes_read = _integer(
        resources.get("snapshot_bytes_read"),
        name="resource_receipt.snapshot_bytes_read",
    )
    bytes_written = _integer(
        resources.get("snapshot_bytes_written"),
        name="resource_receipt.snapshot_bytes_written",
    )
    uses = value["amortized_uses"]
    if (
        not isinstance(uses, list)
        or not uses
        or any(isinstance(use, bool) or not isinstance(use, int) or use < 1 for use in uses)
    ):
        raise ValueError("amortized_uses must be a non-empty positive integer list")
    ledger = CostLedger(
        records["reference"],
        records["analysis"],
        records["checkpoint_io"],
        records["verification_per_replay"],
        records["replay"],
        records["baseline_replay"],
        peak,
        current_memory,
        peak_memory_semantics,
        checkpoint_io_wall,
        bytes_read,
        bytes_written,
        cpus["reference"],
        cpus["analysis"],
        cpus["checkpoint_io"],
        cpus["verification_per_replay"],
        cpus["replay"],
        cpus["baseline_replay"],
    )
    report = ledger.to_mapping()
    report["amortized"] = {str(use): ledger.amortized(use).to_mapping() for use in uses}
    report["evidence"] = {
        "assembly_sha256": _file_hash(input_path),
        "records": evidence,
        "resource_receipt": {
            "receipt": str(resource_path.resolve()),
            "sha256": resource_hash,
        },
    }
    write_json(Path(args.output), report)
    return 0
