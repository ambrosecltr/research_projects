"""JSON ledgers for analysis, candidate decisions, and honest cost accounting."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO, Protocol

from poetry50m.evaluation.schema import CostRecord
from poetry50m.trajectory._persistence import atomic_write
from poetry50m.trajectory.forecast import ForecastResult
from poetry50m.trajectory.gates import CandidateDecision


class JsonLedger(Protocol):
    def to_mapping(self) -> dict[str, object]: ...


def write_json_ledger(path: Path, ledger: JsonLedger) -> None:
    """Write one deterministic JSON ledger rather than an unstructured log line."""

    payload = json.dumps(ledger.to_mapping(), indent=2, sort_keys=True).encode("utf-8") + b"\n"

    def write(handle: BinaryIO) -> None:
        handle.write(payload)

    atomic_write(path, write)


def _is_non_negative_integer(value: object, *, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    return not isinstance(value, bool) and isinstance(value, int) and value >= 0


def _is_non_negative_number(value: object, *, allow_none: bool = False) -> bool:
    if value is None:
        return allow_none
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(value)
        and value >= 0.0
    )


@dataclass(frozen=True, slots=True)
class AnalysisLedger:
    run_id: str
    forecast: ForecastResult
    analysis_accelerator_seconds: float | None
    checkpoint_io_seconds: float
    anchor_verification_seconds: float
    actual_peak_working_memory_bytes: int | None
    snapshot_bytes_read: int
    snapshot_bytes_written: int

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id must be non-empty")
        times = (
            self.analysis_accelerator_seconds,
            self.checkpoint_io_seconds,
            self.anchor_verification_seconds,
        )
        if not all(value is None or math.isfinite(value) and value >= 0.0 for value in times):
            raise ValueError("analysis times must be finite, non-negative, or unknown")
        if not _is_non_negative_integer(
            self.actual_peak_working_memory_bytes, allow_none=True
        ) or not all(
            _is_non_negative_integer(value)
            for value in (self.snapshot_bytes_read, self.snapshot_bytes_written)
        ):
            raise ValueError("resource measurements must be non-negative")

    def to_mapping(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "forecast": {
                "method": self.forecast.method,
                "source_checkpoint_ids": list(self.forecast.source_checkpoint_ids),
                "source_steps": list(self.forecast.source_steps),
                "target_step": self.forecast.target_step,
                "diagnostics": self.forecast.diagnostics_mapping(),
            },
            "analysis_accelerator_seconds": self.analysis_accelerator_seconds,
            "checkpoint_io_seconds": self.checkpoint_io_seconds,
            "anchor_verification_seconds": self.anchor_verification_seconds,
            "actual_peak_working_memory_bytes": self.actual_peak_working_memory_bytes,
            "snapshot_bytes_read": self.snapshot_bytes_read,
            "snapshot_bytes_written": self.snapshot_bytes_written,
        }


@dataclass(frozen=True, slots=True)
class DecisionLedger:
    run_id: str
    checkpoint_id: str
    decision: CandidateDecision

    def __post_init__(self) -> None:
        if not self.run_id or not self.checkpoint_id:
            raise ValueError("run_id and checkpoint_id must be non-empty")

    def to_mapping(self) -> dict[str, object]:
        return {
            "run_id": self.run_id,
            "checkpoint_id": self.checkpoint_id,
            "decision": self.decision.to_mapping(),
        }


@dataclass(frozen=True, slots=True)
class CpuCost:
    seconds: float

    def __post_init__(self) -> None:
        if not _is_non_negative_number(self.seconds):
            raise ValueError("CPU seconds must be finite and non-negative")


@dataclass(frozen=True, slots=True)
class CostTotals:
    accelerator_seconds: float | None
    wall_seconds: float
    cpu_seconds: float
    estimated_cost_usd: float | None
    device_active_wall_seconds: float | None = None

    def __post_init__(self) -> None:
        if (
            not _is_non_negative_number(self.wall_seconds)
            or not _is_non_negative_number(self.cpu_seconds)
            or not all(
                _is_non_negative_number(value, allow_none=True)
                for value in (
                    self.accelerator_seconds,
                    self.estimated_cost_usd,
                    self.device_active_wall_seconds,
                )
            )
        ):
            raise ValueError("cost totals must be finite and non-negative")

    def to_mapping(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class CostLedger:
    """All one-off and per-replay costs in the same units as ``CostRecord``."""

    reference: CostRecord
    analysis: CostRecord
    checkpoint_io: CostRecord
    verification_per_replay: CostRecord
    replay: CostRecord
    baseline_replay: CostRecord
    actual_peak_working_memory_bytes: int | None
    current_working_memory_bytes: int | None
    peak_memory_semantics: str
    checkpoint_io_wall_seconds: float
    snapshot_bytes_read: int
    snapshot_bytes_written: int
    reference_cpu: CpuCost
    analysis_cpu: CpuCost
    checkpoint_io_cpu: CpuCost
    verification_cpu_per_replay: CpuCost
    replay_cpu: CpuCost
    baseline_replay_cpu: CpuCost

    def __post_init__(self) -> None:
        for record in (
            self.reference,
            self.analysis,
            self.checkpoint_io,
            self.verification_per_replay,
            self.replay,
            self.baseline_replay,
        ):
            if not _is_non_negative_number(record.wall_seconds) or not all(
                _is_non_negative_number(value, allow_none=True)
                for value in (
                    record.accelerator_seconds,
                    record.estimated_cost_usd,
                    record.device_active_wall_seconds,
                )
            ):
                raise ValueError("cost records must be finite and non-negative")
        if not isinstance(self.peak_memory_semantics, str) or not self.peak_memory_semantics:
            raise ValueError("peak_memory_semantics must be a non-empty string")
        if (
            not _is_non_negative_integer(self.actual_peak_working_memory_bytes, allow_none=True)
            or not _is_non_negative_integer(self.current_working_memory_bytes, allow_none=True)
            or not all(
                _is_non_negative_integer(value)
                for value in (self.snapshot_bytes_read, self.snapshot_bytes_written)
            )
            or not _is_non_negative_number(self.checkpoint_io_wall_seconds)
        ):
            raise ValueError("resource measurements must be non-negative")

    @staticmethod
    def _usd(*values: float | None) -> float | None:
        return (
            None
            if any(value is None for value in values)
            else sum(value for value in values if value is not None)
        )

    @staticmethod
    def _record_totals(record: CostRecord, cpu: CpuCost) -> CostTotals:
        return CostTotals(
            record.accelerator_seconds,
            record.wall_seconds,
            cpu.seconds,
            record.estimated_cost_usd,
            record.device_active_wall_seconds,
        )

    @staticmethod
    def _sum(*totals: CostTotals) -> CostTotals:
        return CostTotals(
            accelerator_seconds=CostLedger._optional_sum(
                *(total.accelerator_seconds for total in totals)
            ),
            wall_seconds=sum(total.wall_seconds for total in totals),
            cpu_seconds=sum(total.cpu_seconds for total in totals),
            estimated_cost_usd=CostLedger._usd(*(total.estimated_cost_usd for total in totals)),
            device_active_wall_seconds=CostLedger._optional_sum(
                *(total.device_active_wall_seconds for total in totals)
            ),
        )

    @staticmethod
    def _optional_sum(*values: float | None) -> float | None:
        return (
            None
            if any(value is None for value in values)
            else sum(value for value in values if value is not None)
        )

    @property
    def total_discovery(self) -> CostTotals:
        return self._sum(
            self._record_totals(self.reference, self.reference_cpu),
            self._record_totals(self.analysis, self.analysis_cpu),
            self._record_totals(self.checkpoint_io, self.checkpoint_io_cpu),
        )

    @property
    def accelerated_per_replay(self) -> CostTotals:
        return self._sum(
            self._record_totals(self.replay, self.replay_cpu),
            self._record_totals(self.verification_per_replay, self.verification_cpu_per_replay),
        )

    @property
    def baseline_per_replay(self) -> CostTotals:
        return self._record_totals(self.baseline_replay, self.baseline_replay_cpu)

    def amortized(self, uses: int) -> CostTotals:
        if uses < 1:
            raise ValueError("uses must be positive")
        discovery, replay = self.total_discovery, self.accelerated_per_replay
        usd = None
        if discovery.estimated_cost_usd is not None and replay.estimated_cost_usd is not None:
            usd = (discovery.estimated_cost_usd + uses * replay.estimated_cost_usd) / uses
        return CostTotals(
            self._amortized_optional(
                discovery.accelerator_seconds,
                replay.accelerator_seconds,
                uses,
            ),
            (discovery.wall_seconds + uses * replay.wall_seconds) / uses,
            (discovery.cpu_seconds + uses * replay.cpu_seconds) / uses,
            usd,
            self._amortized_optional(
                discovery.device_active_wall_seconds,
                replay.device_active_wall_seconds,
                uses,
            ),
        )

    @staticmethod
    def _amortized_optional(
        discovery: float | None, replay: float | None, uses: int
    ) -> float | None:
        if discovery is None or replay is None:
            return None
        return (discovery + uses * replay) / uses

    def _break_even(self, discovery: float, baseline: float, accelerated: float) -> int | None:
        saving = baseline - accelerated
        if saving <= 0.0:
            return None
        return max(1, math.floor(discovery / saving) + 1)

    def break_even_uses(self) -> dict[str, int | None]:
        discovery, baseline, accelerated = (
            self.total_discovery,
            self.baseline_per_replay,
            self.accelerated_per_replay,
        )
        discovery_usd = discovery.estimated_cost_usd
        baseline_usd = baseline.estimated_cost_usd
        accelerated_usd = accelerated.estimated_cost_usd
        estimated_cost_usd = (
            None
            if (discovery_usd is None or baseline_usd is None or accelerated_usd is None)
            else self._break_even(discovery_usd, baseline_usd, accelerated_usd)
        )
        accelerator_seconds = self._optional_break_even(
            discovery.accelerator_seconds,
            baseline.accelerator_seconds,
            accelerated.accelerator_seconds,
        )
        device_active_wall_seconds = self._optional_break_even(
            discovery.device_active_wall_seconds,
            baseline.device_active_wall_seconds,
            accelerated.device_active_wall_seconds,
        )
        return {
            "accelerator_seconds": accelerator_seconds,
            "device_active_wall_seconds": device_active_wall_seconds,
            "wall_seconds": self._break_even(
                discovery.wall_seconds, baseline.wall_seconds, accelerated.wall_seconds
            ),
            "cpu_seconds": self._break_even(
                discovery.cpu_seconds, baseline.cpu_seconds, accelerated.cpu_seconds
            ),
            "estimated_cost_usd": estimated_cost_usd,
        }

    def _optional_break_even(
        self,
        discovery: float | None,
        baseline: float | None,
        accelerated: float | None,
    ) -> int | None:
        if discovery is None or baseline is None or accelerated is None:
            return None
        return self._break_even(discovery, baseline, accelerated)

    def to_mapping(self) -> dict[str, object]:
        return {
            "reference": asdict(self.reference),
            "analysis": asdict(self.analysis),
            "checkpoint_io": asdict(self.checkpoint_io),
            "verification_per_replay": asdict(self.verification_per_replay),
            "replay": asdict(self.replay),
            "baseline_replay": asdict(self.baseline_replay),
            "total_discovery": self.total_discovery.to_mapping(),
            "accelerated_per_replay": self.accelerated_per_replay.to_mapping(),
            "baseline_per_replay": self.baseline_per_replay.to_mapping(),
            "amortized_at_one_use": self.amortized(1).to_mapping(),
            "break_even_uses": self.break_even_uses(),
            "actual_peak_working_memory_bytes": self.actual_peak_working_memory_bytes,
            "current_working_memory_bytes": self.current_working_memory_bytes,
            "peak_memory_semantics": self.peak_memory_semantics,
            "checkpoint_io_wall_seconds": self.checkpoint_io_wall_seconds,
            "snapshot_bytes_read": self.snapshot_bytes_read,
            "snapshot_bytes_written": self.snapshot_bytes_written,
        }
