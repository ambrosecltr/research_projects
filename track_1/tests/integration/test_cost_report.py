from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from poetry50m.cli import main


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _reference(path: Path, estimated_cost_usd: float) -> dict[str, object]:
    return {
        "receipt": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "estimated_cost_usd": estimated_cost_usd,
    }


def _mps_receipt(
    *,
    checkpoint_index: int,
    wall_seconds: float,
    process_cpu_seconds: float,
    device_active_wall_seconds: float,
) -> dict[str, object]:
    return {
        "format_version": 1,
        "run_id": "mps-run",
        "checkpoint_sha256": f"{checkpoint_index:064x}",
        "steps": checkpoint_index,
        "tokens": checkpoint_index * 10,
        "wall_seconds": wall_seconds,
        "process_cpu_seconds": process_cpu_seconds,
        "accelerator_seconds": None,
        "device_active_wall_seconds": device_active_wall_seconds,
        "device": "mps",
        "actual_peak_working_memory_bytes": None,
        "current_working_memory_bytes": 1_024,
        "peak_memory_semantics": ("mps_peak_unavailable_current_allocated_reported_separately"),
        "checkpoint_io_wall_seconds": 0.25,
        "snapshot_bytes_read": 512,
        "snapshot_bytes_written": 256,
    }


def _cuda_receipt(
    *,
    checkpoint_index: int,
    wall_seconds: float,
    process_cpu_seconds: float,
    accelerator_seconds: float,
    device_active_wall_seconds: float,
) -> dict[str, object]:
    receipt = _mps_receipt(
        checkpoint_index=checkpoint_index,
        wall_seconds=wall_seconds,
        process_cpu_seconds=process_cpu_seconds,
        device_active_wall_seconds=device_active_wall_seconds,
    )
    receipt.update(
        {
            "accelerator_seconds": accelerator_seconds,
            "device": "cuda",
            "actual_peak_working_memory_bytes": 2_048,
            "peak_memory_semantics": "cuda_max_memory_allocated_since_command_reset",
        }
    )
    return receipt


def test_cost_report_reconciles_known_accelerator_components(tmp_path: Path) -> None:
    reference_receipt = tmp_path / "cuda-reference.receipt.json"
    replay_receipt = tmp_path / "cuda-replay.receipt.json"
    baseline_receipt = tmp_path / "cuda-baseline.receipt.json"
    analysis_receipt = tmp_path / "cuda-analysis.receipt.json"
    _write_json(
        reference_receipt,
        _cuda_receipt(
            checkpoint_index=1,
            wall_seconds=3.0,
            process_cpu_seconds=1.5,
            accelerator_seconds=2.0,
            device_active_wall_seconds=3.0,
        ),
    )
    _write_json(
        replay_receipt,
        _cuda_receipt(
            checkpoint_index=2,
            wall_seconds=1.0,
            process_cpu_seconds=0.5,
            accelerator_seconds=0.6,
            device_active_wall_seconds=1.0,
        ),
    )
    _write_json(
        baseline_receipt,
        _cuda_receipt(
            checkpoint_index=3,
            wall_seconds=3.0,
            process_cpu_seconds=1.5,
            accelerator_seconds=2.0,
            device_active_wall_seconds=3.0,
        ),
    )
    analysis_value = _cuda_receipt(
        checkpoint_index=4,
        wall_seconds=3.5,
        process_cpu_seconds=1.75,
        accelerator_seconds=1.0,
        device_active_wall_seconds=0.5,
    )
    analysis_value["steps"] = 0
    analysis_value["tokens"] = 0
    analysis_value["cost_components"] = {
        "analysis": {
            "steps": 0,
            "tokens": 0,
            "wall_seconds": 2.0,
            "process_cpu_seconds": 1.0,
            "accelerator_seconds": 0.0,
            "device_active_wall_seconds": 0.0,
            "timing_scope": "CPU-only forecast and command remainder",
        },
        "checkpoint_io": {
            "steps": 0,
            "tokens": 0,
            "wall_seconds": 1.0,
            "process_cpu_seconds": 0.5,
            "accelerator_seconds": 0.0,
            "device_active_wall_seconds": 0.0,
            "timing_scope": "checkpoint and snapshot IO",
        },
        "verification_per_replay": {
            "steps": 0,
            "tokens": 0,
            "wall_seconds": 0.5,
            "process_cpu_seconds": 0.25,
            "accelerator_seconds": 1.0,
            "device_active_wall_seconds": 0.5,
            "timing_scope": "synchronized verification",
        },
    }
    _write_json(analysis_receipt, analysis_value)
    analysis_reference = _reference(analysis_receipt, 0.0)
    assembly_value = {
        "format_version": 1,
        "records": {
            "reference": _reference(reference_receipt, 0.0),
            "analysis": analysis_reference,
            "checkpoint_io": analysis_reference,
            "verification_per_replay": analysis_reference,
            "replay": _reference(replay_receipt, 0.0),
            "baseline_replay": _reference(baseline_receipt, 0.0),
        },
        "resource_receipt": _reference(reference_receipt, 0.0),
        "amortized_uses": [2],
    }
    assembly = tmp_path / "cuda-cost-assembly.json"
    output = tmp_path / "cuda-cost-report.json"
    _write_json(assembly, assembly_value)

    assert main(("cost-report", "--input", str(assembly), "--output", str(output))) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["analysis"]["accelerator_seconds"] == 0.0
    assert report["verification_per_replay"]["accelerator_seconds"] == 1.0
    assert report["total_discovery"]["accelerator_seconds"] == 2.0

    assembly_value["format_version"] = True
    _write_json(assembly, assembly_value)
    with pytest.raises(SystemExit, match="cost assembly"):
        main(("cost-report", "--input", str(assembly), "--output", str(output)))
    assembly_value["format_version"] = 1

    analysis_value["accelerator_seconds"] = 1.25
    _write_json(analysis_receipt, analysis_value)
    mismatched_hash = hashlib.sha256(analysis_receipt.read_bytes()).hexdigest()
    for name in ("analysis", "checkpoint_io", "verification_per_replay"):
        assembly_value["records"][name]["sha256"] = mismatched_hash
    _write_json(assembly, assembly_value)
    with pytest.raises(SystemExit, match="do not reconcile"):
        main(
            (
                "cost-report",
                "--input",
                str(assembly),
                "--output",
                str(tmp_path / "cuda-mismatch-report.json"),
            )
        )


def test_mps_receipts_preserve_unknown_accelerator_and_peak_memory(
    tmp_path: Path,
) -> None:
    reference_receipt = tmp_path / "reference.receipt.json"
    replay_receipt = tmp_path / "replay.receipt.json"
    baseline_receipt = tmp_path / "baseline.receipt.json"
    analysis_receipt = tmp_path / "analysis.receipt.json"
    _write_json(
        reference_receipt,
        _mps_receipt(
            checkpoint_index=1,
            wall_seconds=3.0,
            process_cpu_seconds=1.5,
            device_active_wall_seconds=3.0,
        ),
    )
    _write_json(
        replay_receipt,
        _mps_receipt(
            checkpoint_index=2,
            wall_seconds=1.0,
            process_cpu_seconds=0.5,
            device_active_wall_seconds=1.0,
        ),
    )
    _write_json(
        baseline_receipt,
        _mps_receipt(
            checkpoint_index=3,
            wall_seconds=3.0,
            process_cpu_seconds=1.5,
            device_active_wall_seconds=3.0,
        ),
    )
    analysis_value = _mps_receipt(
        checkpoint_index=4,
        wall_seconds=3.5,
        process_cpu_seconds=1.75,
        device_active_wall_seconds=0.5,
    )
    analysis_value["steps"] = 0
    analysis_value["tokens"] = 0
    analysis_value["cost_components"] = {
        "analysis": {
            "steps": 0,
            "tokens": 0,
            "wall_seconds": 2.0,
            "process_cpu_seconds": 1.0,
            "accelerator_seconds": None,
            "device_active_wall_seconds": 0.0,
            "timing_scope": "analysis command remainder",
        },
        "checkpoint_io": {
            "steps": 0,
            "tokens": 0,
            "wall_seconds": 1.0,
            "process_cpu_seconds": 0.5,
            "accelerator_seconds": None,
            "device_active_wall_seconds": 0.0,
            "timing_scope": "checkpoint and snapshot IO",
        },
        "verification_per_replay": {
            "steps": 0,
            "tokens": 0,
            "wall_seconds": 0.5,
            "process_cpu_seconds": 0.25,
            "accelerator_seconds": None,
            "device_active_wall_seconds": 0.5,
            "timing_scope": "synchronized verification",
        },
    }
    _write_json(analysis_receipt, analysis_value)

    analysis_reference = _reference(analysis_receipt, 2.0)
    assembly_value = {
        "format_version": 1,
        "records": {
            "reference": _reference(reference_receipt, 6.0),
            "analysis": analysis_reference,
            "checkpoint_io": {
                **analysis_reference,
                "estimated_cost_usd": 1.0,
            },
            "verification_per_replay": {
                **analysis_reference,
                "estimated_cost_usd": 0.5,
            },
            "replay": _reference(replay_receipt, 1.0),
            "baseline_replay": _reference(baseline_receipt, 3.0),
        },
        "resource_receipt": _reference(reference_receipt, 6.0),
        "amortized_uses": [2],
    }
    assembly = tmp_path / "cost-assembly.json"
    _write_json(assembly, assembly_value)
    output = tmp_path / "cost-report.json"

    assert main(("cost-report", "--input", str(assembly), "--output", str(output))) == 0

    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["total_discovery"]["accelerator_seconds"] is None
    assert report["accelerated_per_replay"]["accelerator_seconds"] is None
    assert report["baseline_per_replay"]["accelerator_seconds"] is None
    assert report["amortized"]["2"]["accelerator_seconds"] is None
    assert report["total_discovery"]["wall_seconds"] == 6.0
    assert report["total_discovery"]["cpu_seconds"] == 3.0
    assert report["total_discovery"]["device_active_wall_seconds"] == 3.0
    assert report["total_discovery"]["estimated_cost_usd"] == 9.0
    assert report["amortized"]["2"]["wall_seconds"] == 4.5
    assert report["amortized"]["2"]["estimated_cost_usd"] == 6.0
    assert report["break_even_uses"] == {
        "accelerator_seconds": None,
        "cpu_seconds": 5,
        "device_active_wall_seconds": 3,
        "estimated_cost_usd": 7,
        "wall_seconds": 5,
    }
    assert report["actual_peak_working_memory_bytes"] is None
    assert report["current_working_memory_bytes"] == 1_024
    assert (
        report["peak_memory_semantics"]
        == "mps_peak_unavailable_current_allocated_reported_separately"
    )
    evidence = report["evidence"]["records"]
    assert evidence["analysis"]["sha256"] == evidence["checkpoint_io"]["sha256"]
    assert evidence["analysis"]["sha256"] == evidence["verification_per_replay"]["sha256"]

    non_reconciling = json.loads(json.dumps(analysis_value))
    non_reconciling["cost_components"]["analysis"]["wall_seconds"] = 2.25
    _write_json(analysis_receipt, non_reconciling)
    non_reconciling_hash = hashlib.sha256(analysis_receipt.read_bytes()).hexdigest()
    for name in ("analysis", "checkpoint_io", "verification_per_replay"):
        assembly_value["records"][name]["sha256"] = non_reconciling_hash
    _write_json(assembly, assembly_value)
    with pytest.raises(SystemExit, match="do not reconcile"):
        main(
            (
                "cost-report",
                "--input",
                str(assembly),
                "--output",
                str(tmp_path / "non-reconciling-report.json"),
            )
        )

    malformed = json.loads(json.dumps(analysis_value))
    malformed["cost_components"]["analysis"]["unexpected"] = True
    _write_json(analysis_receipt, malformed)
    malformed_hash = hashlib.sha256(analysis_receipt.read_bytes()).hexdigest()
    for name in ("analysis", "checkpoint_io", "verification_per_replay"):
        assembly_value["records"][name]["sha256"] = malformed_hash
    _write_json(assembly, assembly_value)
    with pytest.raises(SystemExit, match="must contain exactly"):
        main(
            (
                "cost-report",
                "--input",
                str(assembly),
                "--output",
                str(tmp_path / "malformed-report.json"),
            )
        )

    missing = json.loads(json.dumps(analysis_value))
    del missing["cost_components"]["analysis"]["tokens"]
    _write_json(analysis_receipt, missing)
    missing_hash = hashlib.sha256(analysis_receipt.read_bytes()).hexdigest()
    for name in ("analysis", "checkpoint_io", "verification_per_replay"):
        assembly_value["records"][name]["sha256"] = missing_hash
    _write_json(assembly, assembly_value)
    with pytest.raises(SystemExit, match="must contain exactly"):
        main(
            (
                "cost-report",
                "--input",
                str(assembly),
                "--output",
                str(tmp_path / "missing-field-report.json"),
            )
        )

    invalid_number = json.loads(json.dumps(analysis_value))
    invalid_number["cost_components"]["analysis"]["accelerator_seconds"] = True
    _write_json(analysis_receipt, invalid_number)
    invalid_number_hash = hashlib.sha256(analysis_receipt.read_bytes()).hexdigest()
    for name in ("analysis", "checkpoint_io", "verification_per_replay"):
        assembly_value["records"][name]["sha256"] = invalid_number_hash
    _write_json(assembly, assembly_value)
    with pytest.raises(SystemExit, match="must be numeric"):
        main(
            (
                "cost-report",
                "--input",
                str(assembly),
                "--output",
                str(tmp_path / "invalid-number-report.json"),
            )
        )
