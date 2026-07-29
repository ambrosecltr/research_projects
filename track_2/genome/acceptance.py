from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .evaluation import ComparisonResult, EvaluationResult, FunctionalGate
from .hashing import sha256_file
from .io import atomic_write_json, load_json
from .mgp.policy import ProgramPolicy, audit_program
from .mgp.serialize import load_program
from .protocol import ArtifactBinding
from .state import direct_fp16_delta_bytes, load_state


def _evaluation(value: Mapping[str, Any]) -> EvaluationResult:
    return EvaluationResult(**value)


def _comparison(value: Mapping[str, Any]) -> ComparisonResult:
    return ComparisonResult(
        w0=_evaluation(value["w0"]),
        candidate=_evaluation(value["candidate"]),
        endpoint=None if value.get("endpoint") is None else _evaluation(value["endpoint"]),
        decode_seconds=float(value["decode_seconds"]),
        parameter_distortion=(
            None
            if value.get("parameter_distortion") is None
            else float(value["parameter_distortion"])
        ),
        endpoint_progress=value.get("endpoint_progress"),
        candidate_beats_w0=bool(value["candidate_beats_w0"]),
        logit_kl_to_endpoint=value.get("logit_kl_to_endpoint"),
        top1_agreement=value.get("top1_agreement"),
    )


def accept_target_program(
    *,
    program_directory: str | Path,
    reference_state_path: str | Path,
    evaluation_report_path: str | Path,
    gate: FunctionalGate = FunctionalGate(),
    policy: ProgramPolicy = ProgramPolicy(),
    expected_binding: ArtifactBinding | None = None,
) -> dict[str, Any]:
    root = Path(program_directory)
    program, payloads, manifest = load_program(root)
    audit = audit_program(
        program,
        payloads,
        direct_fp16_delta_bytes=direct_fp16_delta_bytes(load_state(reference_state_path)),
        artifact_directory=root,
        policy=policy,
    )
    evaluation_report = load_json(evaluation_report_path)
    if evaluation_report.get("format") != "GENOME_TARGET_EVALUATION":
        raise ValueError("target evaluation report has the wrong format")
    binding = ArtifactBinding.from_dict(evaluation_report["binding"])
    if expected_binding is not None and binding != expected_binding:
        raise ValueError("target evaluation binding differs from the expected binding")
    if binding.program_id != manifest["program_id"]:
        raise ValueError("evaluation binding program_id differs from the program")
    if binding.program_manifest_sha256 != sha256_file(root / "manifest.json"):
        raise ValueError("evaluation binding manifest SHA differs from the program")
    if binding.payload_sha256 != manifest["payload_sha256"]:
        raise ValueError("evaluation binding payload SHA differs from the program")
    comparison = _comparison(evaluation_report["comparison"])
    accepted = bool(
        audit.primary_budget_pass
        and audit.accepted_structure
        and audit.serialized
        and gate.accept_development(comparison, audit.byte_fraction or 1.0)
    )
    report = {
        "format": "GENOME_TARGET_ACCEPTANCE",
        "version": "2.0.0",
        "accepted": accepted,
        "binding": binding.to_dict(),
        "evaluation_report_sha256": sha256_file(evaluation_report_path),
        "audit": asdict(audit),
        "functional_gate": asdict(gate),
        "comparison": comparison.to_dict(),
    }
    atomic_write_json(root / "acceptance.json", report)
    return report
