from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from .hashing import sha256_file, sha256_json
from .io import atomic_write_json, load_json


@dataclass(frozen=True)
class PredictionSeal:
    run_id: str
    compiler_sha256: str
    evidence_id: str
    source_plan_id: str
    program_id: str
    program_manifest_sha256: str
    runtime_state_sha256: str
    candidate_count: int
    selection_rule: str
    format: str = "GENOME_PREDICTION_SEAL"
    version: str = "1.0.0"

    @property
    def seal_id(self) -> str:
        return sha256_json(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": self.format,
            "version": self.version,
            "run_id": self.run_id,
            "compiler_sha256": self.compiler_sha256,
            "evidence_id": self.evidence_id,
            "source_plan_id": self.source_plan_id,
            "program_id": self.program_id,
            "program_manifest_sha256": self.program_manifest_sha256,
            "runtime_state_sha256": self.runtime_state_sha256,
            "candidate_count": self.candidate_count,
            "selection_rule": self.selection_rule,
        }

    def save(self, path: str | Path) -> None:
        value = self.to_dict()
        value["seal_id"] = self.seal_id
        atomic_write_json(path, value)

    @classmethod
    def load(cls, path: str | Path) -> "PredictionSeal":
        value = load_json(path)
        seal_id = value.pop("seal_id")
        result = cls(**value)
        if result.seal_id != seal_id:
            raise ValueError("prediction seal integrity check failed")
        return result


def build_prediction_seal(
    *,
    run_id: str,
    compiler_path: str | Path,
    evidence_id: str,
    source_plan_id: str,
    program_manifest: str | Path,
    runtime_state: str | Path,
    candidate_count: int,
    selection_rule: str,
) -> PredictionSeal:
    manifest = load_json(program_manifest)
    return PredictionSeal(
        run_id=run_id,
        compiler_sha256=sha256_file(compiler_path),
        evidence_id=evidence_id,
        source_plan_id=source_plan_id,
        program_id=str(manifest["program_id"]),
        program_manifest_sha256=sha256_file(program_manifest),
        runtime_state_sha256=sha256_file(runtime_state),
        candidate_count=candidate_count,
        selection_rule=selection_rule,
    )
