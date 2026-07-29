from __future__ import annotations

import re
import subprocess
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .hashing import sha256_json
from .io import load_yaml

SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class TargetFormula:
    formula_id: str
    status: str
    fit: Mapping[str, Any]
    refinement: Mapping[str, Any]
    data: Mapping[str, Any]
    acceptance: Mapping[str, Any]
    corpus: Mapping[str, Any]
    endpoint_semantics: str

    def __post_init__(self) -> None:
        if self.status not in {"formula-development", "frozen"}:
            raise ValueError("target formula status must be formula-development or frozen")
        if self.formula_id != sha256_json(self.identity_document):
            raise ValueError("declared formula_id does not match the target formula")
        if float(self.fit["budget_fraction"]) != 0.10:
            raise ValueError("the Pythia v1 target byte budget must remain 10%")
        if float(self.acceptance["maximum_target_fraction"]) != 0.10:
            raise ValueError("the Pythia v1 acceptance byte budget must remain 10%")
        if float(self.acceptance["minimum_endpoint_progress"]) != 0.80:
            raise ValueError("the Pythia v1 development gate must remain 80%")
        if int(self.data["development_evaluation_batches"]) < 128:
            raise ValueError("development verification requires at least 128 batches")

    @property
    def identity_document(self) -> dict[str, Any]:
        return {
            "fit": dict(self.fit),
            "refinement": dict(self.refinement),
            "data": dict(self.data),
            "acceptance": dict(self.acceptance),
            "endpoint_semantics": self.endpoint_semantics,
        }

    @classmethod
    def load(cls, path: str | Path) -> TargetFormula:
        value = load_yaml(path)
        formula = cls(
            formula_id=str(value["formula_id"]),
            status=str(value["status"]),
            fit=value["fit"],
            refinement=value["refinement"],
            data=value["data"],
            acceptance=value["acceptance"],
            corpus=value["corpus"],
            endpoint_semantics=str(value["endpoint_semantics"]),
        )
        return formula


@dataclass(frozen=True)
class ArtifactBinding:
    run_id: str
    formula_id: str
    program_id: str
    program_manifest_sha256: str
    payload_sha256: str
    w0_state_id: str
    wt_state_id: str
    evaluation_jsonl_sha256: str
    source_plan_id: str
    code_commit: str

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("artifact binding requires run_id")
        for name in (
            "formula_id",
            "program_id",
            "program_manifest_sha256",
            "payload_sha256",
            "w0_state_id",
            "wt_state_id",
            "evaluation_jsonl_sha256",
            "source_plan_id",
        ):
            if not SHA256_PATTERN.fullmatch(getattr(self, name)):
                raise ValueError(f"artifact binding {name} must be a SHA-256 digest")
        if not re.fullmatch(r"[0-9a-f]{40,64}", self.code_commit):
            raise ValueError("artifact binding code_commit must be a full Git commit")

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> ArtifactBinding:
        return cls(**{name: str(value[name]) for name in cls.__dataclass_fields__})

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


def require_matching_bindings(
    left: ArtifactBinding,
    right: ArtifactBinding,
    *,
    context: str,
) -> None:
    if left != right:
        changed = [
            name
            for name in ArtifactBinding.__dataclass_fields__
            if getattr(left, name) != getattr(right, name)
        ]
        raise ValueError(f"{context} bindings differ: {', '.join(changed)}")


def clean_code_commit(repository: str | Path) -> str:
    root = Path(repository)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise ValueError("production target generation requires a clean committed worktree")
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
