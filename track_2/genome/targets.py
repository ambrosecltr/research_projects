from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from .acceptance import accept_target_program
from .architecture import ArchitectureGraph
from .data import causal_batches_from_jsonl
from .evaluation import FunctionalGate, evaluate_program
from .hashing import sha256_file
from .io import atomic_write_json, load_json
from .mgp import (
    FitConfig,
    audit_program,
    fit_low_rank_program,
    refine_program_functionally,
    save_program,
)
from .protocol import ArtifactBinding, TargetFormula, clean_code_commit
from .sampling import verify_independent_evaluation_sample
from .sources import SourcePlan
from .state import direct_fp16_delta_bytes, load_state, state_id


def produce_target(
    *,
    plan_path: str | Path,
    formula_path: str | Path,
    run_id: str,
    workspace: str | Path,
    output: str | Path,
    repository: str | Path,
) -> dict[str, Any]:
    """Fit, refine, serialize, evaluate and accept one target from one formula."""
    try:
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("transformers is required for target production") from error

    plan = SourcePlan.load(plan_path)
    formula = TargetFormula.load(formula_path)
    life = next((item for item in plan.lives if item.run_id == run_id), None)
    if life is None:
        raise ValueError(f"source plan has no life {run_id}")
    if life.split == "hidden":
        raise ValueError("hidden WT is not available for target production")
    if life.split == "development" and formula.status != "frozen":
        raise ValueError("development targets require the global formula to be frozen")
    code_commit = clean_code_commit(repository)

    root = Path(workspace)
    target = Path(output)
    if target.exists():
        raise FileExistsError(target)
    if life.split == "development":
        _require_regenerated_training_targets(
            plan=plan,
            formula=formula,
            program_root=target.parent,
        )
    canonical = root / "canonical" / "lives" / run_id
    source = root / "source" / "hf" / run_id
    w0_path = canonical / "w0.safetensors"
    wt_path = canonical / "wt.safetensors"
    graph = ArchitectureGraph.from_dict(load_json(canonical / "architecture.json"))
    w0 = load_state(w0_path)
    wt = load_state(wt_path)

    fit_values = dict(formula.fit)
    program, payloads = fit_low_rank_program(
        w0,
        wt,
        graph,
        config=FitConfig(**fit_values, account_for_serialization=True),
    )
    model = GPTNeoXForCausalLM.from_pretrained(
        str(source / "w0"), local_files_only=True, dtype=torch.float32
    )
    teacher = GPTNeoXForCausalLM.from_pretrained(
        str(source / "wt"), local_files_only=True, dtype=torch.float32
    )
    refinement = formula.refinement
    for stage in refinement["stages"]:
        payloads = refine_program_functionally(
            model,
            w0,
            program,
            payloads,
            causal_batches_from_jsonl(formula.data["refinement"]),
            steps=int(stage["steps"]),
            learning_rate=float(stage["learning_rate"]),
            teacher_model=teacher,
            kl_weight=float(refinement["teacher_kl_weight"]),
            anchor_weight=float(refinement["anchor_weight"]),
            device=str(refinement["device"]),
        )

    accounting = save_program(target, program, payloads)
    audit = audit_program(
        program,
        payloads,
        direct_fp16_delta_bytes=direct_fp16_delta_bytes(wt),
        artifact_directory=target,
    )
    atomic_write_json(target / "structural_audit.json", asdict(audit))
    manifest = load_json(target / "manifest.json")

    if life.split == "development":
        formula_sample_receipt = Path(formula.data["formula_sample_receipt"])
        verifier_receipt = Path(formula.data["development_verifier_receipt"])
        verification = verify_independent_evaluation_sample(
            formula_sample_receipt=formula_sample_receipt,
            verifier_receipt=verifier_receipt,
            minimum_batches=int(formula.data["development_evaluation_batches"]),
        )
        evaluation_jsonl = Path(verification["evaluation_jsonl"])
        max_batches = int(formula.data["development_evaluation_batches"])
    else:
        evaluation_jsonl = Path(formula.data["formula_tuning"])
        max_batches = int(formula.data["formula_tuning_batches"])

    config = GPTNeoXConfig.from_pretrained(str(source / "w0"), local_files_only=True)
    comparison = evaluate_program(
        model_factory=lambda: GPTNeoXForCausalLM(config),
        base_state=w0,
        program=program,
        payloads=payloads,
        batches=causal_batches_from_jsonl(evaluation_jsonl),
        endpoint_state=wt,
        device=str(refinement["device"]),
        max_batches=max_batches,
    )
    binding = ArtifactBinding(
        run_id=run_id,
        formula_id=formula.formula_id,
        program_id=str(manifest["program_id"]),
        program_manifest_sha256=sha256_file(target / "manifest.json"),
        payload_sha256=str(manifest["payload_sha256"]),
        w0_state_id=state_id(w0),
        wt_state_id=state_id(wt),
        evaluation_jsonl_sha256=sha256_file(evaluation_jsonl),
        source_plan_id=plan.plan_id,
        code_commit=code_commit,
    )
    evaluation_path = target / "evaluation.json"
    atomic_write_json(
        evaluation_path,
        {
            "format": "GENOME_TARGET_EVALUATION",
            "version": "2.0.0",
            "binding": binding.to_dict(),
            "split": life.split,
            "evaluation_jsonl": str(evaluation_jsonl),
            "evaluation_batches": max_batches,
            "comparison": comparison.to_dict(),
        },
    )
    acceptance_values = formula.acceptance
    report = accept_target_program(
        program_directory=target,
        reference_state_path=wt_path,
        evaluation_report_path=evaluation_path,
        gate=FunctionalGate(
            maximum_target_fraction=float(acceptance_values["maximum_target_fraction"]),
            minimum_development_progress=float(
                acceptance_values["minimum_endpoint_progress"]
            ),
        ),
        expected_binding=binding,
    )
    return {
        "run_id": run_id,
        "split": life.split,
        "formula_id": formula.formula_id,
        "accounting": accounting,
        "audit": asdict(audit),
        "acceptance": report,
    }


def _require_regenerated_training_targets(
    *,
    plan: SourcePlan,
    formula: TargetFormula,
    program_root: Path,
) -> None:
    rejected = {str(item) for item in formula.corpus["rejected_training_lives"]}
    for training_life in (item for item in plan.lives if item.split == "training"):
        acceptance_path = program_root / training_life.run_id / "acceptance.json"
        if not acceptance_path.is_file():
            raise ValueError(
                f"development is locked until training target {training_life.run_id} is regenerated"
            )
        report = load_json(acceptance_path)
        binding = ArtifactBinding.from_dict(report["binding"])
        if binding.formula_id != formula.formula_id or binding.source_plan_id != plan.plan_id:
            raise ValueError(
                f"training target {training_life.run_id} does not use the frozen formula and plan"
            )
        expected = training_life.run_id not in rejected
        if report.get("accepted") is not expected:
            outcome = "accepted" if expected else "rejected"
            raise ValueError(
                f"training target {training_life.run_id} must be {outcome} before development"
            )
