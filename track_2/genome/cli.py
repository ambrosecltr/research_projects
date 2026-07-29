from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import torch
import typer

from .acceptance import accept_target_program
from .adapters import GPTNeoXAdapter
from .architecture import ArchitectureGraph
from .compiler import (
    CompilerConfig,
    CompilerCorpus,
    GenomeCompiler,
    TrainingConfig,
    build_compiler_example,
    train_compiler,
)
from .data import causal_batches_from_jsonl, raw_texts_from_jsonl, token_sequences_from_jsonl
from .evaluation import FunctionalGate, evaluate_program
from .fingerprint import (
    FingerprintBundle,
    corpus_fingerprint,
    merge_fingerprints,
    w0_response_fingerprint,
)
from .hidden import build_prediction_seal
from .io import atomic_write_json, load_json, load_yaml
from .life import ModelLife
from .mgp import (
    audit_program,
    execute_program,
    fit_low_rank_program,
    load_program,
    refine_program_functionally,
    save_program,
)
from .mgp.fit import FitConfig
from .prepare import canonicalize_pythia_life, prepare_pythia_life
from .sampling import prepare_dataset_sample
from .sources import (
    SourcePlan,
    default_pythia_v1_plan,
    materialize_plan,
    resolve_plan,
    reveal_hidden_endpoint,
)
from .state import direct_fp16_delta_bytes, load_state, save_state, state_id
from .workspace import initialize_workspace

app = typer.Typer(
    no_args_is_help=True, help="GENOME: compile model lives into compact executable model programs."
)


def _echo_json(value) -> None:
    typer.echo(json.dumps(value, sort_keys=True, indent=2))


@app.command("init-workspace")
def init_workspace(
    root: Path = typer.Option(..., help="Fresh RunPod network-volume workspace root."),
) -> None:
    _echo_json(initialize_workspace(root))


@app.command("write-source-plan")
def write_source_plan(output: Path = typer.Option(...)) -> None:
    plan = default_pythia_v1_plan()
    plan.save(output)
    _echo_json({"output": str(output), "plan_id": plan.plan_id, "lives": len(plan.lives)})


@app.command("resolve-source-plan")
def resolve_source_plan(
    plan_path: Path = typer.Option(..., "--plan"),
    output: Path = typer.Option(...),
    token: Optional[str] = typer.Option(None, envvar="HF_TOKEN"),
) -> None:
    plan = resolve_plan(SourcePlan.load(plan_path), token=token)
    plan.save(output)
    _echo_json(
        {
            "output": str(output),
            "plan_id": plan.plan_id,
            "pinned_for_materialization": plan.pinned_for_materialization,
            "dataset_commit": plan.dataset_commit,
            "order_commit": plan.order_commit,
        }
    )


@app.command("materialize-sources")
def materialize_sources(
    plan_path: Path = typer.Option(..., "--plan"),
    workspace: Path = typer.Option(...),
    token: Optional[str] = typer.Option(None, envvar="HF_TOKEN"),
) -> None:
    _echo_json(materialize_plan(SourcePlan.load(plan_path), root=workspace, token=token))


@app.command("prepare-dataset-sample")
def prepare_dataset_sample_command(
    repository: str = typer.Option(...),
    revision: str = typer.Option(...),
    tokenizer_path: Path = typer.Option(...),
    output: Path = typer.Option(...),
    filename: str = typer.Option("document-00000-of-00020.bin"),
    examples: int = typer.Option(4096),
    context_length: int = typer.Option(2048),
    seed: int = typer.Option(20260729),
) -> None:
    _echo_json(
        prepare_dataset_sample(
            repository=repository,
            revision=revision,
            tokenizer_path=tokenizer_path,
            output=output,
            filename=filename,
            examples=examples,
            context_length=context_length,
            seed=seed,
        )
    )


@app.command("prepare-life")
def prepare_life_command(
    plan_path: Path = typer.Option(..., "--plan"),
    run_id: str = typer.Option(...),
    workspace: Path = typer.Option(...),
    recipe_path: Path = typer.Option(...),
    evidence_directory: Path = typer.Option(...),
) -> None:
    life = prepare_pythia_life(
        plan=SourcePlan.load(plan_path),
        run_id=run_id,
        workspace=workspace,
        recipe_path=recipe_path,
        evidence_directory=evidence_directory,
    )
    _echo_json({"run_id": life.run_id, "manifest_id": life.manifest_id})


@app.command("canonicalize-life")
def canonicalize_life_command(
    plan_path: Path = typer.Option(..., "--plan"),
    run_id: str = typer.Option(...),
    workspace: Path = typer.Option(...),
) -> None:
    _echo_json(
        canonicalize_pythia_life(
            plan=SourcePlan.load(plan_path),
            run_id=run_id,
            workspace=workspace,
        )
    )


@app.command("validate-life")
def validate_life(path: Path) -> None:
    life = ModelLife.load(path)
    _echo_json(
        {"run_id": life.run_id, "split": life.split, "manifest_id": life.manifest_id, "valid": True}
    )


@app.command("export-graph")
def export_graph(
    state_path: Path = typer.Option(..., "--state"),
    config_path: Path = typer.Option(..., "--config"),
    output: Path = typer.Option(...),
) -> None:
    state = load_state(state_path)
    config = load_json(config_path)
    graph = GPTNeoXAdapter.graph(state, config)
    atomic_write_json(output, graph.to_dict())
    _echo_json({"output": str(output), "graph_id": graph.graph_id, "tensors": len(graph.tensors)})


@app.command("fingerprint-corpus")
def fingerprint_corpus(
    token_jsonl: Path = typer.Option(...),
    output: Path = typer.Option(...),
    text_jsonl: Optional[Path] = typer.Option(None),
) -> None:
    bundle = corpus_fingerprint(
        token_sequences_from_jsonl(token_jsonl),
        raw_texts=None if text_jsonl is None else raw_texts_from_jsonl(text_jsonl),
    )
    bundle.save(output)
    _echo_json({"output": str(output), "fingerprint_id": bundle.fingerprint_id})


@app.command("fingerprint-w0")
def fingerprint_w0(
    snapshot: Path = typer.Option(...),
    probe_jsonl: Path = typer.Option(...),
    output: Path = typer.Option(...),
    device: str = typer.Option("cpu"),
) -> None:
    try:
        from transformers import GPTNeoXForCausalLM
    except ImportError as error:
        raise typer.BadParameter("transformers is required") from error
    model = GPTNeoXForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, dtype=torch.float32
    )
    graph = GPTNeoXAdapter.graph(model.state_dict(), model.config.to_dict())
    roles = {node.name: node.role for node in graph.tensors}
    bundle = w0_response_fingerprint(
        model,
        causal_batches_from_jsonl(probe_jsonl),
        role_by_parameter=roles,
        device=device,
    )
    bundle.save(output)
    _echo_json({"output": str(output), "fingerprint_id": bundle.fingerprint_id})


@app.command("merge-fingerprints")
def merge_fingerprint_command(
    inputs: list[Path] = typer.Argument(...),
    output: Path = typer.Option(...),
) -> None:
    bundle = merge_fingerprints(*(FingerprintBundle.load(path) for path in inputs))
    bundle.save(output)
    _echo_json({"output": str(output), "fingerprint_id": bundle.fingerprint_id})


@app.command("fit-compact-target")
def fit_compact_target(
    w0_path: Path = typer.Option(..., "--w0"),
    wt_path: Path = typer.Option(..., "--wt"),
    graph_path: Path = typer.Option(..., "--graph"),
    output: Path = typer.Option(...),
    budget_fraction: float = typer.Option(0.10),
    max_rank: int = typer.Option(32),
    svd_method: str = typer.Option("randomized"),
    device: str = typer.Option("cpu"),
) -> None:
    w0 = load_state(w0_path)
    wt = load_state(wt_path)
    graph = ArchitectureGraph.from_dict(load_json(graph_path))
    program, payloads = fit_low_rank_program(
        w0,
        wt,
        graph,
        config=FitConfig(
            budget_fraction=budget_fraction,
            max_rank=max_rank,
            account_for_serialization=True,
            svd_method=svd_method,
            device=device,
        ),
    )
    accounting = save_program(output, program, payloads)
    audit = audit_program(
        program,
        payloads,
        direct_fp16_delta_bytes=direct_fp16_delta_bytes(wt),
        artifact_directory=output,
    )
    atomic_write_json(output / "structural_audit.json", asdict(audit))
    _echo_json({"accounting": accounting, "audit": asdict(audit), "candidate_only": True})


@app.command("refine-compact-target")
def refine_compact_target(
    w0_snapshot: Path = typer.Option(...),
    program_path: Path = typer.Option(..., "--program"),
    probe_jsonl: Path = typer.Option(...),
    output: Path = typer.Option(...),
    teacher_snapshot: Optional[Path] = typer.Option(None),
    steps: int = typer.Option(100),
    learning_rate: float = typer.Option(0.001),
    kl_weight: float = typer.Option(0.1),
    device: str = typer.Option("cuda"),
) -> None:
    try:
        from transformers import GPTNeoXForCausalLM
    except ImportError as error:
        raise typer.BadParameter("transformers is required") from error
    model = GPTNeoXForCausalLM.from_pretrained(
        str(w0_snapshot), local_files_only=True, dtype=torch.float32
    )
    base_state = GPTNeoXAdapter.canonical_state(model.state_dict())
    teacher = None
    if teacher_snapshot is not None:
        teacher = GPTNeoXForCausalLM.from_pretrained(
            str(teacher_snapshot), local_files_only=True, dtype=torch.float32
        )
    program, payloads, _ = load_program(program_path)
    refined = refine_program_functionally(
        model,
        base_state,
        program,
        payloads,
        causal_batches_from_jsonl(probe_jsonl),
        steps=steps,
        learning_rate=learning_rate,
        teacher_model=teacher,
        kl_weight=kl_weight if teacher is not None else 0.0,
        device=device,
    )
    accounting = save_program(output, program, refined)
    audit = audit_program(
        program,
        refined,
        direct_fp16_delta_bytes=direct_fp16_delta_bytes(base_state),
        artifact_directory=output,
    )
    atomic_write_json(output / "structural_audit.json", asdict(audit))
    _echo_json(
        {
            "output": str(output),
            "accounting": accounting,
            "audit": asdict(audit),
            "candidate_only": True,
        }
    )


@app.command("audit-program")
def audit_program_command(
    program_path: Path = typer.Option(..., "--program"),
    reference_state: Path = typer.Option(..., help="WT or W0 state used only for parameter count."),
) -> None:
    program, payloads, _ = load_program(program_path)
    audit = audit_program(
        program,
        payloads,
        direct_fp16_delta_bytes=direct_fp16_delta_bytes(load_state(reference_state)),
        artifact_directory=program_path,
    )
    _echo_json(asdict(audit))
    if not audit.eligible_for_function_gate:
        raise typer.Exit(code=2)


@app.command("accept-target")
def accept_target(
    program_path: Path = typer.Option(..., "--program"),
    reference_state: Path = typer.Option(...),
    evaluation_report: Path = typer.Option(...),
    minimum_progress: float = typer.Option(0.80),
) -> None:
    report = accept_target_program(
        program_directory=program_path,
        reference_state_path=reference_state,
        evaluation_report_path=evaluation_report,
        gate=FunctionalGate(minimum_development_progress=minimum_progress),
    )
    _echo_json(report)
    if not report["accepted"]:
        raise typer.Exit(code=2)


@app.command("decode")
def decode(
    w0_path: Path = typer.Option(..., "--w0"),
    program_path: Path = typer.Option(..., "--program"),
    output: Path = typer.Option(...),
) -> None:
    w0 = load_state(w0_path)
    program, payloads, _ = load_program(program_path)
    state = execute_program(w0, program, payloads)
    save_state(output, state)
    _echo_json({"output": str(output), "state_id": state_id(state)})


@app.command("evaluate-program")
def evaluate_program_command(
    w0_snapshot: Path = typer.Option(...),
    program_path: Path = typer.Option(...),
    evaluation_jsonl: Path = typer.Option(...),
    output: Path = typer.Option(...),
    wt_snapshot: Optional[Path] = typer.Option(None),
    device: str = typer.Option("cpu"),
    max_batches: int = typer.Option(16),
) -> None:
    try:
        from transformers import GPTNeoXConfig, GPTNeoXForCausalLM
    except ImportError as error:
        raise typer.BadParameter("transformers is required") from error
    w0_model = GPTNeoXForCausalLM.from_pretrained(
        str(w0_snapshot), local_files_only=True, dtype=torch.float32
    )
    w0 = GPTNeoXAdapter.canonical_state(w0_model.state_dict())
    endpoint_state = None
    if wt_snapshot is not None:
        wt_model = GPTNeoXForCausalLM.from_pretrained(
            str(wt_snapshot), local_files_only=True, dtype=torch.float32
        )
        endpoint_state = GPTNeoXAdapter.canonical_state(wt_model.state_dict())
    config = GPTNeoXConfig.from_pretrained(str(w0_snapshot), local_files_only=True)
    program, payloads, _ = load_program(program_path)
    comparison = evaluate_program(
        model_factory=lambda: GPTNeoXForCausalLM(config),
        base_state=w0,
        program=program,
        payloads=payloads,
        batches=causal_batches_from_jsonl(evaluation_jsonl),
        endpoint_state=endpoint_state,
        device=device,
        max_batches=max_batches,
    )
    atomic_write_json(output, comparison.to_dict())
    _echo_json(comparison.to_dict())


@app.command("train-compiler")
def train_compiler_command(
    corpus_path: Path = typer.Option(..., "--corpus"),
    config_path: Path = typer.Option(..., "--config"),
    output: Path = typer.Option(...),
    overwrite: bool = typer.Option(False),
    resume_from: Optional[Path] = typer.Option(None),
) -> None:
    value = load_yaml(config_path)
    compiler_config = CompilerConfig(**value.get("compiler", {}))
    training_config = TrainingConfig(**value.get("training", {}))
    summary = train_compiler(
        CompilerCorpus.load(corpus_path),
        output=output,
        compiler_config=compiler_config,
        training_config=training_config,
        overwrite=overwrite,
        resume_from=resume_from,
    )
    _echo_json(summary)


@app.command("compile")
def compile_program(
    compiler_path: Path = typer.Option(...),
    graph_path: Path = typer.Option(...),
    w0_path: Path = typer.Option(...),
    fingerprint_path: Path = typer.Option(...),
    recipe_path: Path = typer.Option(...),
    config_path: Path = typer.Option(...),
    output: Path = typer.Option(...),
) -> None:
    from safetensors.torch import load_file as load_safe

    value = load_yaml(config_path)
    config = CompilerConfig(**value.get("compiler", {}))
    graph = ArchitectureGraph.from_dict(load_json(graph_path))
    w0 = load_state(w0_path)
    fingerprint = FingerprintBundle.load(fingerprint_path)
    recipe = load_json(recipe_path)
    example = build_compiler_example(
        graph,
        w0,
        fingerprint,
        recipe,
        global_feature_dim=config.global_feature_dim,
        tensor_feature_dim=config.tensor_feature_dim,
        base_state_id=state_id(w0),
    )
    compiler = GenomeCompiler(config)
    compiler.load_state_dict(load_safe(str(compiler_path), device="cpu"), strict=True)
    program, payloads = compiler.generate_program(
        example, direct_fp16_delta_bytes=direct_fp16_delta_bytes(w0)
    )
    accounting = save_program(output, program, payloads)
    decoded = execute_program(w0, program, payloads)
    save_state(output / "candidate.safetensors", decoded)
    _echo_json({"accounting": accounting, "candidate_state_id": state_id(decoded)})


@app.command("seal-hidden")
def seal_hidden(
    run_id: str = typer.Option(...),
    compiler_path: Path = typer.Option(...),
    evidence_id: str = typer.Option(...),
    source_plan_id: str = typer.Option(...),
    program_manifest: Path = typer.Option(...),
    runtime_state: Path = typer.Option(...),
    candidate_count: int = typer.Option(1),
    selection_rule: str = typer.Option("one-shot-single-candidate"),
    output: Path = typer.Option(...),
) -> None:
    seal = build_prediction_seal(
        run_id=run_id,
        compiler_path=compiler_path,
        evidence_id=evidence_id,
        source_plan_id=source_plan_id,
        program_manifest=program_manifest,
        runtime_state=runtime_state,
        candidate_count=candidate_count,
        selection_rule=selection_rule,
    )
    seal.save(output)
    _echo_json({"output": str(output), "seal_id": seal.seal_id})


@app.command("reveal-hidden")
def reveal_hidden(
    plan_path: Path = typer.Option(..., "--plan"),
    run_id: str = typer.Option(...),
    prediction_seal: Path = typer.Option(...),
    workspace: Path = typer.Option(...),
    token: Optional[str] = typer.Option(None, envvar="HF_TOKEN"),
) -> None:
    _echo_json(
        reveal_hidden_endpoint(
            SourcePlan.load(plan_path),
            run_id=run_id,
            prediction_seal_path=prediction_seal,
            root=workspace,
            token=token,
        )
    )


if __name__ == "__main__":
    app()
