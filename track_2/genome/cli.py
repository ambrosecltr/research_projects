from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch
import typer

from .adapters.loader import load_adapter
from .architecture_graph import build_architecture_graph
from .bit_accounting import account_mgp
from .codecs import DenseDeltaCodec, LowRankSparseCodec, QuantizedDeltaCodec, SVDCodec
from .config import load_config, require, resolve_config_path
from .evaluator import GenomeGate, evaluate_model_state
from .fingerprint import GradientFingerprintConfig, build_gradient_fingerprint
from .hashing import sha256_file
from .io import ensure_dir, save_tensor_file, write_json
from .mgp.interpreter import decode_program
from .mgp.serializer import load_program, save_program
from .neural import (
    AutodecoderTrainingConfig,
    BlockDecoderConfig,
    fit_autodecoder,
    load_interpreter,
)
from .rate_distortion import RateDistortionPoint, run_rate_distortion
from .repair import LatentRefinementConfig, refine_neural_genome_codes
from .reporting import make_report
from .sensitivity import delta_energy_by_role, singular_summaries
from .specimen import freeze_specimen, load_specimen, verify_specimen_files
from .state import aggregate_statistics_by_role, compute_delta, delta_statistics
from .polypythia.cli import app as polypythia_app

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="GENOME Track 2 research CLI",
)
app.add_typer(polypythia_app, name="polypythia")


def _contract_metadata(specimen: Any, *, research_level: str = "G0") -> dict[str, Any]:
    metadata = {
        "research_level": research_level,
        "specimen_id": specimen.specimen_id,
        "architecture_manifest_sha256": specimen.manifest["contract_hashes"]["architecture"],
        "tensor_inventory_sha256": specimen.manifest["contract_hashes"]["tensor_inventory"],
        "tokenizer_sha256": specimen.manifest["contract_hashes"]["tokenizer"],
        "base_state_sha256": specimen.manifest["state_hashes"]["W0"],
        "conditioning_contract": {
            "base": "W0",
            "target_endpoint_seen_during_fit": research_level == "G0",
        },
    }
    # The target hash is audit metadata for G0 only. It is omitted from G1/G2 compiler inputs.
    if research_level == "G0":
        metadata["target_state_sha256_at_fit"] = specimen.manifest["state_hashes"]["WT"]
    return metadata


def _device_dtype(name: str) -> torch.dtype:
    mapping = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}
    if name not in mapping:
        raise typer.BadParameter(f"unsupported dtype: {name}")
    return mapping[name]


@app.command("track1-preflight")
def track1_preflight_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    endpoint: Path | None = typer.Option(
        None,
        help="Checkpoint to inspect; defaults to configured final or latest R0 checkpoint",
    ),
    output: Path | None = typer.Option(None, help="Optional JSON report path"),
    require_ready: bool = typer.Option(
        False,
        "--require-ready",
        help="Exit non-zero unless every R0 freeze contract passes",
    ),
) -> None:
    """Check exact Track 1 integration and report whether R0 is ready to freeze."""
    value = load_config(config)
    adapter = load_adapter(value)
    preflight = getattr(adapter, "preflight", None)
    if not callable(preflight):
        raise typer.BadParameter(
            f"adapter {adapter.adapter_id!r} does not expose Track 1 preflight checks"
        )
    selected = endpoint
    if selected is None:
        specimen_config = value.get("specimen", {})
        if isinstance(specimen_config, dict) and specimen_config.get("final_checkpoint"):
            selected = resolve_config_path(value, specimen_config["final_checkpoint"])
            if not selected.exists():
                selected = None
    result = preflight(selected)
    if output is not None:
        write_json(output, result)
    typer.echo(json.dumps(result, indent=2, default=str))
    if not result.get("ready_to_freeze", False):
        typer.echo("R0 is not ready to freeze yet; this is expected while the full run is active.")
        if require_ready:
            raise typer.Exit(code=2)


@app.command("freeze")
def freeze_command(config: Path = typer.Option(..., exists=True, readable=True)) -> None:
    """Freeze W0/WT and all Track 1 contracts into an immutable R0 specimen."""
    value = load_config(config)
    adapter = load_adapter(value)
    specimen_config = require(value, "specimen")
    if not isinstance(specimen_config, dict):
        raise typer.BadParameter("specimen configuration must be a mapping")
    base_checkpoint = specimen_config.get("base_checkpoint")
    specimen = freeze_specimen(
        adapter,
        output_dir=resolve_config_path(value, specimen_config["output"]),
        specimen_id=specimen_config["id"],
        final_checkpoint=resolve_config_path(value, specimen_config["final_checkpoint"]),
        base_checkpoint=(
            None if base_checkpoint is None else resolve_config_path(value, base_checkpoint)
        ),
        source_metadata={"config_hash": value["_config_hash"]},
    )
    typer.echo(f"Frozen specimen {specimen.specimen_id}: {specimen.root}")


@app.command("verify")
def verify_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    config: Path = typer.Option(..., exists=True, readable=True),
) -> None:
    """Verify specimen integrity and reproduce W0/WT functional metrics."""
    value = load_config(config)
    adapter = load_adapter(value)
    frozen = load_specimen(specimen)
    integrity = verify_specimen_files(frozen)
    evaluation = value.get("verification", value.get("evaluation", {}))
    kwargs = {
        "split": evaluation.get("split", "validation"),
        "max_batches": evaluation.get("max_batches"),
        "device": evaluation.get("device", "cpu"),
    }
    w0 = evaluate_model_state(adapter, frozen.load_base(), **kwargs)
    wt = evaluate_model_state(adapter, frozen.load_target(), **kwargs)
    result = {
        "integrity": integrity,
        "W0": w0,
        "WT": wt,
        "loss_improvement": w0["mean_loss"] - wt["mean_loss"],
    }
    write_json(frozen.root / "verification.json", result)
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("analyze")
def analyze_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(...),
    top_singular_values: int = typer.Option(32, min=1),
) -> None:
    """Compute Delta-T statistics, role energy, and matrix spectral summaries."""
    frozen = load_specimen(specimen)
    base = frozen.load_base()
    target = frozen.load_target()
    delta = compute_delta(base, target, frozen.inventory)
    rows = delta_statistics(base, target, frozen.inventory)
    result = {
        "specimen_id": frozen.specimen_id,
        "per_tensor": rows,
        "by_role": aggregate_statistics_by_role(rows),
        "delta_energy_by_role": delta_energy_by_role(base, target, frozen.inventory),
        "singular_summaries": singular_summaries(
            delta, frozen.inventory, top_k=top_singular_values
        ),
    }
    ensure_dir(output)
    write_json(output / "delta_analysis.json", result)
    typer.echo(f"Wrote {output / 'delta_analysis.json'}")


@app.command("encode")
def encode_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    codec: str = typer.Option(..., help="dense, int8, int4, svd, or svd_sparse"),
    output: Path = typer.Option(...),
    rank: int = typer.Option(8, min=0),
    sparse_fraction: float = typer.Option(0.001, min=0.0, max=1.0),
    factor_dtype: str = typer.Option("float32"),
    vector_bits: int = typer.Option(8),
    budget_bytes: int | None = typer.Option(None, min=0),
) -> None:
    """Fit a transparent G0 codec and write a frozen MGP artifact."""
    frozen = load_specimen(specimen)
    base = frozen.load_base()
    target = frozen.load_target()
    candidate_id = output.name.removesuffix(".mgp")
    if codec == "dense":
        implementation = DenseDeltaCodec(candidate_id=candidate_id)
        budget = None
    elif codec == "int8":
        implementation = QuantizedDeltaCodec(8, candidate_id=candidate_id)
        budget = None
    elif codec == "int4":
        implementation = QuantizedDeltaCodec(4, candidate_id=candidate_id)
        budget = None
    elif codec == "svd":
        implementation = SVDCodec(
            rank=None if budget_bytes is not None else rank,
            factor_dtype=_device_dtype(factor_dtype),
            vector_bits=vector_bits,
            candidate_id=candidate_id,
        )
        from .types import GenomeBudget

        budget = GenomeBudget(target_bytes=budget_bytes) if budget_bytes is not None else None
    elif codec == "svd_sparse":
        implementation = LowRankSparseCodec(
            rank=rank,
            sparse_fraction=sparse_fraction,
            factor_dtype=_device_dtype(factor_dtype),
            candidate_id=candidate_id,
        )
        budget = None
    else:
        raise typer.BadParameter(f"unknown codec: {codec}")
    program = implementation.fit(
        base,
        target,
        frozen.inventory,
        tied_groups=frozen.tied_groups,
        budget=budget,
        manifest_metadata=_contract_metadata(frozen),
    )
    result = save_program(program, output)
    typer.echo(json.dumps(result, indent=2))


@app.command("fit-neural")
def fit_neural_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(..., help="Output MGP directory"),
    interpreter_output: Path = typer.Option(..., help="Shared interpreter artifact directory"),
    updates: int = typer.Option(2000, min=1),
    batch_size: int = typer.Option(128, min=1),
    learning_rate: float = typer.Option(3e-4, min=1e-8),
    block_size: int = typer.Option(16, min=1),
    hidden_dim: int = typer.Option(256, min=8),
    device: str = typer.Option("cpu"),
) -> None:
    """Fit a G0 role-conditioned neural block genome directly to R0."""
    if output.exists():
        raise typer.BadParameter(f"output already exists: {output}")
    if interpreter_output.exists():
        raise typer.BadParameter(f"interpreter output already exists: {interpreter_output}")
    frozen = load_specimen(specimen)
    result = fit_autodecoder(
        frozen.load_base(),
        frozen.load_target(),
        frozen.inventory,
        tied_groups=frozen.tied_groups,
        interpreter_path=interpreter_output,
        candidate_id=output.name.removesuffix(".mgp"),
        decoder_config=BlockDecoderConfig(
            block_rows=block_size, block_cols=block_size, hidden_dim=hidden_dim
        ),
        training_config=AutodecoderTrainingConfig(
            updates=updates,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
            log_every=max(1, updates // 20),
        ),
        manifest_metadata=_contract_metadata(frozen),
    )
    artifact = save_program(result.program, output)
    write_json(output / "training_metrics.json", result.metrics)
    typer.echo(json.dumps({"mgp": artifact, "interpreter": result.interpreter_info}, indent=2))


@app.command("decode")
def decode_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    mgp: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(...),
    interpreter: Path | None = typer.Option(None, exists=True, file_okay=False),
) -> None:
    """Decode an MGP into a standalone safetensors candidate."""
    frozen = load_specimen(specimen)
    program = load_program(mgp)
    neural = load_interpreter(interpreter) if interpreter else None
    state = decode_program(
        program,
        frozen.load_base(),
        frozen.inventory,
        tied_groups=frozen.tied_groups,
        interpreter=neural,
        contract={
            "architecture_manifest_sha256": frozen.manifest["contract_hashes"]["architecture"],
            "tensor_inventory_sha256": frozen.manifest["contract_hashes"]["tensor_inventory"],
            "base_state_sha256": frozen.manifest["state_hashes"]["W0"],
        },
    )
    save_tensor_file(output, state)
    typer.echo(f"Decoded candidate to {output}")


@app.command("evaluate")
def evaluate_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    mgp: Path = typer.Option(..., exists=True, file_okay=False),
    config: Path = typer.Option(..., exists=True, readable=True),
    output: Path | None = typer.Option(None),
    interpreter: Path | None = typer.Option(None, exists=True, file_okay=False),
) -> None:
    """Run an MGP through the configured development/Genome Gate evaluator."""
    value = load_config(config)
    adapter = load_adapter(value)
    evaluation = value.get("evaluation", {})
    gate = GenomeGate(
        adapter,
        specimen,
        split=evaluation.get("split", "development"),
        max_batches=evaluation.get("max_batches"),
        device=evaluation.get("device", "cpu"),
    )
    neural = (
        load_interpreter(interpreter, device=evaluation.get("device", "cpu"))
        if interpreter
        else None
    )
    report = gate.evaluate_mgp(mgp, interpreter=neural)
    result = report.to_dict()
    result["bit_accounting"] = account_mgp(
        mgp,
        interpreter_path=interpreter,
        base_path=gate.specimen.base_path,
    )
    destination = output or (mgp / "evaluation.json")
    write_json(destination, result)
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("export-track1-checkpoint")
def export_track1_checkpoint_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    mgp: Path = typer.Option(..., exists=True, file_okay=False),
    config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    interpreter: Path | None = typer.Option(None, exists=True, file_okay=False),
) -> None:
    """Export a decoded phenotype for Track 1's existing generation/eval commands."""
    value = load_config(config)
    adapter = load_adapter(value)
    frozen = load_specimen(specimen)
    program = load_program(mgp)
    neural = (
        load_interpreter(interpreter, device="cpu")
        if interpreter is not None
        else None
    )
    state = decode_program(
        program,
        frozen.load_base(),
        frozen.inventory,
        tied_groups=frozen.tied_groups,
        interpreter=neural,
        contract={
            "architecture_manifest_sha256": frozen.manifest["contract_hashes"][
                "architecture"
            ],
            "tensor_inventory_sha256": frozen.manifest["contract_hashes"][
                "tensor_inventory"
            ],
            "base_state_sha256": frozen.manifest["state_hashes"]["W0"],
        },
    )
    template = frozen.manifest.get("source", {}).get("final_checkpoint")
    if not isinstance(template, str) or not template:
        raise typer.BadParameter("specimen does not record its Track 1 template checkpoint")
    result = adapter.export_evaluation_checkpoint(
        state,
        template_checkpoint=template,
        output=output,
        candidate_id=str(program.manifest["candidate_id"]),
        provenance={
            "specimen_id": frozen.specimen_id,
            "specimen_manifest_sha256": sha256_file(frozen.root / "manifest.json"),
            "mgp_manifest_sha256": sha256_file(mgp / "manifest.json"),
            "interpreter_path": None if interpreter is None else str(interpreter.resolve()),
        },
    )
    typer.echo(json.dumps(result, indent=2, default=str))


@app.command("architecture-graph")
def architecture_graph_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(...),
) -> None:
    """Export the tensor-role architecture graph used by future compilers."""
    frozen = load_specimen(specimen)
    graph = build_architecture_graph(frozen.inventory, frozen.tied_groups)
    write_json(output, graph.to_dict())
    typer.echo(f"Wrote architecture graph to {output}")


@app.command("fingerprint")
def fingerprint_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    split: str = typer.Option("fingerprint"),
    max_batches: int = typer.Option(32, min=1),
    sketch_dim_per_role: int = typer.Option(128, min=1),
) -> None:
    """Build a model-native dataset fingerprint from projected W0 gradients."""
    value = load_config(config)
    adapter = load_adapter(value)
    frozen = load_specimen(specimen)
    evaluation = value.get("evaluation", {})
    result = build_gradient_fingerprint(
        adapter,
        frozen.load_base(),
        frozen.inventory,
        config=GradientFingerprintConfig(
            split=split, max_batches=max_batches, sketch_dim_per_role=sketch_dim_per_role
        ),
        device=evaluation.get("device", "cpu"),
    )
    ensure_dir(output)
    tensors = {
        key: result.pop(key)
        for key in ["fingerprint_mean", "fingerprint_std", "fingerprint_min", "fingerprint_max"]
    }
    save_tensor_file(output / "fingerprint.safetensors", tensors)
    write_json(output / "manifest.json", result)
    typer.echo(f"Wrote dataset fingerprint to {output}")


@app.command("rate-distortion")
def rate_distortion_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    ranks: str = typer.Option("0,1,2,4,8,16,32"),
    include_sparse: bool = typer.Option(True),
) -> None:
    """Run the transparent G0 rate-distortion frontier under one evaluator."""
    value = load_config(config)
    adapter = load_adapter(value)
    frozen = load_specimen(specimen)
    evaluation = value.get("evaluation", {})
    gate = GenomeGate(
        adapter,
        frozen,
        split=evaluation.get("split", "development"),
        max_batches=evaluation.get("max_batches"),
        device=evaluation.get("device", "cpu"),
    )
    parsed_ranks = [int(item.strip()) for item in ranks.split(",") if item.strip()]
    points = [
        RateDistortionPoint("quantized", "int8", bits=8),
        RateDistortionPoint("quantized", "int4", bits=4),
        *[RateDistortionPoint("svd", f"svd_r{rank}", rank=rank) for rank in parsed_ranks],
    ]
    if include_sparse:
        points.extend(
            RateDistortionPoint(
                "svd_sparse", f"svd_r{rank}_sp1e3", rank=rank, sparse_fraction=0.001
            )
            for rank in parsed_ranks
            if rank > 0
        )
    results = run_rate_distortion(
        frozen,
        gate,
        output_dir=output,
        points=points,
        manifest_metadata=_contract_metadata(frozen),
    )
    typer.echo(f"Completed {len(results)} rate-distortion points in {output}")


@app.command("refine-latent")
def refine_latent_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    mgp: Path = typer.Option(..., exists=True, file_okay=False),
    interpreter: Path = typer.Option(..., exists=True, file_okay=False),
    config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    steps: int = typer.Option(100, min=1),
    learning_rate: float = typer.Option(1e-3, min=1e-8),
    split: str = typer.Option("probe"),
) -> None:
    """Freeze the interpreter and repair a candidate by optimizing genome codes only."""
    value = load_config(config)
    adapter = load_adapter(value)
    frozen = load_specimen(specimen)
    device = value.get("evaluation", {}).get("device", "cpu")
    result = refine_neural_genome_codes(
        adapter,
        load_program(mgp),
        load_interpreter(interpreter, device=device),
        frozen.load_base(device=device),
        frozen.inventory,
        tied_groups=frozen.tied_groups,
        config=LatentRefinementConfig(
            steps=steps, learning_rate=learning_rate, split=split, device=device
        ),
    )
    save_program(result.program, output)
    write_json(output / "refinement_metrics.json", result.metrics)
    typer.echo(f"Wrote refined genome to {output}")


@app.command("report")
def report_command(
    input_root: list[Path] = typer.Argument(..., exists=True),
    output: Path = typer.Option(...),
    title: str = typer.Option("GENOME experiment report"),
) -> None:
    """Build Markdown/CSV/JSON summaries from immutable evaluation reports."""
    paths = make_report(input_root, output_dir=output, title=title)
    typer.echo(json.dumps({key: str(value) for key, value in paths.items()}, indent=2))


@app.command("demo")
def demo_command(
    output: Path = typer.Option(Path("artifacts/demo")),
    updates: int = typer.Option(80, min=1, help="Tiny Track 1 training updates"),
    neural: bool = typer.Option(False, help="Also fit a small neural genome"),
    force: bool = typer.Option(False, help="Delete an existing demo directory"),
) -> None:
    """Run the complete G0 pipeline on a deterministic tiny causal language model."""
    from examples.tiny_track1 import TinyTrack1Adapter, train_reference

    if output.exists():
        if not force:
            raise typer.BadParameter(f"output already exists: {output}; pass --force")
        shutil.rmtree(output)
    ensure_dir(output / "source")
    adapter = TinyTrack1Adapter()
    training = train_reference(output / "source" / "R0.pt", adapter=adapter, updates=updates)
    specimen = freeze_specimen(
        adapter,
        output_dir=output / "specimen",
        specimen_id="tiny_R0",
        final_checkpoint=output / "source" / "R0.pt",
        source_metadata={"demo": True, "training": training},
    )
    base = specimen.load_base()
    target = specimen.load_target()
    metadata = _contract_metadata(specimen)
    genome_dir = ensure_dir(output / "genomes")
    codecs = [
        DenseDeltaCodec(candidate_id="tiny_dense"),
        QuantizedDeltaCodec(8, candidate_id="tiny_int8"),
        QuantizedDeltaCodec(4, candidate_id="tiny_int4"),
        SVDCodec(rank=4, factor_dtype=torch.float32, candidate_id="tiny_svd_r4"),
        LowRankSparseCodec(
            rank=4, sparse_fraction=0.01, factor_dtype=torch.float32, candidate_id="tiny_svd_sparse"
        ),
    ]
    reports = []
    gate = GenomeGate(adapter, specimen, split="development", max_batches=4)
    for codec in codecs:
        program = codec.fit(
            base,
            target,
            specimen.inventory,
            tied_groups=specimen.tied_groups,
            manifest_metadata=metadata,
        )
        path = genome_dir / f"{program.manifest['candidate_id']}.mgp"
        save_program(program, path)
        report = gate.evaluate_mgp(path).to_dict()
        report["bit_accounting"] = account_mgp(path, base_path=specimen.base_path)
        write_json(path / "evaluation.json", report)
        reports.append(report)

    if neural:
        interpreter_path = output / "interpreters" / "tiny_block_v0"
        result = fit_autodecoder(
            base,
            target,
            specimen.inventory,
            tied_groups=specimen.tied_groups,
            interpreter_path=interpreter_path,
            candidate_id="tiny_neural",
            decoder_config=BlockDecoderConfig(
                block_rows=8,
                block_cols=8,
                global_code_dim=32,
                layer_code_dim=16,
                tensor_code_dim=16,
                hidden_dim=128,
                depth=3,
            ),
            training_config=AutodecoderTrainingConfig(
                updates=400,
                batch_size=64,
                learning_rate=1e-3,
                device="cpu",
                log_every=40,
            ),
            manifest_metadata=metadata,
        )
        path = genome_dir / "tiny_neural.mgp"
        save_program(result.program, path)
        neural_interpreter = load_interpreter(interpreter_path)
        report = gate.evaluate_mgp(path, interpreter=neural_interpreter).to_dict()
        report["bit_accounting"] = account_mgp(
            path, interpreter_path=interpreter_path, base_path=specimen.base_path
        )
        write_json(path / "evaluation.json", report)
        write_json(path / "training_metrics.json", result.metrics)
        reports.append(report)

    summary = {
        "training": training,
        "specimen": str(specimen.root),
        "reports": reports,
    }
    write_json(output / "summary.json", summary)
    lines = [
        "# GENOME tiny end-to-end demo",
        "",
        f"Track 1 training updates: **{updates}**",
        f"Final tiny training loss: **{training['final_train_loss']:.6f}**",
        "",
        "| Candidate | Artifact bytes | Loss gap | Relative parameter L2 | Decision |",
        "|---|---:|---:|---:|---|",
    ]
    for report in reports:
        row_template = (
            "| {candidate_id} | {bytes:,} | {loss_gap:.6f} | "
            "{relative_l2:.6f} | {decision} |"
        )
        lines.append(
            row_template.format(
                candidate_id=report["candidate_id"],
                bytes=report["bit_accounting"]["artifact_bytes"],
                loss_gap=report["functional_metrics"]["loss_gap"],
                relative_l2=report["parameter_metrics"]["relative_l2"],
                decision=report["decision"],
            )
        )
    (output / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    typer.echo(f"Demo complete: {output / 'SUMMARY.md'}")


if __name__ == "__main__":
    app()
