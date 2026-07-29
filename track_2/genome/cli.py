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
from .compact_targets import (
    CompactTargetConfig,
    fit_compact_svd_target,
    serialize_and_audit_compiler_target,
)
from .config import load_config, require, resolve_config_path
from .evaluator import GenomeGate, evaluate_model_state
from .fingerprint import GradientFingerprintConfig, build_gradient_fingerprint
from .hashing import sha256_file
from .io import ensure_dir, read_json, save_tensor_file, write_json
from .life_schema import ModelLifeManifest
from .mgp.interpreter import decode_program
from .mgp.serializer import load_program, save_program
from .polypythia.cli import app as polypythia_app
from .program_tokens import ProgramTokenizationConfig, program_to_sequence, sequence_to_program
from .rate_distortion import RateDistortionPoint, run_rate_distortion
from .reporting import make_report
from .source_audit import SourceAuditManifest
from .sensitivity import delta_energy_by_role, singular_summaries
from .specimen import freeze_specimen, load_specimen, verify_specimen_files
from .state import aggregate_statistics_by_role, compute_delta, delta_statistics

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="GENOME Track 2 research CLI",
)
app.add_typer(polypythia_app, name="polypythia")


@app.command("validate-life")
def validate_life_command(
    manifest: Path = typer.Option(..., exists=True, readable=True),
) -> None:
    """Validate a complete model-life record and its endpoint-free compiler view."""
    value = read_json(manifest)
    if not isinstance(value, dict):
        raise typer.BadParameter("model-life manifest must be a JSON object")
    life = ModelLifeManifest.from_dict(value)
    typer.echo(
        json.dumps(
            {
                "valid": True,
                "run_id": life.run_id,
                "split": life.split,
                "completeness": life.completeness,
                "content_sha256": life.content_sha256,
                "compiler_view": life.compiler_view(),
            },
            indent=2,
        )
    )


@app.command("audit-source")
def audit_source_command(
    manifest: Path = typer.Option(..., exists=True, readable=True),
) -> None:
    """Validate a public source audit, split plan, and labelled storage estimates."""
    value = read_json(manifest)
    if not isinstance(value, dict):
        raise typer.BadParameter("source-audit manifest must be a JSON object")
    audit = SourceAuditManifest.from_dict(value)
    typer.echo(
        json.dumps(
            {
                "valid": True,
                "content_sha256": audit.content_sha256,
                "estimated_active_endpoint_pair_bytes": (
                    audit.estimated_approved_endpoint_pair_bytes()
                ),
                "estimated_maximum_catalog_bytes": audit.estimated_maximum_catalog_bytes(),
            },
            indent=2,
        )
    )


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
    """Check legacy G0 or future evaluation-only Track 1 integration."""
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
    """Freeze an evaluation specimen; this does not add it to compiler training."""
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


@app.command("fit-compact-target")
def fit_compact_target_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(...),
    target_fraction: float = typer.Option(0.10, min=0.0001, max=0.10),
    max_rank: int = typer.Option(64, min=0),
) -> None:
    """Fit, serialize, and byte-audit a transparent compact target candidate."""
    frozen = load_specimen(specimen)
    result = fit_compact_svd_target(
        frozen.load_base(),
        frozen.load_target(),
        frozen.inventory,
        tied_groups=frozen.tied_groups,
        config=CompactTargetConfig(
            target_fraction_of_fp16_delta=target_fraction,
            max_rank=max_rank,
        ),
        candidate_id=output.name.removesuffix(".mgp"),
        manifest_metadata=_contract_metadata(frozen),
    )
    serialized = serialize_and_audit_compiler_target(
        result,
        frozen.inventory,
        output,
    )
    typer.echo(json.dumps(serialized.audit.to_dict(), indent=2))
    if not serialized.audit.serialized_policy_ready:
        raise typer.Exit(code=2)


@app.command("audit-program-tokens")
def audit_program_tokens_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    mgp: Path = typer.Option(..., exists=True, file_okay=False),
    coefficient_chunk_dim: int = typer.Option(16, min=1),
) -> None:
    """Tokenize and parse a compact program with the deterministic inverse."""
    frozen = load_specimen(specimen)
    program = load_program(mgp)
    config = ProgramTokenizationConfig(coefficient_chunk_dim=coefficient_chunk_dim)
    sequence = program_to_sequence(program, frozen.inventory, config=config)
    restored = sequence_to_program(
        sequence,
        frozen.inventory,
        tied_groups=frozen.tied_groups,
        config=config,
        candidate_id="token-roundtrip-audit",
    )
    typer.echo(
        json.dumps(
            {
                "valid": True,
                "token_count": int(sequence.token_ids.numel()),
                "numeric_token_count": int(sequence.numeric_mask.sum().item()),
                "restored_payload_tensors": sorted(restored.payload_tensors),
            },
            indent=2,
        )
    )


@app.command("decode")
def decode_command(
    specimen: Path = typer.Option(..., exists=True, file_okay=False),
    mgp: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(...),
) -> None:
    """Decode an MGP with the deterministic Runtime."""
    frozen = load_specimen(specimen)
    program = load_program(mgp)
    state = decode_program(
        program,
        frozen.load_base(),
        frozen.inventory,
        tied_groups=frozen.tied_groups,
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
) -> None:
    """Run a deterministic MGP through the configured Genome Gate evaluator."""
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
    report = gate.evaluate_mgp(mgp)
    result = report.to_dict()
    result["bit_accounting"] = account_mgp(
        mgp,
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
) -> None:
    """Export a deterministic candidate for legacy G0 or future Track 1 evaluation."""
    value = load_config(config)
    adapter = load_adapter(value)
    frozen = load_specimen(specimen)
    program = load_program(mgp)
    state = decode_program(
        program,
        frozen.load_base(),
        frozen.inventory,
        tied_groups=frozen.tied_groups,
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
    force: bool = typer.Option(False, help="Delete an existing demo directory"),
) -> None:
    """Run the deterministic Runtime and codec pipeline on a tiny test fixture."""
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
