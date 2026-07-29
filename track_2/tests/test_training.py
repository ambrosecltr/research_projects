from __future__ import annotations

from pathlib import Path

import torch
from safetensors.torch import save_file

from genome.architecture import graph_from_state
from genome.compiler import (
    CompilerConfig,
    CompilerCorpus,
    CompilerRecord,
    TrainingConfig,
    build_compiler_corpus,
    train_compiler,
)
from genome.fingerprint import corpus_fingerprint
from genome.hashing import sha256_file, sha256_json
from genome.io import atomic_write_json, load_json
from genome.mgp import FitConfig, fit_low_rank_program, save_program
from genome.protocol import ArtifactBinding
from genome.sources import default_pythia_v1_plan
from genome.state import state_id


def _accepted_program(
    *,
    root: Path,
    run_id: str,
    program,
    payloads,
    w0: dict[str, torch.Tensor],
    wt: dict[str, torch.Tensor],
    w0_path: Path,
    wt_path: Path,
    evaluation_jsonl: Path,
    formula_id: str,
    source_plan_id: str,
    accepted: bool = True,
) -> tuple[Path, Path]:
    root.mkdir(parents=True)
    save_program(root, program, payloads)
    manifest = load_json(root / "manifest.json")
    binding = ArtifactBinding(
        run_id=run_id,
        formula_id=formula_id,
        program_id=manifest["program_id"],
        program_manifest_sha256=sha256_file(root / "manifest.json"),
        payload_sha256=manifest["payload_sha256"],
        w0_state_id=state_id(w0),
        wt_state_id=state_id(wt),
        evaluation_jsonl_sha256=sha256_file(evaluation_jsonl),
        source_plan_id=source_plan_id,
        code_commit="a" * 40,
    )
    evaluation_path = root / "evaluation.json"
    atomic_write_json(
        evaluation_path,
        {
            "format": "GENOME_TARGET_EVALUATION",
            "binding": binding.to_dict(),
            "comparison": {},
        },
    )
    atomic_write_json(
        root / "acceptance.json",
        {
            "format": "GENOME_TARGET_ACCEPTANCE",
            "accepted": accepted,
            "binding": binding.to_dict(),
            "evaluation_report_sha256": sha256_file(evaluation_path),
        },
    )
    return root, evaluation_path


def test_compiler_training_smoke(tmp_path: Path) -> None:
    torch.manual_seed(11)
    w0 = {
        "layers.0.weight": torch.randn(16, 12),
        "layers.0.bias": torch.randn(16),
        "layers.1.weight": torch.randn(12, 16),
        "final_norm.weight": torch.ones(12),
    }
    wt = {name: value + torch.randn_like(value) * 0.01 for name, value in w0.items()}
    graph = graph_from_state(w0, family="toy", config={"hidden": 12})
    graph_path = tmp_path / "graph.json"
    atomic_write_json(graph_path, graph.to_dict())
    w0_path = tmp_path / "w0.safetensors"
    save_file(w0, str(w0_path))
    wt_path = tmp_path / "wt.safetensors"
    save_file(wt, str(wt_path))
    fingerprint_path = tmp_path / "fingerprint"
    corpus_fingerprint([[1, 2, 3], [4, 5]]).save(fingerprint_path)
    recipe_path = tmp_path / "recipe.json"
    atomic_write_json(recipe_path, {"optimizer": {"lr": 0.001}, "tokens": 1000})
    program, payloads = fit_low_rank_program(
        w0,
        wt,
        graph,
        config=FitConfig(budget_fraction=0.5, max_rank=2, svd_method="exact"),
    )
    evaluation_jsonl = tmp_path / "evaluation.jsonl"
    evaluation_jsonl.write_text('{"input_ids":[1,2]}\n')
    formula_id = "b" * 64
    source_plan_id = "c" * 64
    records = []
    for run_id, split in (("train", "training"), ("dev", "development")):
        program_path, evaluation_path = _accepted_program(
            root=tmp_path / f"program-{run_id}",
            run_id=run_id,
            program=program,
            payloads=payloads,
            w0=w0,
            wt=wt,
            w0_path=w0_path,
            wt_path=wt_path,
            evaluation_jsonl=evaluation_jsonl,
            formula_id=formula_id,
            source_plan_id=source_plan_id,
        )
        records.append(
            CompilerRecord(
                run_id=run_id,
                split=split,
                graph_path=str(graph_path),
                w0_path=str(w0_path),
                wt_path=str(wt_path),
                fingerprint_path=str(fingerprint_path),
                recipe_path=str(recipe_path),
                program_path=str(program_path),
                evaluation_report_path=str(evaluation_path),
                evaluation_jsonl=str(evaluation_jsonl),
            )
        )
    corpus = CompilerCorpus(
        records=tuple(records),
        formula_id=formula_id,
        source_plan_id=source_plan_id,
    )
    summary = train_compiler(
        corpus,
        output=tmp_path / "run",
        compiler_config=CompilerConfig(
            global_feature_dim=16,
            tensor_feature_dim=12,
            coordinate_feature_dim=4,
            d_model=16,
            n_heads=4,
            transformer_layers=1,
            message_layers=1,
            max_rank=2,
            target_fraction=0.5,
            manifest_reserve_bytes=0,
        ),
        training_config=TrainingConfig(
            epochs=1,
            device="cpu",
            checkpoint_every=100,
            development_every=1,
            functional_weight=0.0,
        ),
        free_running_evaluator=lambda *args, **kwargs: {"mean_endpoint_progress": 0.5},
    )
    assert summary["steps"] == 1
    assert (tmp_path / "run" / "best-compiler.safetensors").is_file()
    assert (tmp_path / "run" / "summary.json").is_file()


def test_compiler_corpus_builder_includes_14m_seed9_and_excludes_hidden(
    tmp_path: Path,
) -> None:
    plan = default_pythia_v1_plan()
    program_root = tmp_path / "accepted"
    workspace = tmp_path / "workspace"
    evaluation_jsonl = tmp_path / "evaluation.jsonl"
    evaluation_jsonl.write_text('{"input_ids":[1,2]}\n')
    formula_value = {
        "fit": {"budget_fraction": 0.10},
        "refinement": {},
        "data": {
            "refinement": str(evaluation_jsonl),
            "formula_tuning": str(evaluation_jsonl),
            "formula_tuning_batches": 16,
            "formula_sample_receipt": str(tmp_path / "formula-receipt.json"),
            "development_verifier": str(evaluation_jsonl),
            "development_verifier_receipt": str(tmp_path / "verifier-receipt.json"),
            "development_evaluation_batches": 128,
        },
        "acceptance": {
            "maximum_target_fraction": 0.10,
            "minimum_endpoint_progress": 0.80,
        },
        "corpus": {
            "expected_training_records": 16,
            "expected_development_records": 2,
            "rejected_training_lives": ["pythia-14m-seed5"],
        },
        "endpoint_semantics": "same-corpus functional endpoint",
    }
    formula_id = sha256_json(
        {
            key: formula_value[key]
            for key in ("fit", "refinement", "data", "acceptance", "endpoint_semantics")
        }
    )
    formula_path = tmp_path / "formula.json"
    atomic_write_json(
        formula_path,
        {
            **formula_value,
            "formula_id": formula_id,
            "status": "frozen",
        },
    )
    w0 = {"weight": torch.zeros(16)}
    wt = {"weight": torch.ones(16)}
    graph = graph_from_state(w0, family="toy", config={})
    fitted, payloads = fit_low_rank_program(
        w0,
        wt,
        graph,
        config=FitConfig(budget_fraction=0.9, max_rank=1, svd_method="exact"),
    )
    for life in plan.lives:
        if life.split == "hidden":
            continue
        canonical = workspace / "canonical" / "lives" / life.run_id
        canonical.mkdir(parents=True)
        w0_path = canonical / "w0.safetensors"
        wt_path = canonical / "wt.safetensors"
        save_file(w0, str(w0_path))
        save_file(wt, str(wt_path))
        _accepted_program(
            root=program_root / life.run_id,
            run_id=life.run_id,
            program=fitted,
            payloads=payloads,
            w0=w0,
            wt=wt,
            w0_path=w0_path,
            wt_path=wt_path,
            evaluation_jsonl=evaluation_jsonl,
            formula_id=formula_id,
            source_plan_id=plan.plan_id,
            accepted=life.run_id != "pythia-14m-seed5",
        )
    corpus = build_compiler_corpus(
        plan,
        workspace=workspace,
        program_root=program_root,
        formula_path=formula_path,
    )
    by_id = {record.run_id: record for record in corpus.records}
    assert len(corpus.records) == 18
    assert by_id["pythia-14m-seed9"].split == "training"
    assert "pythia-14m-seed5" not in by_id
    assert "pythia-31m-seed9" not in by_id


def test_compiler_training_resume(tmp_path: Path) -> None:
    torch.manual_seed(19)
    w0 = {
        "layers.0.weight": torch.randn(12, 8),
        "layers.0.bias": torch.randn(12),
        "layers.1.weight": torch.randn(8, 12),
        "final_norm.weight": torch.ones(8),
    }
    wt = {name: value + torch.randn_like(value) * 0.01 for name, value in w0.items()}
    graph = graph_from_state(w0, family="toy", config={"hidden": 8})
    graph_path = tmp_path / "graph.json"
    atomic_write_json(graph_path, graph.to_dict())
    w0_path = tmp_path / "w0.safetensors"
    save_file(w0, str(w0_path))
    wt_path = tmp_path / "wt.safetensors"
    save_file(wt, str(wt_path))
    fingerprint_path = tmp_path / "fingerprint"
    corpus_fingerprint([[1, 2, 3]]).save(fingerprint_path)
    recipe_path = tmp_path / "recipe.json"
    atomic_write_json(recipe_path, {"tokens": 100})
    program, payloads = fit_low_rank_program(
        w0, wt, graph, config=FitConfig(budget_fraction=0.5, max_rank=2, svd_method="exact")
    )
    evaluation_jsonl = tmp_path / "evaluation.jsonl"
    evaluation_jsonl.write_text('{"input_ids":[1,2]}\n')
    formula_id = "d" * 64
    source_plan_id = "e" * 64
    records = []
    for run_id, split in (("train", "training"), ("dev", "development")):
        program_path, evaluation_path = _accepted_program(
            root=tmp_path / f"program-{run_id}",
            run_id=run_id,
            program=program,
            payloads=payloads,
            w0=w0,
            wt=wt,
            w0_path=w0_path,
            wt_path=wt_path,
            evaluation_jsonl=evaluation_jsonl,
            formula_id=formula_id,
            source_plan_id=source_plan_id,
        )
        records.append(
            CompilerRecord(
                run_id=run_id,
                split=split,
                graph_path=str(graph_path),
                w0_path=str(w0_path),
                wt_path=str(wt_path),
                fingerprint_path=str(fingerprint_path),
                recipe_path=str(recipe_path),
                program_path=str(program_path),
                evaluation_report_path=str(evaluation_path),
                evaluation_jsonl=str(evaluation_jsonl),
            )
        )
    corpus = CompilerCorpus(
        records=tuple(records),
        formula_id=formula_id,
        source_plan_id=source_plan_id,
    )
    compiler_config = CompilerConfig(
        global_feature_dim=16,
        tensor_feature_dim=12,
        coordinate_feature_dim=4,
        d_model=16,
        n_heads=4,
        transformer_layers=1,
        message_layers=1,
        max_rank=2,
        target_fraction=0.5,
        manifest_reserve_bytes=0,
    )
    first_config = TrainingConfig(
        epochs=1,
        device="cpu",
        checkpoint_every=100,
        development_every=1,
        functional_weight=0.0,
    )
    run = tmp_path / "run"
    train_compiler(
        corpus,
        output=run,
        compiler_config=compiler_config,
        training_config=first_config,
        free_running_evaluator=lambda *args, **kwargs: {"mean_endpoint_progress": 0.5},
    )
    resume = run / "checkpoints" / "final-step-00000001"
    summary = train_compiler(
        corpus,
        output=run,
        compiler_config=compiler_config,
        training_config=TrainingConfig(**{**first_config.__dict__, "epochs": 2}),
        resume_from=resume,
        free_running_evaluator=lambda *args, **kwargs: {"mean_endpoint_progress": 0.6},
    )
    assert summary["steps"] == 2
    assert (run / "checkpoints" / "final-step-00000002").is_dir()
