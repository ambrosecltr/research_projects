from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from genome.architecture import graph_from_state
from genome.compiler import CompilerConfig, CompilerCorpus, CompilerRecord, TrainingConfig, train_compiler
from genome.fingerprint import corpus_fingerprint
from genome.io import atomic_write_json
from genome.mgp import FitConfig, fit_low_rank_program, save_program


def test_compiler_training_smoke(tmp_path: Path) -> None:
    torch.manual_seed(11)
    w0 = {
        "layers.0.weight": torch.randn(16, 12),
        "layers.0.bias": torch.randn(16),
        "layers.1.weight": torch.randn(12, 16),
        "final_norm.weight": torch.ones(12),
    }
    wt = {
        name: value + torch.randn_like(value) * 0.01
        for name, value in w0.items()
    }
    graph = graph_from_state(w0, family="toy", config={"hidden": 12})
    graph_path = tmp_path / "graph.json"
    atomic_write_json(graph_path, graph.to_dict())
    w0_path = tmp_path / "w0.safetensors"
    save_file(w0, str(w0_path))
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
    program_path = tmp_path / "program"
    save_program(program_path, program, payloads)
    atomic_write_json(program_path / "acceptance.json", {"accepted": True})
    records = tuple(
        CompilerRecord(
            run_id=run_id,
            split=split,
            graph_path=str(graph_path),
            w0_path=str(w0_path),
            fingerprint_path=str(fingerprint_path),
            recipe_path=str(recipe_path),
            program_path=str(program_path),
        )
        for run_id, split in (("train", "training"), ("dev", "development"))
    )
    summary = train_compiler(
        CompilerCorpus(records=records),
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
    )
    assert summary["steps"] == 1
    assert (tmp_path / "run" / "best-compiler.safetensors").is_file()
    assert (tmp_path / "run" / "summary.json").is_file()



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
    fingerprint_path = tmp_path / "fingerprint"
    corpus_fingerprint([[1, 2, 3]]).save(fingerprint_path)
    recipe_path = tmp_path / "recipe.json"
    atomic_write_json(recipe_path, {"tokens": 100})
    program, payloads = fit_low_rank_program(
        w0, wt, graph, config=FitConfig(budget_fraction=0.5, max_rank=2, svd_method="exact")
    )
    program_path = tmp_path / "program"
    save_program(program_path, program, payloads)
    atomic_write_json(program_path / "acceptance.json", {"accepted": True})
    records = tuple(
        CompilerRecord(
            run_id=run_id,
            split=split,
            graph_path=str(graph_path),
            w0_path=str(w0_path),
            fingerprint_path=str(fingerprint_path),
            recipe_path=str(recipe_path),
            program_path=str(program_path),
        )
        for run_id, split in (("train", "training"), ("dev", "development"))
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
        CompilerCorpus(records=records),
        output=run,
        compiler_config=compiler_config,
        training_config=first_config,
    )
    resume = run / "checkpoints" / "final-step-00000001"
    summary = train_compiler(
        CompilerCorpus(records=records),
        output=run,
        compiler_config=compiler_config,
        training_config=TrainingConfig(**{**first_config.__dict__, "epochs": 2}),
        resume_from=resume,
    )
    assert summary["steps"] == 2
    assert (run / "checkpoints" / "final-step-00000002").is_dir()
