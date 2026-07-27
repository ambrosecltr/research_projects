"""A real tiny synthetic run exercises the public executable workflow end to end."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pytest
import torch

from poetry50m.cli import main


def _json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _jsonl(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
    )


def _invoke(*arguments: str) -> None:
    assert main(arguments) == 0


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(),
                reason="MPS is unavailable",
            ),
        ),
    ],
)
def test_tiny_synthetic_cli_workflow(tmp_path: Path, device: str) -> None:
    corpus, prompts, thoughts, pairings = [], [], [], []
    for index in range(40):
        document_id = f"synthetic-{index:03d}"
        block_id = f"{document_id}:poem:0"
        text = f"moon {index} remembers the river\nquiet bells carry amber wind"
        corpus.append(
            {
                "document_id": document_id,
                "provenance": {
                    "work": f"Synthetic work {index}",
                    "author": "integration test",
                    "licence": "synthetic-test-only",
                    "source": "generated in test",
                    "rights_status": "synthetic",
                },
                "text": text,
                "raw_text": text,
                "source_path": "",
                "metadata": {},
                "transformation_lineage": ["test_fixture"],
                "blocks": [
                    {
                        "block_id": block_id,
                        "kind": "poem",
                        "text": text,
                        "poem_id": f"poem-{index}",
                        "start_char": 0,
                        "end_char": len(text),
                        "metadata": {},
                    }
                ],
            }
        )
        prompts.append(
            {
                "prompt_id": f"prompt-{index}",
                "document_id": document_id,
                "prompt": f"A moonlit river {index}",
                "method": "imagery",
                "source_attribution": "integration fixture",
            }
        )
        thoughts.append(
            {
                "thought_id": f"thought-{index}",
                "document_id": document_id,
                "text": f"Use quiet image {index}.",
                "method": "editorial",
                "source_attribution": "integration fixture",
            }
        )
        if index == 0:
            pairings.append(
                {
                    "pairing_id": "pair-cross-document",
                    "target_document_id": document_id,
                    "target_block_id": block_id,
                    "prompt_id": "prompt-1",
                    "thought_id": "thought-1",
                    "transformation_lineage": ["fixture_pairing"],
                }
            )
    _jsonl(tmp_path / "corpus.jsonl", corpus)
    _jsonl(tmp_path / "prompts.jsonl", prompts)
    _jsonl(tmp_path / "thoughts.jsonl", thoughts)
    _jsonl(tmp_path / "pairings.jsonl", pairings)
    _json(
        tmp_path / "data.json",
        {
            "format_version": 1,
            "manifest_format": "jsonl",
            "manifest_schema": "SourceDocument",
            "split": {"salt": "integration", "train": 0.8, "validation": 0.1, "test": 0.1},
            "tokenizer": {
                "vocab_size": 300,
                "min_frequency": 1,
                "special_tokens": [
                    "<|pad|>",
                    "<|bos|>",
                    "<|eos|>",
                    "<|prompt|>",
                    "<|thought|>",
                    "<|poem|>",
                    "<|mask|>",
                ],
            },
            "packing": {"sequence_length": 64},
            "objectives": {"conditional_poetry": 1.0, "auxiliary_prose_ntp": 0.0},
            "rights": {"allow_synthetic": True},
        },
    )
    _json(
        tmp_path / "model.json",
        {
            "architecture": "gpt",
            "vocab_size": 300,
            "max_seq_len": 64,
            "d_model": 16,
            "n_layers": 1,
            "n_heads": 2,
            "ffn_dim": 32,
            "dropout": 0.0,
            "rope_base": 10000.0,
            "rope_fraction": 1.0,
            "norm_epsilon": 1e-6,
            "linear_bias": False,
            "tie_embeddings": True,
            "ignore_index": -100,
        },
    )
    _json(
        tmp_path / "train.json",
        {
            "max_steps": 8,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "warmup_steps": 0,
            "device": device,
            "precision": "none",
            "seed": 7,
            "deterministic": True,
            "log_every_steps": 1,
            "checkpoint_every_steps": 0,
            "trajectory_every_steps": 0,
            "trajectory_capture_steps": [2],
            "analysis_every_steps": 0,
        },
    )
    _json(
        tmp_path / "trajectory.json",
        {
            "linear": {"max_extrapolation_ratio": 2.0},
            "low_rank": {
                "max_rank": 2,
                "energy_threshold": 0.9,
                "polynomial_degree": 1,
                "max_extrapolation_ratio": 2.0,
                "eigenvalue_relative_floor": 1e-8,
            },
            "safety": {
                "max_relative_tensor_norm": 100.0,
                "min_relative_tensor_norm": 0.0,
                "near_zero_norm": 1e-9,
            },
            "gates": {
                "max_verification_loss_increase": 100.0,
                "max_post_leap_loss_increase": 100.0,
                "max_anchor_mse": 100.0,
                "max_anchor_cosine_distance": 100.0,
                "max_anchor_symmetric_kl": 100.0,
            },
        },
    )
    _json(
        tmp_path / "suite.json",
        {
            "suite_id": "tiny-integration",
            "version": 1,
            "cases": [
                {"case_id": "river", "prompt": "A moonlit river", "keywords": ["moon", "river"]},
                {"case_id": "bells", "prompt": "Quiet amber bells", "keywords": ["bells", "amber"]},
            ],
        },
    )
    _json(
        tmp_path / "run-policy.json",
        {
            "format_version": 1,
            "trajectory_config_sha256": hashlib.sha256(
                (tmp_path / "trajectory.json").read_bytes()
            ).hexdigest(),
            "verification": {
                "fixed_heldout_batches": 2,
                "anchor_positions_per_batch": 2,
                "fixed_probe_batches": 1,
                "probe_steps": 1,
                "optimizer_policy": "retain",
            },
        },
    )
    prepared, r0, r1, r2 = (
        tmp_path / "prepared",
        tmp_path / "r0",
        tmp_path / "r1",
        tmp_path / "r2",
    )
    _invoke(
        "prepare",
        "--corpus-manifest",
        str(tmp_path / "corpus.jsonl"),
        "--prompts",
        str(tmp_path / "prompts.jsonl"),
        "--thoughts",
        str(tmp_path / "thoughts.jsonl"),
        "--pairings",
        str(tmp_path / "pairings.jsonl"),
        "--config",
        str(tmp_path / "data.json"),
        "--output",
        str(prepared),
    )
    common = (
        "--prepared",
        str(prepared),
        "--model-config",
        str(tmp_path / "model.json"),
        "--train-config",
        str(tmp_path / "train.json"),
        "--batch-size",
        "2",
    )
    r0_base = (*common, "--run-dir", str(r0))
    target_policy = ("--run-policy", str(tmp_path / "run-policy.json"))
    r1_base = (*common, *target_policy, "--run-dir", str(r1))
    r2_base = (*common, *target_policy, "--run-dir", str(r2), "--data-seed", "19")
    _invoke("train", *r0_base, "--until-step", "4")
    telemetry_after_train = (r0 / "telemetry.jsonl").read_bytes()
    with pytest.raises(SystemExit, match="fresh training requires"):
        main(("train", *r0_base, "--until-step", "1"))
    assert (r0 / "telemetry.jsonl").read_bytes() == telemetry_after_train
    shutil.copy(r0 / "checkpoints" / "final.pt", r0 / "checkpoints" / "phase1.pt")
    shutil.copy(r0 / "trajectory" / "final.pt", r0 / "trajectory" / "phase1.pt")
    r0_manifest_before = (r0 / "run.manifest.json").read_bytes()
    with pytest.raises(SystemExit, match="cannot change"):
        main(
            (
                "train",
                *r0_base,
                *target_policy,
                "--resume",
                str(r0 / "checkpoints" / "phase1.pt"),
                "--seal-endpoint",
            )
        )
    assert (r0 / "run.manifest.json").read_bytes() == r0_manifest_before
    _invoke(
        "endpoint-analyze",
        "--snapshots",
        str(r0 / "trajectory" / "initial.pt"),
        str(r0 / "trajectory" / "step_00000002.pt"),
        str(r0 / "trajectory" / "phase1.pt"),
        "--model-config",
        str(tmp_path / "model.json"),
        "--output",
        str(r0 / "endpoint-geometry.json"),
    )
    _invoke("train", *r1_base, "--until-step", "4", "--seal-endpoint")
    _invoke("train", *r2_base, "--until-step", "4", "--seal-endpoint")
    if device == "mps":
        _invoke(
            "score",
            *r0_base,
            "--checkpoint",
            str(r0 / "checkpoints" / "phase1.pt"),
            "--output",
            str(r0 / "difficulty.jsonl"),
        )
        _invoke(
            "analyze",
            *r1_base,
            "--checkpoint",
            str(r1 / "checkpoints" / "final.pt"),
            "--snapshots",
            str(r1 / "trajectory" / "initial.pt"),
            str(r1 / "trajectory" / "final.pt"),
            "--trajectory-config",
            str(tmp_path / "trajectory.json"),
            "--target-step",
            "6",
            "--scope",
            "online",
            "--output-dir",
            str(r1 / "mps-transport"),
        )
        r0_receipt = r0 / "train.receipt.json"
        r1_receipt = r1 / "train.receipt.json"
        analysis_receipt = r1 / "mps-transport" / "analysis.receipt.json"
        train_cost = json.loads(r0_receipt.read_text(encoding="utf-8"))
        analysis_cost = json.loads(analysis_receipt.read_text(encoding="utf-8"))
        assert train_cost["accelerator_seconds"] is None
        assert train_cost["actual_peak_working_memory_bytes"] is None
        assert train_cost["current_working_memory_bytes"] >= 0
        assert analysis_cost["accelerator_seconds"] is None
        assert analysis_cost["actual_peak_working_memory_bytes"] is None
        assert analysis_cost["cost_components"]["analysis"]["device_active_wall_seconds"] == 0.0
        assert (
            analysis_cost["device_active_wall_seconds"]
            == analysis_cost["cost_components"]["verification_per_replay"][
                "device_active_wall_seconds"
            ]
        )

        def mps_receipt(path: Path) -> dict[str, object]:
            return {
                "receipt": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "estimated_cost_usd": None,
            }

        analysis_reference = mps_receipt(analysis_receipt)
        _json(
            tmp_path / "mps-cost.json",
            {
                "format_version": 1,
                "records": {
                    "reference": mps_receipt(r0_receipt),
                    "analysis": analysis_reference,
                    "checkpoint_io": analysis_reference,
                    "verification_per_replay": analysis_reference,
                    "replay": mps_receipt(r1_receipt),
                    "baseline_replay": mps_receipt(r0_receipt),
                },
                "resource_receipt": mps_receipt(r0_receipt),
                "amortized_uses": [1, 3],
            },
        )
        _invoke(
            "cost-report",
            "--input",
            str(tmp_path / "mps-cost.json"),
            "--output",
            str(tmp_path / "mps-cost-report.json"),
        )
        mps_cost_report = json.loads(
            (tmp_path / "mps-cost-report.json").read_text(encoding="utf-8")
        )
        assert mps_cost_report["total_discovery"]["accelerator_seconds"] is None
        assert mps_cost_report["actual_peak_working_memory_bytes"] is None
        assert mps_cost_report["total_discovery"]["wall_seconds"] > 0.0
        assert mps_cost_report["total_discovery"]["cpu_seconds"] > 0.0
        return
    r1_manifest_before = (r1 / "run.manifest.json").read_bytes()
    r1_policy_path = r1 / "run.policy.commitment.json"
    r1_policy_before = r1_policy_path.read_bytes()
    r1_telemetry_before = (r1 / "telemetry.jsonl").read_bytes()
    tampered_policy = json.loads(r1_policy_before)
    tampered_policy["heldout_anchor_selection_sha256"] = "0" * 64
    _json(r1_policy_path, tampered_policy)
    with pytest.raises(SystemExit, match="policy commitment does not match"):
        main(
            (
                "train",
                *r1_base,
                "--resume",
                str(r1 / "checkpoints" / "final.pt"),
                "--seal-endpoint",
            )
        )
    assert (r1 / "telemetry.jsonl").read_bytes() == r1_telemetry_before
    r1_policy_path.write_bytes(r1_policy_before)
    with pytest.raises(SystemExit, match="cannot change"):
        main(
            (
                "train",
                *r1_base,
                "--resume",
                str(r1 / "checkpoints" / "final.pt"),
            )
        )
    assert (r1 / "run.manifest.json").read_bytes() == r1_manifest_before
    _invoke(
        "score",
        *r0_base,
        "--checkpoint",
        str(r0 / "checkpoints" / "phase1.pt"),
        "--output",
        str(r0 / "difficulty.jsonl"),
    )
    assert (r0 / "telemetry.jsonl").read_bytes() == telemetry_after_train
    strict_run = tmp_path / "strict-run"
    strict_base = (
        "--prepared",
        str(prepared),
        "--model-config",
        str(tmp_path / "model.json"),
        "--train-config",
        str(tmp_path / "train.json"),
        "--run-dir",
        str(strict_run),
        "--batch-size",
        "2",
        "--curriculum",
        "strict_hard_to_easy",
        "--difficulty",
        str(r0 / "difficulty.jsonl"),
    )
    _invoke("train", *strict_base, "--until-step", "1")
    _invoke(
        "generate",
        *r0_base,
        "--checkpoint",
        str(r0 / "checkpoints" / "phase1.pt"),
        "--suite",
        str(tmp_path / "suite.json"),
        "--output",
        str(r0 / "generations.jsonl"),
        "--max-new-tokens",
        "8",
        "--evaluation-manifest-id",
        "tiny-phase-comparison",
    )
    assert (r0 / "telemetry.jsonl").read_bytes() == telemetry_after_train
    _invoke(
        "metrics",
        "--prepared",
        str(prepared),
        "--suite",
        str(tmp_path / "suite.json"),
        "--records",
        str(r0 / "generations.jsonl"),
        "--manifest",
        str(r0 / "generations.manifest.json"),
        "--output",
        str(r0 / "metrics.json"),
    )
    _invoke(
        "analyze",
        *r1_base,
        "--checkpoint",
        str(r1 / "checkpoints" / "final.pt"),
        "--snapshots",
        str(r0 / "trajectory" / "initial.pt"),
        str(r0 / "trajectory" / "phase1.pt"),
        "--trajectory-config",
        str(tmp_path / "trajectory.json"),
        "--target-step",
        "6",
        "--scope",
        "level1",
        "--reference-manifest",
        str(r0 / "run.manifest.json"),
        "--target-manifest",
        str(r1 / "run.manifest.json"),
        "--apply",
        "--output-dir",
        str(r1 / "transport"),
    )
    _invoke(
        "analyze",
        *r2_base,
        "--checkpoint",
        str(r2 / "checkpoints" / "final.pt"),
        "--snapshots",
        str(r0 / "trajectory" / "initial.pt"),
        str(r0 / "trajectory" / "phase1.pt"),
        "--trajectory-config",
        str(tmp_path / "trajectory.json"),
        "--target-step",
        "6",
        "--method",
        "low-rank",
        "--scope",
        "level2",
        "--reference-manifest",
        str(r0 / "run.manifest.json"),
        "--target-manifest",
        str(r2 / "run.manifest.json"),
        "--output-dir",
        str(r2 / "level2"),
    )
    _invoke(
        "eval-loss",
        *r0_base,
        "--checkpoint",
        str(r0 / "checkpoints" / "phase1.pt"),
        "--split",
        "validation",
        "--output",
        str(r0 / "validation-loss.json"),
    )
    _invoke(
        "eval-loss",
        *r0_base,
        "--checkpoint",
        str(r0 / "checkpoints" / "phase1.pt"),
        "--split",
        "test",
        "--output",
        str(r0 / "test-loss.json"),
    )
    assert (r0 / "telemetry.jsonl").read_bytes() == telemetry_after_train
    _invoke(
        "generate",
        *r1_base,
        "--checkpoint",
        str(r1 / "transport" / "post_transport_checkpoint.pt"),
        "--suite",
        str(tmp_path / "suite.json"),
        "--output",
        str(r1 / "accelerated.jsonl"),
        "--max-new-tokens",
        "8",
        "--evaluation-manifest-id",
        "tiny-phase-comparison",
    )
    tampered_records = [
        json.loads(line)
        for line in (r1 / "accelerated.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    tampered_records[0]["case_id"] = (
        "bells" if tampered_records[0]["case_id"] == "river" else "river"
    )
    _jsonl(r1 / "tampered.jsonl", tampered_records)
    tampered_manifest = json.loads((r1 / "accelerated.manifest.json").read_text(encoding="utf-8"))
    tampered_manifest["records_hash"] = hashlib.sha256(
        (r1 / "tampered.jsonl").read_bytes()
    ).hexdigest()
    _json(r1 / "tampered.manifest.json", tampered_manifest)
    with pytest.raises(SystemExit, match="case/seed"):
        main(
            (
                "blind-pack",
                "--suite",
                str(tmp_path / "suite.json"),
                "--records-a",
                str(r0 / "generations.jsonl"),
                "--records-b",
                str(r1 / "tampered.jsonl"),
                "--manifest-a",
                str(r0 / "generations.manifest.json"),
                "--manifest-b",
                str(r1 / "tampered.manifest.json"),
                "--checkpoint-id",
                "tiny-phase-comparison",
                "--blind-seed",
                "3",
                "--candidate-a",
                "reference",
                "--candidate-b",
                "accelerated",
                "--output",
                str(r1 / "rejected-blind.jsonl"),
                "--max-new-tokens",
                "8",
            )
        )
    _invoke(
        "blind-pack",
        "--suite",
        str(tmp_path / "suite.json"),
        "--records-a",
        str(r0 / "generations.jsonl"),
        "--records-b",
        str(r1 / "accelerated.jsonl"),
        "--manifest-a",
        str(r0 / "generations.manifest.json"),
        "--manifest-b",
        str(r1 / "accelerated.manifest.json"),
        "--checkpoint-id",
        "tiny-phase-comparison",
        "--blind-seed",
        "3",
        "--candidate-a",
        "reference",
        "--candidate-b",
        "accelerated",
        "--output",
        str(r1 / "blind.jsonl"),
        "--max-new-tokens",
        "8",
    )
    blind_rows = [
        json.loads(line) for line in (r1 / "blind.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    _jsonl(
        r1 / "judgments.jsonl",
        [
            {
                "comparison_id": row["comparison_id"],
                "prompt_relevance": "tie",
                "poetic_quality": "tie",
                "image_music": "tie",
                "degeneration": "tie",
                "notes": "tiny mechanics judgment",
            }
            for row in blind_rows
        ],
    )
    _invoke(
        "blind-aggregate",
        "--blind-pack",
        str(r1 / "blind.jsonl"),
        "--key",
        str(r1 / "blind.key.json"),
        "--judgments",
        str(r1 / "judgments.jsonl"),
        "--candidate-a",
        "reference",
        "--candidate-b",
        "accelerated",
        "--output",
        str(r1 / "blind-tallies.json"),
    )

    def receipt(path: Path) -> dict[str, object]:
        return {
            "receipt": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "estimated_cost_usd": None,
        }

    r0_receipt = r0 / "train.receipt.json"
    r1_receipt = r1 / "train.receipt.json"
    analysis_receipt = r1 / "transport" / "analysis.receipt.json"
    analysis_cost = json.loads(analysis_receipt.read_text(encoding="utf-8"))
    components = analysis_cost["cost_components"]
    assert set(components) == {
        "analysis",
        "checkpoint_io",
        "verification_per_replay",
    }
    assert sum(component["wall_seconds"] for component in components.values()) == pytest.approx(
        analysis_cost["wall_seconds"]
    )
    assert sum(
        component["process_cpu_seconds"] for component in components.values()
    ) == pytest.approx(analysis_cost["process_cpu_seconds"])
    _json(
        tmp_path / "cost.json",
        {
            "format_version": 1,
            "records": {
                "reference": receipt(r0_receipt),
                "analysis": receipt(analysis_receipt),
                "checkpoint_io": receipt(analysis_receipt),
                "verification_per_replay": receipt(analysis_receipt),
                "replay": receipt(r1_receipt),
                "baseline_replay": receipt(r0_receipt),
            },
            "resource_receipt": receipt(r0_receipt),
            "amortized_uses": [1, 3],
        },
    )
    _invoke(
        "cost-report",
        "--input",
        str(tmp_path / "cost.json"),
        "--output",
        str(tmp_path / "cost-report.json"),
    )
    cost_report = json.loads((tmp_path / "cost-report.json").read_text(encoding="utf-8"))
    reference_cost = json.loads(r0_receipt.read_text(encoding="utf-8"))
    assert cost_report["total_discovery"]["accelerator_seconds"] is None
    assert cost_report["amortized"]["3"]["accelerator_seconds"] is None
    assert cost_report["total_discovery"]["wall_seconds"] == pytest.approx(
        reference_cost["command_wall_seconds"]
        + components["analysis"]["wall_seconds"]
        + components["checkpoint_io"]["wall_seconds"]
    )
    assert cost_report["total_discovery"]["cpu_seconds"] == pytest.approx(
        reference_cost["process_cpu_seconds"]
        + components["analysis"]["process_cpu_seconds"]
        + components["checkpoint_io"]["process_cpu_seconds"]
    )
    _invoke(
        "train",
        *r1_base,
        "--resume",
        str(r1 / "transport" / "post_transport_checkpoint.pt"),
        "--seal-endpoint",
    )
    resumed = json.loads((r1 / "train.receipt.json").read_text(encoding="utf-8"))
    assert (r1 / "transport" / "post_transport_checkpoint.pt").is_file()
    assert (r1 / "blind.key.json").is_file()
    assert (r2 / "level2" / "experiment.manifest.json").is_file()
    assert (r0 / "endpoint-geometry.json").is_file()
    assert resumed["global_step"] == 8
    assert resumed["optimizer_steps_executed"] == 6
    assert resumed["virtual_steps_skipped"] == 2
