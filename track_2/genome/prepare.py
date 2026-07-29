from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from .adapters import GPTNeoXAdapter
from .hashing import sha256_file
from .io import atomic_write_json, load_json
from .life import ArtifactRef, CheckpointRef, DatasetRef, ModelLife, TokenizerRef, TrainingStage
from .sources import SourcePlan
from .state import save_state


def _artifact(path: Path, *, revision: str, licence: str) -> ArtifactRef:
    return ArtifactRef(
        uri=str(path),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        revision=revision,
        licence=licence,
    )


def prepare_pythia_life(
    *,
    plan: SourcePlan,
    run_id: str,
    workspace: str | Path,
    recipe_path: str | Path,
    evidence_directory: str | Path,
) -> ModelLife:
    """Canonicalize one materialized Pythia life and write its strict manifest."""
    try:
        from transformers import AutoTokenizer, GPTNeoXForCausalLM
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("transformers is required to prepare Pythia lives") from error
    source = next((item for item in plan.lives if item.run_id == run_id), None)
    if source is None:
        raise KeyError(run_id)
    root = Path(workspace)
    source_root = root / "source" / "hf" / run_id
    w0_snapshot = source_root / "w0"
    if not w0_snapshot.is_dir():
        raise FileNotFoundError(w0_snapshot)
    wt_snapshot = source_root / "wt"
    if source.split != "hidden" and not wt_snapshot.is_dir():
        raise FileNotFoundError(wt_snapshot)
    recipe = load_json(recipe_path)
    output = root / "canonical" / "lives" / run_id
    output.mkdir(parents=True, exist_ok=False)

    w0_model = GPTNeoXForCausalLM.from_pretrained(
        str(w0_snapshot), local_files_only=True, torch_dtype=torch.float32
    )
    w0 = GPTNeoXAdapter.canonical_state(w0_model.state_dict())
    if not GPTNeoXAdapter.roundtrip_equal(w0):
        raise ValueError("W0 native/canonical/native round trip failed")
    w0_path = output / "w0.safetensors"
    save_state(w0_path, w0)
    graph = GPTNeoXAdapter.graph(w0, w0_model.config.to_dict())
    graph_path = output / "architecture.json"
    atomic_write_json(graph_path, graph.to_dict())
    config_path = output / "model_config.json"
    atomic_write_json(config_path, w0_model.config.to_dict())

    wt_path: Path | None = None
    if source.split != "hidden":
        wt_model = GPTNeoXForCausalLM.from_pretrained(
            str(wt_snapshot), local_files_only=True, torch_dtype=torch.float32
        )
        wt = GPTNeoXAdapter.canonical_state(wt_model.state_dict())
        if set(wt) != set(w0):
            raise ValueError("W0 and WT tensor names differ")
        wt_path = output / "wt.safetensors"
        save_state(wt_path, wt)

    tokenizer = AutoTokenizer.from_pretrained(str(w0_snapshot), local_files_only=True)
    tokenizer_files = []
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.json",
        "merges.txt",
    ):
        path = w0_snapshot / name
        if path.is_file():
            tokenizer_files.append(_artifact(path, revision=str(source.w0_commit), licence=source.licence))
    if not tokenizer_files:
        raise ValueError("no tokenizer files found in W0 snapshot")

    evidence_path = Path(evidence_directory) / "fingerprint.safetensors"
    if not evidence_path.is_file():
        raise FileNotFoundError(evidence_path)
    evidence_artifact = _artifact(evidence_path, revision=plan.dataset_commit or plan.dataset_revision, licence="source-defined")
    stage = recipe["stage"]
    dataset = recipe["dataset"]
    w0_ref = CheckpointRef(
        checkpoint_id="w0",
        step=0,
        tokens_seen=0,
        access="available",
        artifact=_artifact(w0_path, revision=str(source.w0_commit), licence=source.licence),
    )
    endpoint_ref = CheckpointRef(
        checkpoint_id="wt",
        step=int(stage["steps"]),
        tokens_seen=int(stage["tokens"]),
        access="sealed" if source.split == "hidden" else "available",
        artifact=None
        if wt_path is None
        else _artifact(wt_path, revision=str(source.wt_commit), licence=source.licence),
    )
    life = ModelLife(
        run_id=run_id,
        lineage_id=f"pythia-{source.size}-standard",
        split=source.split,
        completeness="complete",
        architecture_family="gpt_neox",
        architecture=w0_model.config.to_dict(),
        tensor_inventory=_artifact(graph_path, revision=str(source.w0_commit), licence=source.licence),
        tokenizer=TokenizerRef(
            repository=plan.tokenizer_repository,
            revision=str(plan.tokenizer_commit),
            tokenizer_class=tokenizer.__class__.__name__,
            vocab_size=len(tokenizer),
            special_tokens={
                name: int(value)
                for name, value in tokenizer.special_tokens_map_extended.items()
                if isinstance(value, int)
            },
            files=tuple(tokenizer_files),
        ),
        initialization=w0_ref,
        datasets=(
            DatasetRef(
                dataset_id=str(dataset["dataset_id"]),
                repository=str(dataset["repository"]),
                revision=str(dataset["revision"]),
                split=str(dataset["split"]),
                licence=str(dataset["licence"]),
                order_id=None if dataset.get("order_id") is None else str(dataset["order_id"]),
                mixture_weight=1.0,
                semantic_evidence=evidence_artifact,
            ),
        ),
        stages=(
            TrainingStage(
                stage_id="pretraining",
                kind="pretraining",
                objective=str(stage["objective"]),
                dataset_ids=(str(dataset["dataset_id"]),),
                start_checkpoint_id="w0",
                end_checkpoint_id="wt",
                steps=int(stage["steps"]),
                tokens=int(stage["tokens"]),
                context_length=int(stage["context_length"]),
                global_batch_tokens=int(stage["global_batch_tokens"]),
                precision=str(stage["precision"]),
                optimizer=dict(stage["optimizer"]),
                schedule=dict(stage["schedule"]),
                data_order_id=None if dataset.get("order_id") is None else str(dataset["order_id"]),
            ),
        ),
        trajectory=(),
        endpoint=endpoint_ref,
        compiler_evidence=evidence_artifact,
        source={
            "repository": source.repository,
            "w0_commit": source.w0_commit,
            "wt_commit": source.wt_commit if source.split != "hidden" else None,
            "source_plan_id": plan.plan_id,
        },
    )
    life.save(output / "life.json")
    return life
