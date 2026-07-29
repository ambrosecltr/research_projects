from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Mapping

import torch

from .adapters import GPTNeoXAdapter
from .hashing import sha256_file
from .io import atomic_write_json, load_json
from .life import ArtifactRef, CheckpointRef, DatasetRef, ModelLife, TokenizerRef, TrainingStage
from .sources import SourcePlan
from .state import save_state, state_id

CANONICAL_PROBE_TOKEN_IDS = tuple(range(16))


def _artifact(path: Path, *, revision: str, licence: str) -> ArtifactRef:
    return ArtifactRef(
        uri=str(path),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
        revision=revision,
        licence=licence,
    )


def _canonicalize_snapshot(
    snapshot: Path,
    *,
    probe_token_ids: tuple[int, ...],
) -> tuple[dict[str, torch.Tensor], dict[str, Any], dict[str, Any]]:
    try:
        from transformers import GPTNeoXForCausalLM
    except ImportError as error:  # pragma: no cover
        raise RuntimeError("transformers is required to canonicalize Pythia lives") from error
    model = GPTNeoXForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, torch_dtype=torch.float32
    )
    model.eval()
    native = model.state_dict()
    canonical = GPTNeoXAdapter.canonical_state(native)
    roundtrip = GPTNeoXAdapter.native_state(canonical)
    if set(roundtrip) != set(native) or any(
        not torch.equal(roundtrip[name], native[name].cpu()) for name in native
    ):
        raise ValueError(f"native/canonical/native tensor round trip failed for {snapshot}")
    probe = torch.tensor([probe_token_ids], dtype=torch.long)
    with torch.inference_mode():
        native_logits = model(input_ids=probe, use_cache=False).logits
    restored = GPTNeoXForCausalLM(model.config)
    restored.load_state_dict(roundtrip, strict=True)
    restored.eval()
    with torch.inference_mode():
        restored_logits = restored(input_ids=probe, use_cache=False).logits
    if not torch.isfinite(native_logits).all() or not torch.isfinite(restored_logits).all():
        raise ValueError(f"round-trip probe produced non-finite logits for {snapshot}")
    if not torch.equal(native_logits, restored_logits):
        raise ValueError(f"round-trip logits changed for {snapshot}")
    audit = {
        "snapshot": str(snapshot),
        "tensor_count": len(canonical),
        "parameter_count": sum(tensor.numel() for tensor in canonical.values()),
        "state_id": state_id(canonical),
        "tensor_roundtrip_exact": True,
        "logits_roundtrip_exact": True,
        "maximum_logit_difference": 0.0,
    }
    config = model.config.to_dict()
    config.pop("_name_or_path", None)
    config.pop("transformers_version", None)
    return canonical, config, audit


def canonicalize_pythia_life(
    *,
    plan: SourcePlan,
    run_id: str,
    workspace: str | Path,
    probe_token_ids: tuple[int, ...] = CANONICAL_PROBE_TOKEN_IDS,
) -> dict[str, Any]:
    """Canonicalize and functionally verify one materialized Pythia life."""
    source = next((item for item in plan.lives if item.run_id == run_id), None)
    if source is None:
        raise KeyError(run_id)
    root = Path(workspace)
    source_root = root / "source" / "hf" / run_id
    snapshots = [("w0", source_root / "w0")]
    if source.split != "hidden":
        snapshots.append(("wt", source_root / "wt"))
    for _, snapshot in snapshots:
        if not snapshot.is_dir():
            raise FileNotFoundError(snapshot)
    output = root / "canonical" / "lives" / run_id
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=f".{run_id}-", dir=output.parent) as staging_value:
        staging = Path(staging_value)
        audits: dict[str, dict[str, Any]] = {}
        w0, config, audits["w0"] = _canonicalize_snapshot(
            snapshots[0][1], probe_token_ids=probe_token_ids
        )
        save_state(staging / "w0.safetensors", w0)
        graph = GPTNeoXAdapter.graph(w0, config)
        atomic_write_json(staging / "architecture.json", graph.to_dict())
        atomic_write_json(staging / "model_config.json", config)
        if source.split != "hidden":
            wt, wt_config, audits["wt"] = _canonicalize_snapshot(
                snapshots[1][1], probe_token_ids=probe_token_ids
            )
            if set(wt) != set(w0):
                raise ValueError("W0 and WT tensor names differ")
            if wt_config != config:
                raise ValueError("W0 and WT model configurations differ")
            save_state(staging / "wt.safetensors", wt)
        audit = {
            "format": "GENOME_CANONICALIZATION_AUDIT",
            "version": "1.0.0",
            "run_id": run_id,
            "split": source.split,
            "probe_token_ids": list(probe_token_ids),
            "snapshots": audits,
            "verified": True,
        }
        atomic_write_json(staging / "canonicalization.json", audit)
        staging.rename(output)
    return audit


def prepare_pythia_life(
    *,
    plan: SourcePlan,
    run_id: str,
    workspace: str | Path,
    recipe_path: str | Path,
    evidence_directory: str | Path,
) -> ModelLife:
    """Finalize one canonical Pythia life after semantic evidence is available."""
    try:
        from transformers import AutoTokenizer
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
    w0_path = output / "w0.safetensors"
    graph_path = output / "architecture.json"
    config_path = output / "model_config.json"
    audit_path = output / "canonicalization.json"
    for path in (w0_path, graph_path, config_path, audit_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    audit = load_json(audit_path)
    if audit.get("run_id") != run_id or audit.get("verified") is not True:
        raise ValueError("canonicalization audit is missing or invalid")
    wt_path = None if source.split == "hidden" else output / "wt.safetensors"
    if wt_path is not None and not wt_path.is_file():
        raise FileNotFoundError(wt_path)
    model_config = load_json(config_path)

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
        architecture=model_config,
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
