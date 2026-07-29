from __future__ import annotations

import json
import math
import shutil
import time
from collections.abc import Mapping, Sequence
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any

import torch

from ..adapters.gpt_neox import model_from_canonical_state
from ..bit_accounting import account_mgp
from ..hashing import sha256_directory, sha256_file, sha256_json, sha256_state_dict
from ..io import (
    load_tensor_file,
    read_json,
    replace_directory_atomic,
    resolve_artifact_directory,
    resolve_artifact_member,
    resolve_artifact_relative_directory,
    save_tensor_file,
    temporary_directory,
    write_json,
)
from ..metrics import logits_kl, parameter_metrics, topk_agreement
from ..mgp.interpreter import decode_program
from ..mgp.serializer import load_program
from ..neural.multilife_decoder import load_shared_decoder
from ..state import apply_delta, compute_delta
from ..tensor_inventory import tied_owner_map
from ..types import TensorSpec
from .lives import CanonicalModelLife


def _validated_prediction(path: str | Path, hidden_run_id: str) -> dict[str, Any]:
    root = Path(path).expanduser().resolve(strict=True)
    value = read_json(resolve_artifact_member(root, "manifest.json", field="manifest"))
    if (
        not isinstance(value, dict)
        or value.get("format") != "GENOME_HIDDEN_PREDICTION"
        or value.get("version") != "0.1.0"
    ):
        raise ValueError("unsupported hidden prediction artifact")
    if value.get("hidden_run_id") != hidden_run_id:
        raise ValueError("hidden prediction targets a different run")
    if value.get("target_endpoint_seen") is not False:
        raise ValueError("hidden prediction does not prove endpoint isolation")
    content = dict(value)
    declared = content.pop("content_sha256", None)
    if sha256_json(content) != declared:
        raise ValueError("hidden prediction manifest hash mismatch")
    code_path = resolve_artifact_member(
        root,
        value["code_file"],
        field="prediction.code_file",
    )
    if sha256_file(code_path) != value["code_sha256"]:
        raise ValueError("hidden prediction code hash mismatch")
    program_path = resolve_artifact_directory(
        root,
        value["mgp_path"],
        field="prediction.mgp_path",
    )
    if sha256_directory(program_path) != value["predicted_mgp_sha256"]:
        raise ValueError("hidden prediction MGP hash mismatch")
    return value


def _load_revealed_state_triplet(
    *,
    sealed_hidden_life: CanonicalModelLife,
    revealed_hidden_life: CanonicalModelLife,
    runtime_execution_path: str | Path,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, torch.Tensor],
    dict[str, Any],
]:
    if sealed_hidden_life.run_id != revealed_hidden_life.run_id:
        raise ValueError("sealed and revealed hidden lives differ")
    if (
        sealed_hidden_life.manifest["W0"]["canonical_state_sha256"]
        != revealed_hidden_life.manifest["W0"]["canonical_state_sha256"]
    ):
        raise ValueError("hidden W0 changed during endpoint reveal")
    runtime_root = Path(runtime_execution_path).expanduser().resolve(strict=True)
    runtime_manifest_path = resolve_artifact_member(
        runtime_root,
        "manifest.json",
        field="runtime.manifest",
    )
    runtime_manifest = read_json(runtime_manifest_path)
    if not isinstance(runtime_manifest, dict):
        raise TypeError("runtime manifest must be an object")
    if (
        runtime_manifest.get("format") != "GENOME_HIDDEN_RUNTIME_EXECUTION"
        or runtime_manifest.get("version") != "0.1.0"
    ):
        raise ValueError("unsupported hidden runtime execution")
    content = dict(runtime_manifest)
    declared = content.pop("content_sha256", None)
    if sha256_json(content) != declared:
        raise ValueError("hidden runtime execution manifest hash mismatch")
    if runtime_manifest.get("target_endpoint_seen") is not False:
        raise ValueError("runtime candidate was not produced under endpoint isolation")
    candidate_path = resolve_artifact_member(
        runtime_root,
        runtime_manifest["candidate_state_file"],
        field="runtime.candidate_state_file",
    )
    if sha256_file(candidate_path) != runtime_manifest["candidate_state_file_sha256"]:
        raise ValueError("runtime candidate state hash mismatch")
    candidate = load_tensor_file(candidate_path)
    if sha256_state_dict(candidate) != runtime_manifest["candidate_state_sha256"]:
        raise ValueError("runtime candidate state content hash mismatch")
    return (
        revealed_hidden_life.load_base(),
        candidate,
        revealed_hidden_life.load_target(),
        runtime_manifest,
    )


def _plain_json(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _plain_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain_json(item) for item in value]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        return _plain_json(item_method())
    raise TypeError(f"LM Evaluation Harness produced a non-JSON value: {type(value)!r}")


def _task_source_sha256(path: Path) -> str:
    files = sorted(
        item for item in path.rglob("*") if item.is_file() and item.suffix in {".py", ".yaml"}
    )
    if not files:
        raise ValueError(f"LM Evaluation Harness task directory has no source files: {path}")
    records = []
    for file in files:
        if file.is_symlink():
            raise ValueError(f"LM Evaluation Harness task source must not be a symlink: {file}")
        records.append(
            {
                "path": file.relative_to(path).as_posix(),
                "sha256": sha256_file(file),
            }
        )
    return sha256_json(records)


@torch.inference_mode()
def execute_hidden_prediction(
    hidden_life: CanonicalModelLife,
    *,
    tensor_specs: Sequence[TensorSpec],
    tied_groups: Sequence[Sequence[str]],
    shared_decoder_path: str | Path,
    prediction_path: str | Path,
    config_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
) -> dict[str, Any]:
    if hidden_life.split != "hidden":
        raise ValueError("runtime execution requires a hidden-split life")
    target = hidden_life.manifest.get("WT")
    if not isinstance(target, Mapping) or target.get("canonical_file") is not None:
        raise ValueError("hidden endpoint must remain unavailable during runtime execution")
    prediction_root = Path(prediction_path).expanduser().resolve(strict=True)
    prediction = _validated_prediction(prediction_root, hidden_life.run_id)
    interpreter, decoder_manifest, _ = load_shared_decoder(
        shared_decoder_path,
        device=device,
    )
    decoder_root = Path(shared_decoder_path).expanduser().resolve(strict=True)
    if sha256_file(decoder_root / "manifest.json") != prediction["shared_decoder_manifest_sha256"]:
        raise ValueError("hidden prediction and decoder do not match")
    program_path = resolve_artifact_directory(
        prediction_root,
        prediction["mgp_path"],
        field="prediction.mgp_path",
    )
    program = load_program(program_path)
    base = hidden_life.load_base()
    start = time.perf_counter()
    candidate = decode_program(
        program,
        base,
        tensor_specs,
        tied_groups=tied_groups,
        interpreter=interpreter,
        verify_checksums=False,
    )
    decode_seconds = time.perf_counter() - start
    if set(candidate) != {spec.name for spec in tensor_specs}:
        raise ValueError("decoded hidden candidate inventory is incomplete")
    if any(not torch.isfinite(tensor).all() for tensor in candidate.values()):
        raise ValueError("decoded hidden candidate contains non-finite values")
    model = model_from_canonical_state(config_path, candidate, device=device)
    vocab_size = int(model.config.vocab_size)
    input_ids = torch.tensor([[0, min(1, vocab_size - 1)]], device=device)
    logits = model(input_ids=input_ids).logits
    if logits.shape != (1, 2, vocab_size) or not torch.isfinite(logits).all():
        raise ValueError("decoded hidden candidate failed the model forward-pass smoke")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    temp = temporary_directory(destination.parent, f".{destination.name}.building.")
    try:
        state_path = temp / "predicted_WT.safetensors"
        save_tensor_file(state_path, candidate)
        manifest = {
            "format": "GENOME_HIDDEN_RUNTIME_EXECUTION",
            "version": "0.1.0",
            "hidden_run_id": hidden_life.run_id,
            "target_endpoint_seen": False,
            "prediction_manifest_sha256": sha256_file(prediction_root / "manifest.json"),
            "shared_decoder_manifest_sha256": sha256_file(decoder_root / "manifest.json"),
            "candidate_state_file": "predicted_WT.safetensors",
            "candidate_state_file_sha256": sha256_file(state_path),
            "candidate_state_sha256": sha256_state_dict(candidate),
            "decode_seconds": decode_seconds,
            "tensor_count": len(candidate),
            "parameter_count": sum(tensor.numel() for tensor in candidate.values()),
            "forward_smoke": {
                "input_shape": [1, 2],
                "logit_shape": list(logits.shape),
                "finite": True,
            },
            "decoder_artifact_version": decoder_manifest["version"],
        }
        manifest["content_sha256"] = sha256_json(manifest)
        write_json(temp / "manifest.json", manifest, canonical=True)
        replace_directory_atomic(temp, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def load_evaluation_texts(path: str | Path) -> list[str]:
    source = Path(path).expanduser().resolve(strict=True)
    if source.suffix == ".json":
        value = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
            raise TypeError("evaluation JSON must be an array of strings")
        texts = [item for item in value if item.strip()]
    else:
        texts = []
        with source.open("r", encoding="utf-8") as handle:
            for line in handle:
                item = json.loads(line)
                if isinstance(item, str):
                    text = item
                elif isinstance(item, Mapping) and isinstance(item.get("text"), str):
                    text = item["text"]
                else:
                    raise TypeError("evaluation JSONL rows must be strings or contain text")
                if text.strip():
                    texts.append(text)
    if not texts:
        raise ValueError("evaluation corpus contains no text")
    return texts


def materialize_wikitext_evaluation(
    *,
    output_path: str | Path,
    cache_dir: str | Path,
    repository: str,
    revision: str,
    configuration: str,
    split: str,
) -> dict[str, Any]:
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    from datasets import load_dataset

    dataset = load_dataset(
        repository,
        configuration,
        split=split,
        revision=revision,
        cache_dir=str(cache_dir),
    )
    texts = [
        str(item["text"])
        for item in dataset
        if isinstance(item, Mapping) and isinstance(item.get("text"), str) and item["text"].strip()
    ]
    if not texts:
        raise ValueError("pinned Wikitext evaluation split contains no text")
    temp = temporary_directory(destination.parent, f".{destination.name}.building.")
    try:
        records = temp / "texts.jsonl"
        with records.open("w", encoding="utf-8") as handle:
            for text in texts:
                handle.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        manifest = {
            "format": "GENOME_EVALUATION_TEXT_CORPUS",
            "version": "0.1.0",
            "repository": repository,
            "revision": revision,
            "configuration": configuration,
            "split": split,
            "record_count": len(texts),
            "records_file": "texts.jsonl",
            "records_sha256": sha256_file(records),
        }
        manifest["content_sha256"] = sha256_json(manifest)
        write_json(temp / "manifest.json", manifest, canonical=True)
        replace_directory_atomic(temp, destination)
        return manifest
    except BaseException:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def _token_batches(
    texts: Sequence[str],
    *,
    tokenizer_path: str | Path,
    sequence_length: int,
    batch_size: int,
    max_batches: int | None,
) -> list[torch.Tensor]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    eos = tokenizer.eos_token_id
    if eos is None:
        raise ValueError("Pythia tokenizer has no EOS token")
    tokens: list[int] = []
    for text in texts:
        tokens.extend(tokenizer.encode(text, add_special_tokens=False))
        tokens.append(eos)
    width = sequence_length + 1
    sequences = [
        tokens[start : start + width] for start in range(0, len(tokens) - width + 1, width)
    ]
    if not sequences:
        raise ValueError("evaluation corpus is shorter than one requested sequence")
    if max_batches is not None:
        sequences = sequences[: max_batches * batch_size]
    batches = []
    for start in range(0, len(sequences), batch_size):
        chunk = sequences[start : start + batch_size]
        if len(chunk) == batch_size:
            batches.append(torch.tensor(chunk, dtype=torch.long))
    if not batches:
        raise ValueError("evaluation corpus did not form a complete batch")
    return batches


@torch.inference_mode()
def _evaluate_state(
    state: Mapping[str, torch.Tensor],
    *,
    config_path: str | Path,
    batches: Sequence[torch.Tensor],
    device: str,
    anchors_per_batch: int,
) -> dict[str, Any]:
    model = model_from_canonical_state(config_path, state, device=device)
    device_object = torch.device(device)
    total_loss = 0.0
    total_tokens = 0
    anchors = []
    start = time.perf_counter()
    for batch in batches:
        input_ids = batch.to(device_object)
        logits = model(input_ids=input_ids[:, :-1]).logits
        targets = input_ids[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="sum",
        )
        total_loss += float(loss.item())
        total_tokens += targets.numel()
        flat = logits.reshape(-1, logits.shape[-1])
        count = min(anchors_per_batch, flat.shape[0])
        positions = (
            torch.linspace(
                0,
                flat.shape[0] - 1,
                steps=count,
                device=flat.device,
            )
            .round()
            .to(torch.long)
        )
        anchors.append(flat.index_select(0, positions).to(torch.float32).cpu())
    elapsed = time.perf_counter() - start
    mean_loss = total_loss / total_tokens
    return {
        "batch_count": len(batches),
        "token_count": total_tokens,
        "mean_loss": mean_loss,
        "perplexity": math.exp(min(mean_loss, 80.0)),
        "seconds": elapsed,
        "anchors": anchors,
    }


def _function_comparison(
    candidate: Mapping[str, Any],
    target: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_anchors = candidate["anchors"]
    target_anchors = target["anchors"]
    if len(candidate_anchors) != len(target_anchors):
        raise ValueError("functional anchor counts differ")
    kl = []
    top1 = []
    top5 = []
    for left, right in zip(candidate_anchors, target_anchors, strict=True):
        kl.append(logits_kl(left, right))
        top1.append(topk_agreement(left, right, 1))
        top5.append(topk_agreement(left, right, 5))
    return {
        "candidate_mean_loss": candidate["mean_loss"],
        "target_mean_loss": target["mean_loss"],
        "loss_gap": candidate["mean_loss"] - target["mean_loss"],
        "candidate_perplexity": candidate["perplexity"],
        "target_perplexity": target["perplexity"],
        "anchor_logit_kl": sum(kl) / len(kl),
        "top1_agreement": sum(top1) / len(top1),
        "top5_agreement": sum(top5) / len(top5),
    }


def _matched_state_evaluation(
    *,
    base: Mapping[str, torch.Tensor],
    candidate: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
    config_path: str | Path,
    batches: Sequence[torch.Tensor],
    device: str,
    anchors_per_batch: int,
) -> dict[str, Any]:
    base_eval = _evaluate_state(
        base,
        config_path=config_path,
        batches=batches,
        device=device,
        anchors_per_batch=anchors_per_batch,
    )
    candidate_eval = _evaluate_state(
        candidate,
        config_path=config_path,
        batches=batches,
        device=device,
        anchors_per_batch=anchors_per_batch,
    )
    target_eval = _evaluate_state(
        target,
        config_path=config_path,
        batches=batches,
        device=device,
        anchors_per_batch=anchors_per_batch,
    )
    comparison = _function_comparison(candidate_eval, target_eval)
    for result in (base_eval, candidate_eval, target_eval):
        result.pop("anchors")
    return {
        "W0": base_eval,
        "decoded_WT": candidate_eval,
        "true_WT": target_eval,
        "decoded_vs_true": comparison,
    }


def evaluate_development_svd_frontier(
    development_life: CanonicalModelLife,
    *,
    tensor_specs: Sequence[TensorSpec],
    tied_groups: Sequence[Sequence[str]],
    ranks: Sequence[int],
    config_path: str | Path,
    tokenizer_path: str | Path,
    evaluation_texts_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
    sequence_length: int = 512,
    batch_size: int = 4,
    anchors_per_batch: int = 8,
) -> dict[str, Any]:
    if development_life.split != "development":
        raise ValueError("SVD frontier requires one development-split life")
    checked_ranks = tuple(sorted(set(ranks)))
    if not checked_ranks or any(
        isinstance(rank, bool) or not isinstance(rank, int) or rank < 0
        for rank in checked_ranks
    ):
        raise ValueError("SVD frontier ranks must be non-negative integers")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)

    base = development_life.load_base()
    target = development_life.load_target()
    delta = compute_delta(base, target, tensor_specs)
    aliases = tied_owner_map(tied_groups)
    unique_specs = [spec for spec in tensor_specs if spec.name not in aliases]
    matrix_specs = [spec for spec in unique_specs if len(spec.shape) == 2]
    maximum_rank = max(min(spec.shape) for spec in matrix_specs)
    if checked_ranks[-1] > maximum_rank:
        raise ValueError("SVD frontier rank exceeds the largest matrix rank")

    active_device = torch.device(device)
    factors: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    with torch.inference_mode():
        for spec in matrix_specs:
            u, singular, vh = torch.linalg.svd(
                delta[spec.name].to(device=active_device, dtype=torch.float32),
                full_matrices=False,
            )
            factors[spec.name] = (
                u.to(dtype=torch.float16, device="cpu"),
                singular.to(dtype=torch.float16, device="cpu"),
                vh.to(dtype=torch.float16, device="cpu"),
            )

    texts = load_evaluation_texts(evaluation_texts_path)
    batches = _token_batches(
        texts,
        tokenizer_path=tokenizer_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        max_batches=None,
    )
    base_evaluation = _evaluate_state(
        base,
        config_path=config_path,
        batches=batches,
        device=device,
        anchors_per_batch=anchors_per_batch,
    )
    target_evaluation = _evaluate_state(
        target,
        config_path=config_path,
        batches=batches,
        device=device,
        anchors_per_batch=anchors_per_batch,
    )

    candidates = []
    for rank in checked_ranks:
        reconstructed_delta = {}
        payload_values = 0
        with torch.inference_mode():
            for spec in tensor_specs:
                if spec.name in aliases:
                    reconstructed_delta[spec.name] = torch.zeros_like(delta[spec.name])
                elif len(spec.shape) == 2:
                    u, singular, vh = factors[spec.name]
                    active_rank = min(rank, singular.numel())
                    if active_rank == 0:
                        value = torch.zeros_like(delta[spec.name])
                    else:
                        left = u[:, :active_rank].to(device=active_device, dtype=torch.float32)
                        scale = singular[:active_rank].to(
                            device=active_device,
                            dtype=torch.float32,
                        )
                        right = vh[:active_rank].to(
                            device=active_device,
                            dtype=torch.float32,
                        )
                        value = ((left * scale) @ right).cpu()
                    reconstructed_delta[spec.name] = value
                    payload_values += active_rank * (spec.shape[0] + spec.shape[1] + 1)
                else:
                    reconstructed_delta[spec.name] = (
                        delta[spec.name].to(torch.float16).to(torch.float32)
                    )
                    payload_values += spec.numel
        candidate = apply_delta(
            base,
            reconstructed_delta,
            tensor_specs,
            tied_groups=tied_groups,
        )
        candidate_evaluation = _evaluate_state(
            candidate,
            config_path=config_path,
            batches=batches,
            device=device,
            anchors_per_batch=anchors_per_batch,
        )
        comparison = _function_comparison(candidate_evaluation, target_evaluation)
        candidate_evaluation.pop("anchors")
        candidates.append(
            {
                "rank_per_matrix": rank,
                "factor_dtype": "float16",
                "payload_bytes_per_life": payload_values * 2,
                "parameter_metrics": parameter_metrics(candidate, target, tensor_specs),
                "functional": {
                    "candidate": candidate_evaluation,
                    "candidate_vs_true": comparison,
                },
            }
        )

    base_evaluation.pop("anchors")
    target_evaluation.pop("anchors")
    result = {
        "format": "GENOME_DEVELOPMENT_SVD_FRONTIER",
        "version": "0.1.0",
        "research_level": "G0",
        "development_run_id": development_life.run_id,
        "target_endpoint_seen": True,
        "hidden_endpoints_seen": False,
        "factor_dtype": "float16",
        "ranks": list(checked_ranks),
        "evaluation_corpus_sha256": sha256_file(evaluation_texts_path),
        "evaluation": {
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "batch_count": len(batches),
            "anchors_per_batch": anchors_per_batch,
            "W0": base_evaluation,
            "true_WT": target_evaluation,
        },
        "candidates": candidates,
    }
    result["content_sha256"] = sha256_json(result)
    write_json(destination, result, canonical=True)
    return result


def evaluate_shared_decoder_corpus(
    *,
    training_lives: Sequence[CanonicalModelLife],
    development_life: CanonicalModelLife,
    tensor_specs: Sequence[TensorSpec],
    tied_groups: Sequence[Sequence[str]],
    shared_decoder_path: str | Path,
    development_code_path: str | Path,
    config_path: str | Path,
    tokenizer_path: str | Path,
    evaluation_texts_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
    sequence_length: int = 512,
    batch_size: int = 4,
    max_batches: int | None = None,
    anchors_per_batch: int = 8,
) -> dict[str, Any]:
    if not training_lives or any(life.split != "training" for life in training_lives):
        raise ValueError("shared decoder audit accepts only training-split training lives")
    if development_life.split != "development":
        raise ValueError("shared decoder audit requires one development-split life")
    all_lives = [*training_lives, development_life]
    if len({life.run_id for life in all_lives}) != len(all_lives):
        raise ValueError("shared decoder audit model-life IDs must be unique")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    decoder_root = Path(shared_decoder_path).expanduser().resolve(strict=True)
    interpreter, decoder_manifest, _ = load_shared_decoder(
        decoder_root,
        device=device,
    )
    interpreter_root = resolve_artifact_directory(
        decoder_root,
        decoder_manifest["interpreter"]["path"],
        field="interpreter.path",
    )
    raw_training_records = decoder_manifest.get("codes")
    if not isinstance(raw_training_records, list) or any(
        not isinstance(record, Mapping) or not isinstance(record.get("run_id"), str)
        for record in raw_training_records
    ):
        raise TypeError("shared decoder fitted-code records are invalid")
    training_records = {str(record["run_id"]): record for record in raw_training_records}
    if len(training_records) != len(raw_training_records):
        raise ValueError("shared decoder fitted-code records contain duplicate model-life IDs")
    if set(training_records) != {life.run_id for life in training_lives}:
        raise ValueError("shared decoder fitted-code records differ from training lives")
    if decoder_manifest.get("hidden_run_ids") != []:
        raise ValueError("shared decoder manifest reports hidden endpoint access")

    development_root = Path(development_code_path).expanduser().resolve(strict=True)
    development_manifest_path = resolve_artifact_member(
        development_root,
        "manifest.json",
        field="development.manifest",
    )
    development_manifest = read_json(development_manifest_path)
    if not isinstance(development_manifest, dict):
        raise TypeError("development fitted-code manifest must be an object")
    if (
        development_manifest.get("format") != "GENOME_FITTED_LIFE_CODE"
        or development_manifest.get("version") != "0.1.0"
    ):
        raise ValueError("unsupported development fitted-code artifact")
    development_content = dict(development_manifest)
    development_declared = development_content.pop("content_sha256", None)
    if sha256_json(development_content) != development_declared:
        raise ValueError("development fitted-code manifest hash mismatch")
    if development_manifest.get("run_id") != development_life.run_id:
        raise ValueError("development fitted-code artifact targets another life")
    if (
        development_manifest.get("split") != "development"
        or development_manifest.get("target_endpoint_seen_during_fit") is not True
    ):
        raise ValueError("development code was not fitted as a development G0 genome")
    if development_manifest.get("shared_decoder_manifest_sha256") != sha256_file(
        decoder_root / "manifest.json"
    ):
        raise ValueError("development fitted-code artifact uses another shared decoder")

    texts = load_evaluation_texts(evaluation_texts_path)
    batches = _token_batches(
        texts,
        tokenizer_path=tokenizer_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    life_results = []
    for life in all_lives:
        if life.split == "training":
            record = training_records[life.run_id]
            program_path = resolve_artifact_relative_directory(
                decoder_root,
                record["genome_path"],
                field=f"codes.{life.run_id}.genome_path",
            )
            expected_program_sha256 = record["genome_sha256"]
        else:
            program_path = resolve_artifact_directory(
                development_root,
                development_manifest["genome_path"],
                field="development.genome_path",
            )
            expected_program_sha256 = development_manifest["genome_sha256"]
        if sha256_directory(program_path) != expected_program_sha256:
            raise ValueError(f"fitted genome hash mismatch for {life.run_id}")
        base = life.load_base()
        target = life.load_target()
        program = load_program(program_path)
        start = time.perf_counter()
        candidate = decode_program(
            program,
            base,
            tensor_specs,
            tied_groups=tied_groups,
            interpreter=interpreter,
            verify_checksums=False,
        )
        decode_seconds = time.perf_counter() - start
        matched = _matched_state_evaluation(
            base=base,
            candidate=candidate,
            target=target,
            config_path=config_path,
            batches=batches,
            device=device,
            anchors_per_batch=anchors_per_batch,
        )
        life_results.append(
            {
                "run_id": life.run_id,
                "split": life.split,
                "decode_seconds": decode_seconds,
                "parameter_metrics": parameter_metrics(candidate, target, tensor_specs),
                "functional": matched,
                "bytes": account_mgp(
                    program_path,
                    interpreter_path=interpreter_root,
                    base_path=resolve_artifact_member(
                        life.root,
                        life.manifest["W0"]["canonical_file"],
                        field=f"{life.run_id}.W0",
                    ),
                    amortization_count=len(all_lives),
                ),
            }
        )
    relative_l2 = [item["parameter_metrics"]["relative_l2"] for item in life_results]
    loss_gaps = [item["functional"]["decoded_vs_true"]["loss_gap"] for item in life_results]
    logit_kl = [item["functional"]["decoded_vs_true"]["anchor_logit_kl"] for item in life_results]
    result = {
        "format": "GENOME_SHARED_DECODER_EVALUATION",
        "version": "0.1.0",
        "research_level": "G0",
        "training_run_ids": [life.run_id for life in training_lives],
        "development_run_id": development_life.run_id,
        "hidden_endpoints_seen": False,
        "shared_decoder_manifest_sha256": sha256_file(decoder_root / "manifest.json"),
        "development_code_manifest_sha256": sha256_file(development_manifest_path),
        "evaluation_corpus_sha256": sha256_file(evaluation_texts_path),
        "evaluation": {
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "batch_count": len(batches),
            "anchors_per_batch": anchors_per_batch,
        },
        "summary": {
            "life_count": len(life_results),
            "mean_parameter_relative_l2": sum(relative_l2) / len(relative_l2),
            "worst_parameter_relative_l2": max(relative_l2),
            "mean_loss_gap": sum(loss_gaps) / len(loss_gaps),
            "worst_loss_gap": max(loss_gaps),
            "mean_anchor_logit_kl": sum(logit_kl) / len(logit_kl),
            "worst_anchor_logit_kl": max(logit_kl),
        },
        "lives": life_results,
    }
    result["content_sha256"] = sha256_json(result)
    write_json(destination, result, canonical=True)
    return result


def evaluate_revealed_prediction(
    *,
    sealed_hidden_life: CanonicalModelLife,
    revealed_hidden_life: CanonicalModelLife,
    runtime_execution_path: str | Path,
    tensor_specs: Sequence[TensorSpec],
    config_path: str | Path,
    tokenizer_path: str | Path,
    evaluation_texts_path: str | Path,
    output_path: str | Path,
    device: str = "cuda",
    sequence_length: int = 512,
    batch_size: int = 4,
    max_batches: int | None = None,
    anchors_per_batch: int = 8,
) -> dict[str, Any]:
    base, candidate, target, runtime_manifest = _load_revealed_state_triplet(
        sealed_hidden_life=sealed_hidden_life,
        revealed_hidden_life=revealed_hidden_life,
        runtime_execution_path=runtime_execution_path,
    )
    texts = load_evaluation_texts(evaluation_texts_path)
    batches = _token_batches(
        texts,
        tokenizer_path=tokenizer_path,
        sequence_length=sequence_length,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    matched = _matched_state_evaluation(
        base=base,
        candidate=candidate,
        target=target,
        config_path=config_path,
        batches=batches,
        device=device,
        anchors_per_batch=anchors_per_batch,
    )
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    result = {
        "format": "GENOME_HIDDEN_EVALUATION",
        "version": "0.1.0",
        "research_level": "G2",
        "hidden_run_id": revealed_hidden_life.run_id,
        "one_shot": True,
        "target_endpoint_seen_during_compile": False,
        "early_training_prefix_used": False,
        "repair_or_polishing_used": False,
        "runtime_candidate_state_sha256": runtime_manifest["candidate_state_sha256"],
        "evaluation_corpus_sha256": sha256_file(evaluation_texts_path),
        "evaluation": {
            "sequence_length": sequence_length,
            "batch_size": batch_size,
            "batch_count": len(batches),
            "anchors_per_batch": anchors_per_batch,
        },
        "W0": matched["W0"],
        "predicted_WT": matched["decoded_WT"],
        "true_WT": matched["true_WT"],
        "predicted_vs_true": matched["decoded_vs_true"],
        "parameter_metrics": parameter_metrics(candidate, target, tensor_specs),
    }
    result["content_sha256"] = sha256_json(result)
    write_json(destination, result, canonical=True)
    return result


def evaluate_lm_harness_revealed_prediction(
    *,
    sealed_hidden_life: CanonicalModelLife,
    revealed_hidden_life: CanonicalModelLife,
    runtime_execution_path: str | Path,
    config_path: str | Path,
    tokenizer_path: str | Path,
    task_directory: str | Path,
    tasks: Sequence[str],
    output_path: str | Path,
    device: str = "cuda",
    num_fewshot: int = 0,
    batch_size: int | str = "auto",
    max_batch_size: int = 64,
    bootstrap_iters: int = 100_000,
    random_seed: int = 0,
    numpy_random_seed: int = 1234,
    torch_random_seed: int = 1234,
    fewshot_random_seed: int = 1234,
) -> dict[str, Any]:
    if not tasks or any(not isinstance(task, str) or not task for task in tasks):
        raise ValueError("LM Evaluation Harness tasks must be non-empty strings")
    if len(set(tasks)) != len(tasks):
        raise ValueError("LM Evaluation Harness tasks must be unique")
    for name, value in (
        ("num_fewshot", num_fewshot),
        ("max_batch_size", max_batch_size),
        ("bootstrap_iters", bootstrap_iters),
    ):
        minimum = 0 if name == "num_fewshot" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            qualifier = "non-negative" if minimum == 0 else "positive"
            raise ValueError(f"{name} must be a {qualifier} integer")
    destination = Path(output_path)
    if destination.exists():
        raise FileExistsError(destination)
    task_root = Path(task_directory).expanduser().resolve(strict=True)
    if not task_root.is_dir():
        raise ValueError(f"LM Evaluation Harness task directory is invalid: {task_root}")
    base, candidate, target, runtime_manifest = _load_revealed_state_triplet(
        sealed_hidden_life=sealed_hidden_life,
        revealed_hidden_life=revealed_hidden_life,
        runtime_execution_path=runtime_execution_path,
    )

    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager
    from transformers import AutoTokenizer

    task_manager = TaskManager(
        include_path=str(task_root),
        include_defaults=False,
    )
    missing = sorted(set(tasks) - set(task_manager.all_tasks))
    if missing:
        raise ValueError(f"LM Evaluation Harness tasks are not defined: {missing}")
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        use_fast=True,
    )
    state_results: dict[str, Any] = {}
    start_all = time.perf_counter()
    for label, state in (
        ("W0", base),
        ("predicted_WT", candidate),
        ("true_WT", target),
    ):
        model = model_from_canonical_state(config_path, state, device=device)
        harness_model = HFLM(
            pretrained=model,
            backend="causal",
            tokenizer=tokenizer,
            device=device,
            dtype=torch.float32,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
        )
        start = time.perf_counter()
        evaluation = lm_eval.simple_evaluate(
            model=harness_model,
            tasks=list(tasks),
            num_fewshot=num_fewshot,
            batch_size=batch_size,
            max_batch_size=max_batch_size,
            device=device,
            bootstrap_iters=bootstrap_iters,
            log_samples=False,
            task_manager=task_manager,
            random_seed=random_seed,
            numpy_random_seed=numpy_random_seed,
            torch_random_seed=torch_random_seed,
            fewshot_random_seed=fewshot_random_seed,
        )
        if not isinstance(evaluation, Mapping):
            raise TypeError("LM Evaluation Harness returned no evaluation result")
        state_results[label] = {
            "seconds": time.perf_counter() - start,
            "results": _plain_json(evaluation.get("results")),
            "versions": _plain_json(evaluation.get("versions")),
            "n_shot": _plain_json(evaluation.get("n-shot")),
            "higher_is_better": _plain_json(evaluation.get("higher_is_better")),
            "n_samples": _plain_json(evaluation.get("n-samples")),
        }
        del harness_model
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    result = {
        "format": "GENOME_HIDDEN_LM_EVAL",
        "version": "0.1.0",
        "research_level": "G2",
        "hidden_run_id": revealed_hidden_life.run_id,
        "one_shot": True,
        "target_endpoint_seen_during_compile": False,
        "early_training_prefix_used": False,
        "repair_or_polishing_used": False,
        "runtime_candidate_state_sha256": runtime_manifest["candidate_state_sha256"],
        "lm_eval_version": package_version("lm-eval"),
        "task_directory_sha256": _task_source_sha256(task_root),
        "tasks": list(tasks),
        "num_fewshot": num_fewshot,
        "batch_size": batch_size,
        "max_batch_size": max_batch_size,
        "bootstrap_iters": bootstrap_iters,
        "seeds": {
            "python": random_seed,
            "numpy": numpy_random_seed,
            "torch": torch_random_seed,
            "fewshot": fewshot_random_seed,
        },
        "total_seconds": time.perf_counter() - start_all,
        "states": state_results,
    }
    result["content_sha256"] = sha256_json(result)
    write_json(destination, result, canonical=True)
    return result
