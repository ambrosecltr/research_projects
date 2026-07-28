"""The executable, local-only workflow for an auditable personal research run."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path
from typing import BinaryIO

import torch
from tokenizers import Tokenizer

from poetry50m.config import (
    file_hash,
    load_mapping,
)
from poetry50m.data import PreparedBatchStream, PreparedDataConfig, prepare_data
from poetry50m.data.artifacts import (
    read_conditional_examples,
    read_packed_sequences,
    read_poetry_ntp_examples,
    read_prose_examples,
)
from poetry50m.data.difficulty import DifficultyLedger, DifficultyRecord
from poetry50m.data.schema import ObjectiveMix
from poetry50m.data.splits import SplitRatios
from poetry50m.data.tokenizer import TokenizerSpec
from poetry50m.evaluation import (
    PromptSuite,
    blind_comparison_pack,
    keyword_relevance,
    multi_seed_generation_requests,
    repetition_metrics,
    structural_metrics,
    training_overlaps,
)
from poetry50m.inference import (
    load_generation_records,
    run_generation_manifest,
)
from poetry50m.model import ModelConfig
from poetry50m.training import TrainConfig, Trainer
from poetry50m.trajectory._persistence import atomic_write


def _write_json(path: Path, value: object) -> None:
    payload = (
        json.dumps(
            value,
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")

    def write(handle: BinaryIO) -> None:
        handle.write(payload)

    atomic_write(path, write)


def _model_config(path: Path) -> ModelConfig:
    return ModelConfig.from_mapping(load_mapping(path))


def _train_config(path: Path) -> TrainConfig:
    return TrainConfig.from_mapping(load_mapping(path))


def _data_config(path: Path) -> PreparedDataConfig:
    value = load_mapping(path)
    expected = {
        "format_version",
        "manifest_format",
        "manifest_schema",
        "split",
        "tokenizer",
        "packing",
        "objectives",
        "rights",
    }
    if (
        set(value) != expected
        or isinstance(value["format_version"], bool)
        or not isinstance(value["format_version"], int)
        or value["format_version"] != 1
        or value["manifest_format"] != "jsonl"
        or value["manifest_schema"] != "SourceDocument"
    ):
        raise ValueError("unsupported data configuration schema")
    split, tokenizer, packing, objectives, rights = (
        value[name] for name in ("split", "tokenizer", "packing", "objectives", "rights")
    )
    if not all(isinstance(item, dict) for item in (split, tokenizer, packing, objectives, rights)):
        raise ValueError("data configuration sections must be objects")
    if (
        set(split) != {"salt", "train", "validation", "test"}
        or set(tokenizer) != {"vocab_size", "min_frequency", "special_tokens"}
        or set(packing) != {"sequence_length"}
        or set(objectives) != {"conditional_poetry", "auxiliary_prose_ntp", "poetry_ntp"}
        or set(rights) != {"allow_synthetic"}
    ):
        raise ValueError("data configuration contains unknown or missing keys")
    special_tokens = tokenizer["special_tokens"]
    if not isinstance(special_tokens, list) or not all(
        isinstance(item, str) for item in special_tokens
    ):
        raise ValueError("tokenizer.special_tokens must be a string list")
    numeric = (
        split["train"],
        split["validation"],
        split["test"],
        tokenizer["vocab_size"],
        tokenizer["min_frequency"],
        packing["sequence_length"],
        objectives["conditional_poetry"],
        objectives["auxiliary_prose_ntp"],
        objectives["poetry_ntp"],
    )
    if any(isinstance(item, bool) or not isinstance(item, (int, float)) for item in numeric):
        raise ValueError("data configuration numeric values must be numbers")
    if any(
        isinstance(item, bool) or not isinstance(item, int)
        for item in (
            tokenizer["vocab_size"],
            tokenizer["min_frequency"],
            packing["sequence_length"],
        )
    ):
        raise ValueError("tokenizer and packing sizes must be integers")
    if not isinstance(split["salt"], str) or not isinstance(rights["allow_synthetic"], bool):
        raise ValueError("split.salt must be a string and rights.allow_synthetic a boolean")
    return PreparedDataConfig(
        SplitRatios(float(split["train"]), float(split["validation"]), float(split["test"])),
        split["salt"],
        TokenizerSpec(tokenizer["vocab_size"], tokenizer["min_frequency"], tuple(special_tokens)),
        packing["sequence_length"],
        ObjectiveMix(
            float(objectives["conditional_poetry"]),
            float(objectives["auxiliary_prose_ntp"]),
            float(objectives["poetry_ntp"]),
        ),
        rights["allow_synthetic"],
    )


def _trainer(
    args: argparse.Namespace,
    *,
    resume: Path | None = None,
    read_only: bool = False,
) -> tuple[Trainer, PreparedBatchStream]:
    from poetry50m.workflows.training import trainer as create_trainer

    return create_trainer(args, resume=resume, read_only=read_only)


def corpus_acquire_command(args: argparse.Namespace) -> int:
    from poetry50m.data.hf_sources import acquire_hf_sources

    acquire_hf_sources(
        Path(args.sources_config),
        Path(args.output),
    )
    return 0


def corpus_build_command(args: argparse.Namespace) -> int:
    from poetry50m.data.knowledge_corpus import build_knowledge_corpus

    build_knowledge_corpus(
        acquisition_directory=Path(args.acquisition),
        sources_config=Path(args.sources_config),
        selection_config=Path(args.selection_config),
        output_directory=Path(args.output),
    )
    return 0


def prepare_command(args: argparse.Namespace) -> int:
    data_config = _data_config(Path(args.config))
    artifact = prepare_data(
        corpus_manifest=Path(args.corpus_manifest),
        prompt_records=Path(args.prompts),
        thought_records=Path(args.thoughts),
        pairings=Path(args.pairings) if args.pairings else None,
        output_directory=Path(args.output),
        config=data_config,
    )
    objective_stats = artifact.metadata.get("train_objective_stats")
    if not isinstance(objective_stats, dict):
        raise ValueError("prepared metadata lacks train objective statistics")

    def prepared_pack_count(objective: str) -> int:
        stats = objective_stats.get(objective)
        if not isinstance(stats, dict):
            raise ValueError(f"prepared metadata lacks {objective} statistics")
        count = stats.get("pack_count")
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            raise ValueError(f"prepared metadata has an invalid {objective} pack count")
        return count

    if (
        data_config.objective_mix.auxiliary_prose_ntp > 0.0
        and prepared_pack_count("auxiliary_prose_ntp") == 0
    ):
        raise ValueError(
            "auxiliary_prose_ntp is enabled but preparation produced no attributed prose packs"
        )
    if data_config.objective_mix.poetry_ntp > 0.0 and prepared_pack_count("poetry_ntp") == 0:
        raise ValueError("poetry_ntp is enabled but preparation produced no book-verse packs")
    _write_json(
        Path(args.output) / "prepare.receipt.json",
        {
            "artifact": str(artifact.root),
            "metadata_hash": file_hash(artifact.root / "metadata.json"),
            "synthetic_allowed": artifact.metadata["config"]["allow_synthetic"],
        },
    )
    return 0


def plan_exposure_command(args: argparse.Namespace) -> int:
    """Write a reviewable two-pass training horizon without creating a trainer."""
    from poetry50m.workflows.exposure_plan import (
        derived_train_config,
        exact_trainable_parameter_count,
        exposure_receipt,
        plan_exposure,
    )
    from poetry50m.workflows.training import prepared_stream

    prepared = Path(args.prepared)
    model_config_path = Path(args.model_config)
    train_config_path = Path(args.train_config)
    base_train = _train_config(train_config_path)
    model = _model_config(model_config_path)
    parameter_count = exact_trainable_parameter_count(model)
    if parameter_count != args.expected_parameter_count:
        raise ValueError(
            f"model has {parameter_count:,} trainable parameters; expected "
            f"{args.expected_parameter_count:,}"
        )
    data_seed = args.data_seed if args.data_seed is not None else base_train.seed
    stream = prepared_stream(prepared, args.batch_size, data_seed)
    plan = plan_exposure(
        stream,
        parameter_count=parameter_count,
        tokens_per_parameter_per_pass=args.tokens_per_parameter_per_pass,
        passes=args.passes,
    )
    derived = derived_train_config(base_train, planned_steps=plan.planned_steps)
    receipt = exposure_receipt(
        prepared=prepared,
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        batch_size=args.batch_size,
        data_seed=data_seed,
        plan=plan,
        derived_config=derived,
    )
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(f"exposure-plan output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}.planning-", dir=output.parent))
    try:
        _write_json(temporary / "receipt.json", receipt)
        _write_json(temporary / "train_config.json", asdict(derived))
        os.replace(temporary, output)
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise
    return 0


def train_command(args: argparse.Namespace) -> int:
    from poetry50m.workflows.training import train_command as execute

    return execute(args, write_json=_write_json)


def sft_train_command(args: argparse.Namespace) -> int:
    from poetry50m.workflows.sft import sft_train_command as execute

    return execute(args, write_json=_write_json)


def sft_validate_command(args: argparse.Namespace) -> int:
    from poetry50m.workflows.sft import sft_validate_command as execute

    return execute(args, write_json=_write_json)


def score_command(args: argparse.Namespace) -> int:
    trainer, _ = _trainer(args, resume=Path(args.checkpoint), read_only=True)
    tokenizer = Tokenizer.from_file(str(Path(args.prepared) / "tokenizer.json"))
    pad_id = tokenizer.token_to_id("<|pad|>")
    if pad_id is None:
        raise ValueError("prepared tokenizer lacks <|pad|>")
    ledger = DifficultyLedger()
    model = trainer.model.eval()
    with torch.inference_mode():
        for pack in read_packed_sequences(Path(args.prepared) / "train.packed.jsonl"):
            inputs = torch.tensor([pack.input_ids[:-1]], dtype=torch.long, device=trainer.device)
            targets = torch.tensor([pack.input_ids[1:]], dtype=torch.long, device=trainer.device)
            mask = torch.tensor([pack.loss_mask[1:]], dtype=torch.bool, device=trainer.device)
            output = model(inputs, targets=targets, loss_mask=mask)
            if output.loss is None or output.token_count < 1:
                raise ValueError(f"pack {pack.pack_id} has no supervised targets")
            ledger.record(
                DifficultyRecord(
                    f"{pack.objective}:pack:{pack.pack_id}",
                    output.token_count,
                    float(output.loss.item()) * output.token_count,
                    0,
                )
            )
    ledger.save(Path(args.output))
    return 0


def generate_command(args: argparse.Namespace) -> int:
    trainer, _ = _trainer(args, resume=Path(args.checkpoint), read_only=True)
    suite = PromptSuite.load(Path(args.suite))
    checkpoint_hash = file_hash(Path(args.checkpoint))
    requests = multi_seed_generation_requests(
        suite,
        checkpoint_id=checkpoint_hash,
        seeds=tuple(args.seeds),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    records = run_generation_manifest(
        trainer.model,
        Tokenizer.from_file(str(Path(args.prepared) / "tokenizer.json")),
        requests,
        Path(args.output),
    )
    _write_json(
        Path(args.output).with_suffix(".manifest.json"),
        {
            "suite_id": suite.suite_id,
            "suite_version": suite.version,
            "request_count": len(requests),
            "fixed_seeds": list(args.seeds),
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "record_count": len(records),
            "evaluation_manifest_id": args.evaluation_manifest_id,
            "candidate_checkpoint": str(Path(args.checkpoint).resolve()),
            "candidate_checkpoint_hash": checkpoint_hash,
            "records_hash": file_hash(Path(args.output)),
        },
    )
    return 0


_GENERATION_RECEIPT_KEYS = {
    "suite_id",
    "suite_version",
    "request_count",
    "fixed_seeds",
    "max_new_tokens",
    "temperature",
    "top_p",
    "record_count",
    "evaluation_manifest_id",
    "candidate_checkpoint",
    "candidate_checkpoint_hash",
    "records_hash",
}


def _generation_receipt(path: Path) -> dict[str, object]:
    receipt = load_mapping(path)
    if set(receipt) != _GENERATION_RECEIPT_KEYS:
        raise ValueError("generation receipt does not use the required immutable schema")
    string_fields = (
        "suite_id",
        "evaluation_manifest_id",
        "candidate_checkpoint",
        "candidate_checkpoint_hash",
        "records_hash",
    )
    integer_fields = ("suite_version", "request_count", "max_new_tokens", "record_count")
    if any(not isinstance(receipt[name], str) or not receipt[name] for name in string_fields):
        raise TypeError("generation receipt identity fields must be non-empty strings")
    if any(
        isinstance(receipt[name], bool) or not isinstance(receipt[name], int)
        for name in integer_fields
    ):
        raise TypeError("generation receipt count fields must be integers")
    for name in ("temperature", "top_p"):
        value = receipt[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"generation receipt {name} must be numeric")
        if not torch.isfinite(torch.tensor(float(value))).item():
            raise ValueError(f"generation receipt {name} must be finite")
    seeds = receipt["fixed_seeds"]
    if (
        not isinstance(seeds, list)
        or len(seeds) != 3
        or len(set(seeds)) != 3
        or any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds)
    ):
        raise ValueError("generation receipt must contain three distinct non-negative seeds")
    return receipt


def metrics_command(args: argparse.Namespace) -> int:
    suite = PromptSuite.load(Path(args.suite))
    receipt = _generation_receipt(Path(args.manifest))
    if receipt["suite_id"] != suite.suite_id or receipt["suite_version"] != suite.version:
        raise ValueError("generation receipt does not match this fixed prompt suite")
    if receipt["records_hash"] != file_hash(Path(args.records)):
        raise ValueError("generation records do not match their immutable receipt")
    seeds = receipt["fixed_seeds"]
    assert isinstance(seeds, list)
    max_new_tokens = receipt["max_new_tokens"]
    temperature = receipt["temperature"]
    top_p = receipt["top_p"]
    assert isinstance(max_new_tokens, int) and not isinstance(max_new_tokens, bool)
    assert isinstance(temperature, (int, float)) and not isinstance(temperature, bool)
    assert isinstance(top_p, (int, float)) and not isinstance(top_p, bool)
    requests = multi_seed_generation_requests(
        suite,
        checkpoint_id=str(receipt["candidate_checkpoint_hash"]),
        seeds=tuple(seeds),
        max_new_tokens=max_new_tokens,
        temperature=float(temperature),
        top_p=float(top_p),
    )
    cases = {case.case_id: case for case in suite.cases}
    train_texts = [
        text
        for example in read_conditional_examples(Path(args.prepared) / "train.conditional.jsonl")
        for text in (example.prompt, example.thought or "", example.poem_target)
    ]
    train_texts.extend(
        example.text for example in read_prose_examples(Path(args.prepared) / "train.prose.jsonl")
    )
    train_texts.extend(
        example.text
        for example in read_poetry_ntp_examples(Path(args.prepared) / "train.poetry_ntp.jsonl")
    )
    records = load_generation_records(Path(args.records))
    if receipt["request_count"] != len(requests) or receipt["record_count"] != len(records):
        raise ValueError("generation receipt counts do not match the fixed request/result set")
    records_by_id = {record.request_id: record for record in records}
    if set(records_by_id) != {request.request_id for request in requests}:
        raise ValueError(
            "generation records must exactly cover the fixed three-seed evaluation manifest"
        )
    for request in requests:
        record = records_by_id[request.request_id]
        if (record.case_id, record.seed, record.checkpoint_id) != (
            request.case_id,
            request.seed,
            request.checkpoint_id,
        ):
            raise ValueError("generation records do not match their immutable request manifest")
    overlap_metrics = training_overlaps(
        (record.generated_text for record in records),
        train_texts,
        workers=args.workers,
    )
    rows: list[dict[str, object]] = []
    for record, overlap in zip(records, overlap_metrics, strict=True):
        case = cases.get(record.case_id)
        if case is None:
            raise ValueError(f"unknown generation case {record.case_id}")
        rows.append(
            {
                "request_id": record.request_id,
                "case_id": record.case_id,
                "seed": record.seed,
                "keyword_relevance": keyword_relevance(record.generated_text, case.keywords),
                "repetition": asdict(repetition_metrics(record.generated_text)),
                "structure": asdict(structural_metrics(record.generated_text)),
                "training_overlap": asdict(overlap),
            }
        )
    _write_json(Path(args.output), {"record_count": len(rows), "rows": rows})
    return 0


def blind_command(args: argparse.Namespace) -> int:
    suite = PromptSuite.load(Path(args.suite))
    requests = multi_seed_generation_requests(
        suite,
        checkpoint_id="candidate-independent-request",
        seeds=tuple(args.seeds),
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    manifest_a, manifest_b = (
        _generation_receipt(Path(args.manifest_a)),
        _generation_receipt(Path(args.manifest_b)),
    )
    if (
        manifest_a["evaluation_manifest_id"] != args.checkpoint_id
        or manifest_b["evaluation_manifest_id"] != args.checkpoint_id
    ):
        raise ValueError("generation manifests must use the supplied shared evaluation manifest ID")
    if manifest_a["candidate_checkpoint_hash"] == manifest_b["candidate_checkpoint_hash"]:
        raise ValueError("blind comparison requires two distinct candidate checkpoints")
    for label, manifest in (("A", manifest_a), ("B", manifest_b)):
        checkpoint_path = Path(str(manifest["candidate_checkpoint"]))
        if file_hash(checkpoint_path) != manifest["candidate_checkpoint_hash"]:
            raise ValueError(f"candidate {label} checkpoint does not match its receipt")
    if manifest_a["records_hash"] != file_hash(Path(args.records_a)) or manifest_b[
        "records_hash"
    ] != file_hash(Path(args.records_b)):
        raise ValueError("generation records do not match their immutable manifest receipts")
    settings = (
        "suite_id",
        "suite_version",
        "fixed_seeds",
        "max_new_tokens",
        "temperature",
        "top_p",
    )
    if any(manifest_a[name] != manifest_b[name] for name in settings):
        raise ValueError("blind candidates must share the exact evaluation settings")
    records_a = load_generation_records(Path(args.records_a))
    records_b = load_generation_records(Path(args.records_b))
    if (
        manifest_a["request_count"] != len(requests)
        or manifest_b["request_count"] != len(requests)
        or manifest_a["record_count"] != len(records_a)
        or manifest_b["record_count"] != len(records_b)
    ):
        raise ValueError("blind candidate receipt counts do not match their records")
    if any(record.checkpoint_id != manifest_a["candidate_checkpoint_hash"] for record in records_a):
        raise ValueError("candidate A records have the wrong checkpoint content identity")
    if any(record.checkpoint_id != manifest_b["candidate_checkpoint_hash"] for record in records_b):
        raise ValueError("candidate B records have the wrong checkpoint content identity")
    expected_requests = {request.request_id: request for request in requests}
    for label, records in (("A", records_a), ("B", records_b)):
        request_ids = [record.request_id for record in records]
        if len(request_ids) != len(set(request_ids)):
            raise ValueError(f"candidate {label} records contain duplicate request IDs")
        if set(request_ids) != set(expected_requests):
            raise ValueError(
                f"candidate {label} records must exactly cover the fixed request manifest"
            )
        for record in records:
            expected = expected_requests[record.request_id]
            if (record.case_id, record.seed) != (expected.case_id, expected.seed):
                raise ValueError(
                    f"candidate {label} record case/seed does not match its request ID"
                )
    a = {record.request_id: record.generated_text for record in records_a}
    b = {record.request_id: record.generated_text for record in records_b}
    pack = blind_comparison_pack(
        requests=requests,
        outputs_a=a,
        outputs_b=b,
        blind_seed=args.blind_seed,
        candidate_a_id=args.candidate_a,
        candidate_b_id=args.candidate_b,
    )
    output = Path(args.output)
    pack.save_blind(output)
    pack.save_unblinding_key(output.with_suffix(".key.json"))
    return 0


def eval_loss_command(args: argparse.Namespace) -> int:
    from poetry50m.workflows.reporting import eval_loss_command as execute

    return execute(args, trainer_factory=_trainer, write_json=_write_json)


def blind_aggregate_command(args: argparse.Namespace) -> int:
    from poetry50m.workflows.reporting import blind_aggregate_command as execute

    return execute(
        args,
        load_mapping=load_mapping,
        write_json=_write_json,
    )


def cost_report_command(args: argparse.Namespace) -> int:
    from poetry50m.workflows.reporting import cost_report_command as execute

    return execute(args, load_mapping=load_mapping, write_json=_write_json)


def endpoint_analyze_command(args: argparse.Namespace) -> int:
    from poetry50m.workflows.endpoint import endpoint_analyze_command as execute

    return execute(args, model_config_loader=_model_config, write_json=_write_json)


def analyze_command(args: argparse.Namespace) -> int:
    from poetry50m.workflows.trajectory import analyze_command as execute

    return execute(args, trainer_factory=_trainer, write_json=_write_json)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="poetry50m", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    acquire = commands.add_parser("corpus-acquire")
    acquire.add_argument("--sources-config", required=True)
    acquire.add_argument("--output", required=True)
    acquire.set_defaults(handler=corpus_acquire_command)
    corpus_build = commands.add_parser("corpus-build")
    corpus_build.add_argument("--acquisition", required=True)
    corpus_build.add_argument("--sources-config", required=True)
    corpus_build.add_argument("--selection-config", required=True)
    corpus_build.add_argument("--output", required=True)
    corpus_build.set_defaults(handler=corpus_build_command)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--corpus-manifest", required=True)
    prepare.add_argument("--prompts", required=True)
    prepare.add_argument("--thoughts", required=True)
    prepare.add_argument("--pairings")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--output", required=True)
    prepare.set_defaults(handler=prepare_command)
    exposure = commands.add_parser("plan-exposure")
    exposure.add_argument("--prepared", required=True)
    exposure.add_argument("--model-config", required=True)
    exposure.add_argument("--train-config", required=True)
    exposure.add_argument("--batch-size", type=int, required=True)
    exposure.add_argument("--data-seed", type=int)
    exposure.add_argument("--tokens-per-parameter-per-pass", type=int, default=20)
    exposure.add_argument("--passes", type=int, default=2)
    exposure.add_argument("--expected-parameter-count", type=int, default=8_335_008)
    exposure.add_argument("--output", required=True)
    exposure.set_defaults(handler=plan_exposure_command)

    def training_arguments(
        command: argparse.ArgumentParser, *, run_policy_required: bool = False
    ) -> None:
        command.add_argument("--prepared", required=True)
        command.add_argument("--model-config", required=True)
        command.add_argument("--train-config", required=True)
        command.add_argument("--run-policy", required=run_policy_required)
        command.add_argument("--run-dir", required=True)
        command.add_argument("--batch-size", type=int, required=True)
        command.add_argument("--data-seed", type=int)
        command.add_argument(
            "--curriculum",
            choices=("shuffled", "strict_hard_to_easy", "cyclic_hard_to_easy"),
            default="shuffled",
        )
        command.add_argument("--difficulty")

    train = commands.add_parser("train")
    training_arguments(train)
    train.add_argument("--resume")
    train.add_argument("--until-step", type=int)
    train.add_argument("--seal-endpoint", action="store_true")
    train.set_defaults(handler=train_command)

    def sft_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--mixture", required=True)
        command.add_argument("--tokenizer", required=True)
        command.add_argument("--base-checkpoint", required=True)
        command.add_argument("--base-manifest", required=True)
        command.add_argument("--base-receipt", required=True)
        command.add_argument("--model-config", required=True)
        command.add_argument("--train-config", required=True)
        command.add_argument("--batch-size", type=int, required=True)
        command.add_argument("--data-seed", type=int)

    sft_validate = commands.add_parser("sft-validate")
    sft_arguments(sft_validate)
    sft_validate.add_argument("--output", required=True)
    sft_validate.set_defaults(handler=sft_validate_command)
    sft_train = commands.add_parser("sft-train")
    sft_arguments(sft_train)
    sft_train.add_argument("--run-dir", required=True)
    sft_train.add_argument("--resume")
    sft_train.add_argument("--until-step", type=int)
    sft_train.set_defaults(handler=sft_train_command)
    score = commands.add_parser("score")
    training_arguments(score)
    score.add_argument("--checkpoint", required=True)
    score.add_argument("--output", required=True)
    score.set_defaults(handler=score_command)
    generate = commands.add_parser("generate")
    training_arguments(generate)
    generate.add_argument("--checkpoint", required=True)
    generate.add_argument("--suite", required=True)
    generate.add_argument("--output", required=True)
    generate.add_argument("--evaluation-manifest-id", required=True)
    generate.add_argument("--seeds", type=int, nargs=3, default=(17, 29, 43))
    generate.add_argument("--max-new-tokens", type=int, default=128)
    generate.add_argument("--temperature", type=float, default=0.8)
    generate.add_argument("--top-p", type=float, default=0.95)
    generate.set_defaults(handler=generate_command)
    metrics = commands.add_parser("metrics")
    metrics.add_argument("--prepared", required=True)
    metrics.add_argument("--suite", required=True)
    metrics.add_argument("--records", required=True)
    metrics.add_argument("--manifest", required=True)
    metrics.add_argument("--output", required=True)
    metrics.add_argument("--workers", type=int)
    metrics.set_defaults(handler=metrics_command)
    blind = commands.add_parser("blind-pack")
    blind.add_argument("--suite", required=True)
    blind.add_argument("--records-a", required=True)
    blind.add_argument("--records-b", required=True)
    blind.add_argument("--manifest-a", required=True)
    blind.add_argument("--manifest-b", required=True)
    blind.add_argument("--checkpoint-id", required=True)
    blind.add_argument("--output", required=True)
    blind.add_argument("--blind-seed", type=int, required=True)
    blind.add_argument("--candidate-a", default="candidate_a")
    blind.add_argument("--candidate-b", default="candidate_b")
    blind.add_argument("--seeds", type=int, nargs=3, default=(17, 29, 43))
    blind.add_argument("--max-new-tokens", type=int, default=128)
    blind.add_argument("--temperature", type=float, default=0.8)
    blind.add_argument("--top-p", type=float, default=0.95)
    blind.set_defaults(handler=blind_command)
    evaluate = commands.add_parser("eval-loss")
    training_arguments(evaluate)
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--split", choices=("validation", "test"), required=True)
    evaluate.add_argument("--output", required=True)
    evaluate.set_defaults(handler=eval_loss_command)
    aggregate = commands.add_parser("blind-aggregate")
    aggregate.add_argument("--blind-pack", required=True)
    aggregate.add_argument("--key", required=True)
    aggregate.add_argument("--judgments", required=True)
    aggregate.add_argument("--candidate-a", required=True)
    aggregate.add_argument("--candidate-b", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.set_defaults(handler=blind_aggregate_command)
    cost = commands.add_parser("cost-report")
    cost.add_argument("--input", required=True)
    cost.add_argument("--output", required=True)
    cost.set_defaults(handler=cost_report_command)
    endpoint = commands.add_parser("endpoint-analyze")
    endpoint.add_argument("--snapshots", nargs="+", required=True)
    endpoint.add_argument("--model-config")
    endpoint.add_argument("--output", required=True)
    endpoint.set_defaults(handler=endpoint_analyze_command)
    analyze = commands.add_parser("analyze")
    training_arguments(analyze, run_policy_required=True)
    analyze.add_argument("--checkpoint", required=True)
    analyze.add_argument("--snapshots", nargs="+", required=True)
    analyze.add_argument("--trajectory-config", required=True)
    analyze.add_argument("--target-step", type=int, required=True)
    analyze.add_argument("--scope", choices=("online", "level1", "level2"), default="online")
    analyze.add_argument("--reference-manifest")
    analyze.add_argument("--target-manifest")
    analyze.add_argument("--method", choices=("linear", "low-rank"), default="linear")
    analyze.add_argument("--continued-baseline-snapshot")
    analyze.add_argument("--apply", action="store_true")
    analyze.add_argument("--output-dir", required=True)
    analyze.set_defaults(handler=analyze_command)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.handler(args))
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        raise SystemExit(f"poetry50m: {error}") from error


if __name__ == "__main__":
    raise SystemExit(main())
