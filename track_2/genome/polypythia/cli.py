from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import typer

from ..io import read_yaml
from ..neural import (
    BlockDecoderConfig,
    LatentCodeFitConfig,
    PredictiveCompilerTrainingConfig,
    SharedDecoderTrainingConfig,
    fit_genome_code_with_frozen_decoder,
    predict_hidden_genome,
    train_predictive_compiler,
    train_shared_decoder,
)
from .catalog import load_round_one_catalog
from .evaluate import (
    evaluate_lm_harness_revealed_prediction,
    evaluate_revealed_prediction,
    evaluate_shared_decoder_corpus,
    execute_hidden_prediction,
    materialize_wikitext_evaluation,
)
from .evidence import EvidenceConfig
from .hub import (
    build_source_plan,
    load_source_plan,
    materialize_source_plan,
    save_source_plan,
)
from .lives import load_canonical_life_corpus
from .prepare import prepare_canonical_lives

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="PolyPythia Round One pipeline",
)


def _config(path: Path) -> Mapping[str, Any]:
    value = read_yaml(path)
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise typer.BadParameter("Round One config must be a mapping with string keys")
    return value


def _section(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = value.get(name)
    if not isinstance(section, Mapping) or any(not isinstance(key, str) for key in section):
        raise typer.BadParameter(f"Round One config section {name} must be a mapping")
    return section


@app.command("plan")
def plan_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
) -> None:
    """Pin all ten lives, 154 checkpoints per life, data-order maps, and tokenizer files."""
    catalog = load_round_one_catalog(config)
    result = save_source_plan(build_source_plan(catalog), output)
    typer.echo(json.dumps(result, indent=2))


@app.command("download")
def download_command(
    plan: Path = typer.Option(..., exists=True, readable=True),
    output_root: Path = typer.Option(...),
    splits: str = typer.Option("training,development,hidden"),
    reveal_hidden: bool = typer.Option(False, "--reveal-hidden"),
    prediction_seal: Path | None = typer.Option(None, exists=True, readable=True),
    runtime_execution: Path | None = typer.Option(None, exists=True, readable=True),
    max_workers: int = typer.Option(8, min=1),
) -> None:
    """Download required W0/WT endpoints while enforcing the hidden-endpoint seal."""
    selected_splits = tuple(item.strip() for item in splits.split(",") if item.strip())
    if not selected_splits:
        raise typer.BadParameter("at least one split is required")
    result = materialize_source_plan(
        load_source_plan(plan),
        output_root=output_root,
        splits=selected_splits,
        reveal_hidden=reveal_hidden,
        prediction_seal=prediction_seal,
        runtime_execution=runtime_execution,
        max_workers=max_workers,
    )
    typer.echo(
        json.dumps(
            {
                "checkpoint_count": result["checkpoint_count"],
                "checkpoint_bytes": result["checkpoint_bytes"],
                "reveal_hidden": result["reveal_hidden"],
            },
            indent=2,
        )
    )


@app.command("prepare")
def prepare_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    plan: Path = typer.Option(..., exists=True, readable=True),
    receipt: Path = typer.Option(..., exists=True, readable=True),
    download_root: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(...),
) -> None:
    """Convert downloaded GPT-NeoX endpoints into strict canonical model-life artifacts."""
    value = _config(config)
    compiler = _section(value, "compiler")
    result = prepare_canonical_lives(
        load_source_plan(plan),
        receipt_path=receipt,
        download_root=download_root,
        output_root=output,
        evidence_config=EvidenceConfig(
            initialization_sketch_dim_per_role=int(compiler["initialization_sketch_dim_per_role"]),
            digest_vector_dim=int(compiler["digest_vector_dim"]),
            seed=int(compiler["seed"]),
        ),
    )
    typer.echo(json.dumps(result, indent=2))


def _decoder_configs(
    value: Mapping[str, Any],
    *,
    device: str,
) -> tuple[BlockDecoderConfig, SharedDecoderTrainingConfig]:
    section = _section(value, "decoder")
    block_rows = int(section["block_rows"])
    block_cols = int(section["block_cols"])
    coordinate_frequencies = int(section["coordinate_frequencies"])
    if coordinate_frequencies < 0:
        raise typer.BadParameter("decoder.coordinate_frequencies must be non-negative")
    decoder = BlockDecoderConfig(
        block_rows=block_rows,
        block_cols=block_cols,
        global_code_dim=int(section["global_code_dim"]),
        layer_code_dim=int(section["layer_code_dim"]),
        tensor_code_dim=int(section["tensor_code_dim"]),
        block_code_dim=int(section["block_code_dim"]),
        role_embedding_dim=int(section["role_embedding_dim"]),
        feature_dim=7 + 4 * coordinate_frequencies + block_rows * block_cols,
        hidden_dim=int(section["hidden_dim"]),
        depth=int(section["depth"]),
    )
    training = SharedDecoderTrainingConfig(
        seed=int(section["seed"]),
        updates=int(section["updates"]),
        batch_size=int(section["batch_size"]),
        learning_rate=float(section["learning_rate"]),
        weight_decay=float(section["weight_decay"]),
        code_weight_decay=float(section["code_weight_decay"]),
        grad_clip_norm=float(section["grad_clip_norm"]),
        device=device,
        log_every=max(1, int(section["updates"]) // 100),
    )
    return decoder, training


@app.command("train-decoder")
def train_decoder_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    corpus: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(...),
    device: str = typer.Option("cuda"),
) -> None:
    """Train one shared Neural Genome Decoder and one fitted code per training life."""
    canonical = load_canonical_life_corpus(corpus)
    decoder_config, training_config = _decoder_configs(_config(config), device=device)
    result = train_shared_decoder(
        canonical.for_split("training"),
        tensor_specs=canonical.inventory,
        tied_groups=canonical.tied_groups,
        output_path=output,
        decoder_config=decoder_config,
        training_config=training_config,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("fit-development-code")
def fit_development_code_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    corpus: Path = typer.Option(..., exists=True, file_okay=False),
    shared_decoder: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(...),
    device: str = typer.Option("cuda"),
) -> None:
    """Fit only a development genome code while the shared decoder stays frozen."""
    value = _config(config)
    section = _section(value, "development_code_fit")
    canonical = load_canonical_life_corpus(corpus)
    development = canonical.for_split("development")
    if len(development) != 1:
        raise typer.BadParameter("canonical corpus must contain one development life")
    result = fit_genome_code_with_frozen_decoder(
        development[0],
        tensor_specs=canonical.inventory,
        tied_groups=canonical.tied_groups,
        shared_decoder_path=shared_decoder,
        output_path=output,
        config=LatentCodeFitConfig(
            seed=int(section["seed"]),
            updates=int(section["updates"]),
            batch_size=int(section["batch_size"]),
            learning_rate=float(section["learning_rate"]),
            weight_decay=float(section["weight_decay"]),
            grad_clip_norm=float(section["grad_clip_norm"]),
            device=device,
            log_every=max(1, int(section["updates"]) // 100),
        ),
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("evaluate-decoder")
def evaluate_decoder_command(
    corpus: Path = typer.Option(..., exists=True, file_okay=False),
    shared_decoder: Path = typer.Option(..., exists=True, file_okay=False),
    development_code: Path = typer.Option(..., exists=True, file_okay=False),
    model_config: Path = typer.Option(..., exists=True, readable=True),
    tokenizer: Path = typer.Option(..., exists=True, file_okay=False),
    evaluation_texts: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    device: str = typer.Option("cuda"),
    sequence_length: int = typer.Option(512, min=8),
    batch_size: int = typer.Option(4, min=1),
    max_batches: int | None = typer.Option(None, min=1),
) -> None:
    """Audit shared-decoder reconstruction on all training lives and the development life."""
    canonical = load_canonical_life_corpus(corpus)
    development = canonical.for_split("development")
    if len(development) != 1:
        raise typer.BadParameter("canonical corpus must contain one development life")
    result = evaluate_shared_decoder_corpus(
        training_lives=canonical.for_split("training"),
        development_life=development[0],
        tensor_specs=canonical.inventory,
        tied_groups=canonical.tied_groups,
        shared_decoder_path=shared_decoder,
        development_code_path=development_code,
        config_path=model_config,
        tokenizer_path=tokenizer,
        evaluation_texts_path=evaluation_texts,
        output_path=output,
        device=device,
        sequence_length=sequence_length,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("train-compiler")
def train_compiler_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    corpus: Path = typer.Option(..., exists=True, file_okay=False),
    shared_decoder: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(...),
    device: str = typer.Option("cuda"),
) -> None:
    """Train the GENOME Compiler through the frozen decoder against endpoint Delta-T."""
    value = _config(config)
    section = _section(value, "compiler")
    canonical = load_canonical_life_corpus(corpus)
    development = canonical.for_split("development")
    if len(development) != 1:
        raise typer.BadParameter("canonical corpus must contain one development life")
    result = train_predictive_compiler(
        canonical.for_split("training"),
        development[0],
        tensor_specs=canonical.inventory,
        tied_groups=canonical.tied_groups,
        shared_decoder_path=shared_decoder,
        output_path=output,
        config=PredictiveCompilerTrainingConfig(
            seed=int(section["seed"]),
            updates=int(section["updates"]),
            batch_size=int(section["batch_size"]),
            learning_rate=float(section["learning_rate"]),
            weight_decay=float(section["weight_decay"]),
            rate_weight=float(section["rate_weight"]),
            hidden_dim=int(section["hidden_dim"]),
            depth=int(section["depth"]),
            device=device,
            log_every=max(1, int(section["updates"]) // 100),
        ),
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("predict-hidden")
def predict_hidden_command(
    corpus: Path = typer.Option(..., exists=True, file_okay=False),
    shared_decoder: Path = typer.Option(..., exists=True, file_okay=False),
    compiler: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(...),
    device: str = typer.Option("cuda"),
) -> None:
    """Compile the hidden seed in one shot without WT or an early trajectory."""
    canonical = load_canonical_life_corpus(corpus)
    hidden = canonical.for_split("hidden")
    if len(hidden) != 1:
        raise typer.BadParameter("canonical corpus must contain one hidden life")
    result = predict_hidden_genome(
        hidden[0],
        tensor_specs=canonical.inventory,
        tied_groups=canonical.tied_groups,
        shared_decoder_path=shared_decoder,
        compiler_path=compiler,
        output_path=output,
        device=device,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("execute-hidden")
def execute_hidden_command(
    corpus: Path = typer.Option(..., exists=True, file_okay=False),
    shared_decoder: Path = typer.Option(..., exists=True, file_okay=False),
    prediction: Path = typer.Option(..., exists=True, file_okay=False),
    model_config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    device: str = typer.Option("cuda"),
) -> None:
    """Decode and run the sealed hidden prediction before revealing WT."""
    canonical = load_canonical_life_corpus(corpus)
    hidden = canonical.for_split("hidden")
    if len(hidden) != 1:
        raise typer.BadParameter("canonical corpus must contain one hidden life")
    result = execute_hidden_prediction(
        hidden[0],
        tensor_specs=canonical.inventory,
        tied_groups=canonical.tied_groups,
        shared_decoder_path=shared_decoder,
        prediction_path=prediction,
        config_path=model_config,
        output_path=output,
        device=device,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("prepare-evaluation-texts")
def prepare_evaluation_texts_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    cache_dir: Path = typer.Option(...),
) -> None:
    """Materialize the exact pinned Wikitext test corpus used for functional comparison."""
    evaluation = _section(_config(config), "evaluation")
    corpus = evaluation.get("text_corpus")
    if not isinstance(corpus, Mapping):
        raise typer.BadParameter("evaluation.text_corpus must be a mapping")
    result = materialize_wikitext_evaluation(
        output_path=output,
        cache_dir=cache_dir,
        repository=str(corpus["repository"]),
        revision=str(corpus["revision"]),
        configuration=str(corpus["configuration"]),
        split=str(corpus["split"]),
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("evaluate-hidden")
def evaluate_hidden_command(
    sealed_corpus: Path = typer.Option(..., exists=True, file_okay=False),
    revealed_corpus: Path = typer.Option(..., exists=True, file_okay=False),
    runtime_execution: Path = typer.Option(..., exists=True, file_okay=False),
    model_config: Path = typer.Option(..., exists=True, readable=True),
    tokenizer: Path = typer.Option(..., exists=True, file_okay=False),
    evaluation_texts: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    device: str = typer.Option("cuda"),
    sequence_length: int = typer.Option(512, min=8),
    batch_size: int = typer.Option(4, min=1),
    max_batches: int | None = typer.Option(None, min=1),
) -> None:
    """Compare W0, predicted WT, and revealed WT under the matched hidden protocol."""
    sealed = load_canonical_life_corpus(sealed_corpus)
    revealed = load_canonical_life_corpus(revealed_corpus)
    sealed_hidden = sealed.for_split("hidden")
    revealed_hidden = revealed.for_split("hidden")
    if len(sealed_hidden) != 1 or len(revealed_hidden) != 1:
        raise typer.BadParameter("each corpus must contain one hidden life")
    result = evaluate_revealed_prediction(
        sealed_hidden_life=sealed_hidden[0],
        revealed_hidden_life=revealed_hidden[0],
        runtime_execution_path=runtime_execution,
        tensor_specs=sealed.inventory,
        config_path=model_config,
        tokenizer_path=tokenizer,
        evaluation_texts_path=evaluation_texts,
        output_path=output,
        device=device,
        sequence_length=sequence_length,
        batch_size=batch_size,
        max_batches=max_batches,
    )
    typer.echo(json.dumps(result, indent=2))


@app.command("evaluate-lm-harness")
def evaluate_lm_harness_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    sealed_corpus: Path = typer.Option(..., exists=True, file_okay=False),
    revealed_corpus: Path = typer.Option(..., exists=True, file_okay=False),
    runtime_execution: Path = typer.Option(..., exists=True, file_okay=False),
    model_config: Path = typer.Option(..., exists=True, readable=True),
    tokenizer: Path = typer.Option(..., exists=True, file_okay=False),
    output: Path = typer.Option(...),
    device: str = typer.Option("cuda"),
) -> None:
    """Run the pinned zero-shot task suite on W0, predicted WT, and true WT."""
    value = _config(config)
    evaluation = _section(value, "evaluation")
    lm_eval = evaluation.get("lm_eval")
    if not isinstance(lm_eval, Mapping):
        raise typer.BadParameter("evaluation.lm_eval must be a mapping")
    tasks = lm_eval.get("tasks")
    seeds = lm_eval.get("seeds")
    if not isinstance(tasks, list) or any(not isinstance(task, str) for task in tasks):
        raise typer.BadParameter("evaluation.lm_eval.tasks must be a list of strings")
    if not isinstance(seeds, Mapping):
        raise typer.BadParameter("evaluation.lm_eval.seeds must be a mapping")
    task_directory_value = lm_eval.get("task_directory")
    if not isinstance(task_directory_value, str) or not task_directory_value:
        raise typer.BadParameter("evaluation.lm_eval.task_directory must be a path")
    task_directory = (config.parent / task_directory_value).resolve(strict=True)
    batch_size_value = lm_eval.get("batch_size")
    if not isinstance(batch_size_value, (int, str)) or isinstance(batch_size_value, bool):
        raise typer.BadParameter("evaluation.lm_eval.batch_size must be an integer or string")
    sealed = load_canonical_life_corpus(sealed_corpus)
    revealed = load_canonical_life_corpus(revealed_corpus)
    sealed_hidden = sealed.for_split("hidden")
    revealed_hidden = revealed.for_split("hidden")
    if len(sealed_hidden) != 1 or len(revealed_hidden) != 1:
        raise typer.BadParameter("each corpus must contain one hidden life")
    result = evaluate_lm_harness_revealed_prediction(
        sealed_hidden_life=sealed_hidden[0],
        revealed_hidden_life=revealed_hidden[0],
        runtime_execution_path=runtime_execution,
        config_path=model_config,
        tokenizer_path=tokenizer,
        task_directory=task_directory,
        tasks=tasks,
        output_path=output,
        device=device,
        num_fewshot=int(lm_eval["num_fewshot"]),
        batch_size=batch_size_value,
        max_batch_size=int(lm_eval["max_batch_size"]),
        bootstrap_iters=int(lm_eval["bootstrap_iters"]),
        random_seed=int(seeds["python"]),
        numpy_random_seed=int(seeds["numpy"]),
        torch_random_seed=int(seeds["torch"]),
        fewshot_random_seed=int(seeds["fewshot"]),
    )
    typer.echo(json.dumps(result, indent=2))
