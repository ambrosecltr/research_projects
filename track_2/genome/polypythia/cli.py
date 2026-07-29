"""Pythia source and deterministic evaluation commands.

The failed V4 decoder and compiler commands are available only through
``scripts/legacy_polypythia_v4.py``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import typer

from ..io import read_yaml
from .catalog import load_round_one_catalog
from .evaluate import (
    evaluate_lm_harness_revealed_prediction,
    evaluate_revealed_prediction,
    materialize_wikitext_evaluation,
)
from .hub import (
    build_source_plan,
    load_source_plan,
    materialize_source_plan,
    save_source_plan,
)
from .lives import load_canonical_life_corpus

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Pythia source pinning and deterministic evaluation; no V4 training",
)


def _config(path: Path) -> Mapping[str, Any]:
    value = read_yaml(path)
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise typer.BadParameter("Pythia config must be a mapping with string keys")
    return value


def _section(value: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    section = value.get(name)
    if not isinstance(section, Mapping) or any(not isinstance(key, str) for key in section):
        raise typer.BadParameter(f"Pythia config section {name} must be a mapping")
    return section


@app.command("plan-sources")
def plan_sources_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
) -> None:
    """Pin source revisions and files without downloading model endpoints."""
    catalog = load_round_one_catalog(config)
    result = save_source_plan(build_source_plan(catalog), output)
    typer.echo(json.dumps(result, indent=2))


@app.command("materialize-sources")
def materialize_sources_command(
    plan: Path = typer.Option(..., exists=True, readable=True),
    output_root: Path = typer.Option(...),
    splits: str = typer.Option("training,development,hidden"),
    reveal_hidden: bool = typer.Option(False, "--reveal-hidden"),
    prediction_seal: Path | None = typer.Option(None, exists=True, readable=True),
    runtime_execution: Path | None = typer.Option(None, exists=True, readable=True),
    max_workers: int = typer.Option(8, min=1),
) -> None:
    """Materialize pinned sources while enforcing hidden-endpoint reveal controls."""
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


@app.command("prepare-evaluation-texts")
def prepare_evaluation_texts_command(
    config: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    cache_dir: Path = typer.Option(...),
) -> None:
    """Materialize the pinned Wikitext evaluation corpus."""
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


@app.command("evaluate-revealed")
def evaluate_revealed_command(
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
    """Compare W0, a sealed Runtime output, and a revealed WT."""
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
    """Run the pinned LM Evaluation Harness suite on matched model states."""
    evaluation = _section(_config(config), "evaluation")
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
