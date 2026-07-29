from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

from ..hashing import sha256_json
from ..io import read_yaml

LifeSplit = Literal["training", "development", "hidden"]
_VALID_SPLITS = {"training", "development", "hidden"}


@dataclass(frozen=True)
class CheckpointPolicy:
    steps: tuple[int, ...]
    initial_step: int
    final_step: int

    def __post_init__(self) -> None:
        if not self.steps:
            raise ValueError("checkpoint policy cannot be empty")
        if any(
            isinstance(step, bool) or not isinstance(step, int) or step < 0 for step in self.steps
        ):
            raise ValueError("checkpoint steps must be non-negative integers")
        if tuple(sorted(set(self.steps))) != self.steps:
            raise ValueError("checkpoint steps must be unique and increasing")
        if self.initial_step != self.steps[0]:
            raise ValueError("initial_step must be the first checkpoint")
        if self.final_step != self.steps[-1]:
            raise ValueError("final_step must be the last checkpoint")

    @staticmethod
    def branch(step: int) -> str:
        return f"step{step}"


@dataclass(frozen=True)
class PolyPythiaLife:
    run_id: str
    seed: int
    data_order_seed: int
    repository: str
    split: LifeSplit

    def __post_init__(self) -> None:
        if not self.run_id:
            raise ValueError("run_id cannot be empty")
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError(f"invalid seed for {self.run_id}")
        if (
            isinstance(self.data_order_seed, bool)
            or not isinstance(self.data_order_seed, int)
            or self.data_order_seed < 0
        ):
            raise ValueError(f"invalid data_order_seed for {self.run_id}")
        if "/" not in self.repository:
            raise ValueError(f"repository must be a full Hugging Face ID: {self.repository}")
        if self.split not in _VALID_SPLITS:
            raise ValueError(f"invalid split for {self.run_id}: {self.split}")

    @property
    def endpoint_visible_during_training(self) -> bool:
        return self.split != "hidden"


@dataclass(frozen=True)
class RoundOneCatalog:
    experiment_id: str
    family: str
    architecture: str
    model_size: str
    checkpoints: CheckpointPolicy
    lives: tuple[PolyPythiaLife, ...]
    dataset_repository: str
    corpus_id: str
    corpus_variant: str
    tokenizer_source: str
    training_recipe: dict[str, Any]
    source_config_sha256: str

    def __post_init__(self) -> None:
        if len(self.lives) != 10:
            raise ValueError(f"Round One requires ten independent lives, got {len(self.lives)}")
        run_ids = [life.run_id for life in self.lives]
        repositories = [life.repository for life in self.lives]
        if len(set(run_ids)) != len(run_ids):
            raise ValueError("Round One run IDs must be unique")
        if len(set(repositories)) != len(repositories):
            raise ValueError("Round One repositories must be unique")
        by_split = {
            split: [life for life in self.lives if life.split == split] for split in _VALID_SPLITS
        }
        if len(by_split["training"]) != 8:
            raise ValueError("Round One requires eight training lives")
        if len(by_split["development"]) != 1:
            raise ValueError("Round One requires one development life")
        if len(by_split["hidden"]) != 1:
            raise ValueError("Round One requires one hidden life")
        seed_zero = next((life for life in self.lives if life.run_id == "pythia-14m-seed0"), None)
        if seed_zero is None or seed_zero.repository != "EleutherAI/pythia-14m":
            raise ValueError("seed 0 must use the standard non-deduplicated EleutherAI/pythia-14m")
        if any("deduped" in life.repository.lower() for life in self.lives):
            raise ValueError("PolyPythia Round One must not mix in a deduplicated Pile model")

    def lives_for(self, split: LifeSplit) -> tuple[PolyPythiaLife, ...]:
        return tuple(life for life in self.lives if life.split == split)

    @property
    def hidden_life(self) -> PolyPythiaLife:
        return self.lives_for("hidden")[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "family": self.family,
            "architecture": self.architecture,
            "model_size": self.model_size,
            "checkpoint_steps": list(self.checkpoints.steps),
            "initial_step": self.checkpoints.initial_step,
            "final_step": self.checkpoints.final_step,
            "lives": [
                {
                    "run_id": life.run_id,
                    "seed": life.seed,
                    "data_order_seed": life.data_order_seed,
                    "repository": life.repository,
                    "split": life.split,
                }
                for life in self.lives
            ],
            "dataset_repository": self.dataset_repository,
            "corpus_id": self.corpus_id,
            "corpus_variant": self.corpus_variant,
            "tokenizer_source": self.tokenizer_source,
            "training_recipe": self.training_recipe,
            "source_config_sha256": self.source_config_sha256,
        }


def _mapping(value: object, *, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{field} must be a mapping with string keys")
    return value


def _sequence(value: object, *, field: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    return value


def _integer(value: object, *, field: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise TypeError(f"{field} must be an integer greater than or equal to {minimum}")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise TypeError(f"{field} must be a non-empty string")
    return value


def _life_split(value: object, *, field: str) -> LifeSplit:
    if value not in _VALID_SPLITS:
        raise ValueError(f"{field} must be one of {sorted(_VALID_SPLITS)}")
    return cast(LifeSplit, value)


def load_round_one_catalog(path: str | Path) -> RoundOneCatalog:
    value = _mapping(read_yaml(path), field="config")
    experiment = _mapping(value.get("experiment"), field="experiment")
    source = _mapping(value.get("source"), field="source")
    range_config = _mapping(source.get("checkpoint_range"), field="source.checkpoint_range")
    explicit_steps = [
        _integer(step, field="source.checkpoint_steps[]")
        for step in _sequence(
            source.get("checkpoint_steps"),
            field="source.checkpoint_steps",
        )
    ]
    range_steps = list(
        range(
            _integer(range_config["start"], field="source.checkpoint_range.start"),
            _integer(range_config["stop"], field="source.checkpoint_range.stop") + 1,
            _integer(
                range_config["step"],
                field="source.checkpoint_range.step",
                minimum=1,
            ),
        )
    )
    steps = tuple(sorted(set(explicit_steps + range_steps)))
    raw_lives = _sequence(source.get("lives"), field="source.lives")
    lives = []
    for index, item in enumerate(raw_lives):
        field = f"source.lives[{index}]"
        life = _mapping(item, field=field)
        lives.append(
            PolyPythiaLife(
                run_id=_string(life["run_id"], field=f"{field}.run_id"),
                seed=_integer(life["seed"], field=f"{field}.seed"),
                data_order_seed=_integer(
                    life["data_order_seed"],
                    field=f"{field}.data_order_seed",
                ),
                repository=_string(
                    life["repository"],
                    field=f"{field}.repository",
                ),
                split=_life_split(life["split"], field=f"{field}.split"),
            )
        )
    dataset = _mapping(source.get("dataset"), field="source.dataset")
    catalog = RoundOneCatalog(
        experiment_id=_string(experiment["id"], field="experiment.id"),
        family=_string(source["family"], field="source.family"),
        architecture=_string(source["architecture"], field="source.architecture"),
        model_size=_string(source["model_size"], field="source.model_size"),
        checkpoints=CheckpointPolicy(
            steps=steps,
            initial_step=_integer(source["initial_step"], field="source.initial_step"),
            final_step=_integer(source["final_step"], field="source.final_step"),
        ),
        lives=tuple(lives),
        dataset_repository=_string(dataset["repository"], field="source.dataset.repository"),
        corpus_id=_string(dataset["corpus"], field="source.dataset.corpus"),
        corpus_variant=_string(dataset["variant"], field="source.dataset.variant"),
        tokenizer_source=_string(source["tokenizer_source"], field="source.tokenizer_source"),
        training_recipe=dict(
            _mapping(source.get("training_recipe"), field="source.training_recipe")
        ),
        source_config_sha256=sha256_json(
            {
                "experiment_id": experiment["id"],
                "source": dict(source),
            }
        ),
    )
    if len(catalog.checkpoints.steps) != 154:
        raise ValueError(
            f"PolyPythia life must contain 154 checkpoints, got {len(catalog.checkpoints.steps)}"
        )
    return catalog
