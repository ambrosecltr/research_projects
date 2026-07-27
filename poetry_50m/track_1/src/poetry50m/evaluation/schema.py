"""Versioned schemas for fixed evaluation, generation, and blind judgment."""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Literal


def _text(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.replace("\r\n", "\n").replace("\r", "\n")


def _atomic_write_text(path: Path, payload: str) -> None:
    """Durably replace one text artifact without exposing a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary.unlink(missing_ok=True)


@dataclass(frozen=True, slots=True)
class PromptCase:
    case_id: str
    prompt: str
    keywords: tuple[str, ...]
    expected_stanza_count: int | None = None
    partition: str = "evaluation"

    def __post_init__(self) -> None:
        _text("case_id", self.case_id)
        _text("prompt", self.prompt)
        if (
            not isinstance(self.keywords, tuple)
            or not self.keywords
            or any(not isinstance(keyword, str) or not keyword.strip() for keyword in self.keywords)
        ):
            raise ValueError("prompt cases require at least one non-empty keyword")
        if self.expected_stanza_count is not None and (
            isinstance(self.expected_stanza_count, bool)
            or not isinstance(self.expected_stanza_count, int)
            or self.expected_stanza_count < 1
        ):
            raise ValueError("expected_stanza_count must be positive")
        if not isinstance(self.partition, str) or self.partition not in {
            "development",
            "evaluation",
        }:
            raise ValueError("prompt partition must be development or evaluation")


@dataclass(frozen=True, slots=True)
class PromptSuite:
    suite_id: str
    version: int
    cases: tuple[PromptCase, ...]

    def __post_init__(self) -> None:
        _text("suite_id", self.suite_id)
        if (
            isinstance(self.version, bool)
            or not isinstance(self.version, int)
            or self.version < 1
            or not isinstance(self.cases, tuple)
            or not self.cases
            or any(not isinstance(case, PromptCase) for case in self.cases)
        ):
            raise ValueError("suite version and cases must be non-empty")
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("prompt case IDs must be unique")
        if (
            self.suite_id.startswith("track1-")
            and sum(case.partition == "evaluation" for case in self.cases) != 40
        ):
            raise ValueError("fixed suite requires exactly 40 evaluation prompts")
        if (
            self.suite_id.startswith("track1-")
            and sum(case.partition == "development" for case in self.cases) != 10
        ):
            raise ValueError("fixed suite requires exactly 10 development prompts")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> PromptSuite:
        if any(not isinstance(key, str) for key in value):
            raise ValueError("prompt suite keys must be strings")
        allowed = {"suite_id", "version", "cases"}
        unknown = set(value).difference(allowed)
        if unknown:
            raise ValueError(f"unknown prompt suite keys: {sorted(unknown)!r}")
        suite_id = value.get("suite_id")
        version = value.get("version")
        raw_cases = value.get("cases")
        if (
            not isinstance(suite_id, str)
            or isinstance(version, bool)
            or not isinstance(version, int)
        ):
            raise ValueError("prompt suite requires a string ID and integer version")
        if not isinstance(raw_cases, list):
            raise ValueError("prompt suite cases must be a list")
        cases: list[PromptCase] = []
        for case in raw_cases:
            if not isinstance(case, dict) or not all(isinstance(key, str) for key in case):
                raise ValueError("prompt suite cases must be objects with string keys")
            case_allowed = {
                "case_id",
                "prompt",
                "keywords",
                "expected_stanza_count",
                "partition",
            }
            unknown_case = set(case).difference(case_allowed)
            if unknown_case:
                raise ValueError(f"unknown prompt case keys: {sorted(unknown_case)!r}")
            case_id = case.get("case_id")
            prompt = case.get("prompt")
            keywords = case.get("keywords")
            stanza_count = case.get("expected_stanza_count")
            partition = case.get("partition", "evaluation")
            if (
                not isinstance(case_id, str)
                or not isinstance(prompt, str)
                or not isinstance(keywords, list)
                or not all(isinstance(keyword, str) for keyword in keywords)
                or (
                    stanza_count is not None
                    and (isinstance(stanza_count, bool) or not isinstance(stanza_count, int))
                )
                or not isinstance(partition, str)
            ):
                raise ValueError("prompt suite case has invalid field types")
            cases.append(PromptCase(case_id, prompt, tuple(keywords), stanza_count, partition))
        return cls(suite_id, version, tuple(cases))

    def save(self, path: Path) -> None:
        _atomic_write_text(
            path, json.dumps(asdict(self), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        )

    @classmethod
    def load(cls, path: Path) -> PromptSuite:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("prompt suite must be a JSON object")
        return cls.from_mapping(value)


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    request_id: str
    suite_id: str
    suite_version: int
    case_id: str
    prompt: str
    checkpoint_id: str
    seed: int
    max_new_tokens: int
    temperature: float
    top_p: float
    partition: str = "evaluation"

    def __post_init__(self) -> None:
        for name in ("request_id", "suite_id", "case_id", "prompt", "checkpoint_id"):
            _text(name, getattr(self, name))
        _validate_generation_numbers(
            self.suite_version,
            self.seed,
            self.max_new_tokens,
            self.temperature,
            self.top_p,
        )
        if self.partition not in {"development", "evaluation"}:
            raise ValueError("generation request partition is invalid")


def _validate_generation_numbers(
    suite_version: int,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> None:
    integers = (suite_version, seed, max_new_tokens)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in integers):
        raise ValueError("generation integer settings must be non-boolean integers")
    if suite_version < 1 or seed < 0 or max_new_tokens < 1:
        raise ValueError("invalid generation request integer")
    floats = (temperature, top_p)
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
        for value in floats
    ):
        raise ValueError("generation sampling settings must be finite numbers")
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be positive and top_p in (0, 1]")


def generation_requests(
    suite: PromptSuite,
    *,
    checkpoint_id: str,
    seed: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    partition: str = "evaluation",
) -> tuple[GenerationRequest, ...]:
    _validate_generation_numbers(suite.version, seed, max_new_tokens, temperature, top_p)
    requests: list[GenerationRequest] = []
    for case in suite.cases:
        if case.partition != partition:
            continue
        identity = {
            "suite_id": suite.suite_id,
            "suite_version": suite.version,
            "partition": case.partition,
            "case_id": case.case_id,
            "prompt": case.prompt,
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        stable_id = sha256(
            json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()[:24]
        requests.append(
            GenerationRequest(
                request_id=stable_id,
                suite_id=suite.suite_id,
                suite_version=suite.version,
                case_id=case.case_id,
                prompt=case.prompt,
                checkpoint_id=checkpoint_id,
                seed=seed,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                partition=case.partition,
            )
        )
    return tuple(requests)


def multi_seed_generation_requests(
    suite: PromptSuite,
    *,
    checkpoint_id: str,
    seeds: tuple[int, int, int],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[GenerationRequest, ...]:
    """Expand every fixed prompt across exactly three registered sampling seeds."""
    if len(set(seeds)) != 3:
        raise ValueError("evaluation requires exactly three distinct seeds")
    return tuple(
        request
        for seed in seeds
        for request in generation_requests(
            suite,
            checkpoint_id=checkpoint_id,
            seed=seed,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
        )
    )


def save_generation_manifest(path: Path, requests: Iterable[GenerationRequest]) -> None:
    ordered = sorted(requests, key=lambda request: request.request_id)
    request_ids = [request.request_id for request in ordered]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("generation manifest request IDs must be unique")
    _atomic_write_text(
        path,
        "".join(json.dumps(asdict(item), sort_keys=True) + "\n" for item in ordered),
    )


@dataclass(frozen=True, slots=True)
class BlindComparison:
    comparison_id: str
    request_id: str
    case_id: str
    seed: int
    left_label: str
    right_label: str
    left_text: str
    right_text: str

    def __post_init__(self) -> None:
        for name in (
            "comparison_id",
            "request_id",
            "case_id",
            "left_label",
            "right_label",
            "left_text",
            "right_text",
        ):
            _text(name, getattr(self, name))
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("blind comparison seed must be a non-negative integer")
        if (self.left_label, self.right_label) != ("A", "B"):
            raise ValueError("blind labels must be exactly A and B")


@dataclass(frozen=True, slots=True)
class BlindComparisonPack:
    comparisons: tuple[BlindComparison, ...]
    unblinding_key: Mapping[str, Mapping[str, str]]
    candidate_a_id: str = "candidate_a"
    candidate_b_id: str = "candidate_b"

    def __post_init__(self) -> None:
        _text("candidate_a_id", self.candidate_a_id)
        _text("candidate_b_id", self.candidate_b_id)
        if self.candidate_a_id == self.candidate_b_id:
            raise ValueError("blind candidate identities must differ")
        if (
            not isinstance(self.comparisons, tuple)
            or not self.comparisons
            or any(not isinstance(item, BlindComparison) for item in self.comparisons)
        ):
            raise ValueError("blind comparisons must be a non-empty tuple")
        comparison_ids = [item.comparison_id for item in self.comparisons]
        if len(comparison_ids) != len(set(comparison_ids)):
            raise ValueError("blind comparison IDs must be unique")
        if not isinstance(self.unblinding_key, Mapping):
            raise ValueError("unblinding key must be a mapping")
        if set(self.unblinding_key) != set(comparison_ids):
            raise ValueError("unblinding key must exactly cover blind comparisons")
        expected_candidates = {self.candidate_a_id, self.candidate_b_id}
        for comparison_id, mapping in self.unblinding_key.items():
            if not isinstance(comparison_id, str) or not isinstance(mapping, Mapping):
                raise ValueError("unblinding entries must be string-keyed mappings")
            if set(mapping) != {"A", "B"} or any(
                not isinstance(candidate_id, str) for candidate_id in mapping.values()
            ):
                raise ValueError("each unblinding entry must contain exactly A and B")
            if set(mapping.values()) != expected_candidates:
                raise ValueError("each unblinding entry must bijectively map both candidates")

    def save_blind(self, path: Path) -> None:
        _atomic_write_text(
            path,
            "".join(
                json.dumps(asdict(item), ensure_ascii=False, sort_keys=True) + "\n"
                for item in self.comparisons
            ),
        )

    def save_unblinding_key(self, path: Path) -> None:
        _atomic_write_text(path, json.dumps(self.unblinding_key, indent=2, sort_keys=True) + "\n")


def blind_comparison_pack(
    *,
    requests: Iterable[GenerationRequest],
    outputs_a: Mapping[str, str],
    outputs_b: Mapping[str, str],
    blind_seed: int,
    partition: str = "evaluation",
    candidate_a_id: str = "candidate_a",
    candidate_b_id: str = "candidate_b",
) -> BlindComparisonPack:
    """Blind exact request identities, including all three fixed sampling seeds."""
    if isinstance(blind_seed, bool) or not isinstance(blind_seed, int) or blind_seed < 0:
        raise ValueError("blind seed must be a non-negative integer")
    selected = tuple(request for request in requests if request.partition == partition)
    request_ids = {request.request_id for request in selected}
    if (
        not selected
        or len(request_ids) != len(selected)
        or set(outputs_a) != request_ids
        or set(outputs_b) != request_ids
    ):
        raise ValueError("outputs must exactly cover selected generation request IDs on both sides")
    comparisons: list[BlindComparison] = []
    key: dict[str, dict[str, str]] = {}
    for request in sorted(selected, key=lambda item: item.request_id):
        digest = sha256(f"{blind_seed}\0{request.request_id}".encode()).digest()
        swapped = digest[0] % 2 == 1
        first, second = (
            (outputs_b[request.request_id], outputs_a[request.request_id])
            if swapped
            else (outputs_a[request.request_id], outputs_b[request.request_id])
        )
        comparison_id = sha256(
            f"{blind_seed}\0{request.request_id}\0{first}\0{second}".encode()
        ).hexdigest()[:24]
        comparisons.append(
            BlindComparison(
                comparison_id,
                request.request_id,
                request.case_id,
                request.seed,
                "A",
                "B",
                first,
                second,
            )
        )
        key[comparison_id] = {
            "A": candidate_a_id if not swapped else candidate_b_id,
            "B": candidate_b_id if not swapped else candidate_a_id,
        }
    return BlindComparisonPack(tuple(comparisons), key, candidate_a_id, candidate_b_id)


JudgmentChoice = Literal["A", "B", "tie"]


@dataclass(frozen=True, slots=True)
class BlindJudgment:
    """One completed blind rubric, with a tie allowed for every criterion."""

    comparison_id: str
    prompt_relevance: JudgmentChoice
    poetic_quality: JudgmentChoice
    image_music: JudgmentChoice
    degeneration: JudgmentChoice
    notes: str = ""

    def __post_init__(self) -> None:
        _text("comparison_id", self.comparison_id)
        for choice in (
            self.prompt_relevance,
            self.poetic_quality,
            self.image_music,
            self.degeneration,
        ):
            if not isinstance(choice, str) or choice not in {"A", "B", "tie"}:
                raise ValueError("blind judgment choices must be A, B, or tie")
        if not isinstance(self.notes, str):
            raise ValueError("blind judgment notes must be a string")


@dataclass(frozen=True, slots=True)
class CriterionTally:
    criterion: str
    candidate_a_wins: int
    candidate_b_wins: int
    ties: int

    def __post_init__(self) -> None:
        if self.criterion not in {
            "prompt_relevance",
            "poetic_quality",
            "image_music",
            "degeneration",
        }:
            raise ValueError("unknown blind-judgment criterion")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.candidate_a_wins, self.candidate_b_wins, self.ties)
        ):
            raise ValueError("criterion tallies must be non-negative integers")


def aggregate_blind_judgments(
    pack: BlindComparisonPack, judgments: Iterable[BlindJudgment]
) -> tuple[CriterionTally, ...]:
    """Unblind rubric choices into candidate win/loss/tie totals with complete coverage."""
    key = pack.unblinding_key
    rows = tuple(judgments)
    if any(not isinstance(judgment, BlindJudgment) for judgment in rows):
        raise ValueError("judgments must contain BlindJudgment rows")
    comparison_ids = [judgment.comparison_id for judgment in rows]
    if len(comparison_ids) != len(set(comparison_ids)):
        raise ValueError("duplicate blind judgment comparison ID")
    values = {judgment.comparison_id: judgment for judgment in rows}
    if set(values) != set(key):
        raise ValueError("judgments must cover every comparison exactly once")
    criteria = ("prompt_relevance", "poetic_quality", "image_music", "degeneration")
    totals: list[CriterionTally] = []
    for criterion in criteria:
        a_wins = b_wins = ties = 0
        for comparison_id, judgment in values.items():
            choice = getattr(judgment, criterion)
            if choice == "tie":
                ties += 1
            elif key[comparison_id][choice] == pack.candidate_a_id:
                a_wins += 1
            else:
                b_wins += 1
        totals.append(CriterionTally(criterion, a_wins, b_wins, ties))
    return tuple(totals)


@dataclass(frozen=True, slots=True)
class CostRecord:
    run_id: str
    checkpoint_id: str
    steps: int
    tokens: int
    wall_seconds: float
    accelerator_seconds: float | None
    estimated_cost_usd: float | None = None
    device_active_wall_seconds: float | None = None

    def __post_init__(self) -> None:
        _text("run_id", self.run_id)
        _text("checkpoint_id", self.checkpoint_id)
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0
            for value in (self.steps, self.tokens)
        ):
            raise ValueError("invalid cost record identity or counts")
        times = (
            self.wall_seconds,
            self.accelerator_seconds,
            self.device_active_wall_seconds,
        )
        if any(
            value is not None
            and (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            )
            for value in times
        ):
            raise ValueError("cost times must be finite numbers or null")
        if self.wall_seconds < 0 or any(
            value is not None and value < 0
            for value in (self.accelerator_seconds, self.device_active_wall_seconds)
        ):
            raise ValueError("cost times must be non-negative")
        if self.estimated_cost_usd is not None and (
            isinstance(self.estimated_cost_usd, bool)
            or not isinstance(self.estimated_cost_usd, (int, float))
            or not math.isfinite(self.estimated_cost_usd)
            or self.estimated_cost_usd < 0
        ):
            raise ValueError("estimated cost must be finite and non-negative")
