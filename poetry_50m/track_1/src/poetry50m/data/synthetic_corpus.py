"""Resumable synthetic generation through Cerebras or OpenAI-compatible APIs."""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from hashlib import sha256
from heapq import merge as merge_sorted
from pathlib import Path
from typing import BinaryIO, Literal, cast

from cerebras.cloud.sdk import Cerebras
from cerebras.cloud.sdk.types.chat.chat_completion import ChatCompletionResponse

from poetry50m.config import file_hash, load_mapping
from poetry50m.rate_limit import DualTokenBucket
from poetry50m.trajectory._persistence import atomic_write

from .artifacts import (
    read_pairings,
    read_prompt_records,
    read_thought_records,
    write_pairings,
    write_prompt_records,
    write_thought_records,
)
from .loaders import iter_manifest, write_manifest
from .schema import ContentBlock, PromptMethod, PromptRecord, Provenance, SourceDocument

GeneratorModel = str
ReasoningEffort = Literal["low", "medium", "high"]
Decision = Literal["accept", "reject"]
ResponseFormat = Literal["json-schema", "json-object", "none"]

WORD = re.compile(r"[\w']+", re.UNICODE)
GENERATION_LANES = (
    "concrete observation with precise sensory detail",
    "a narrative turn that changes the speaker's understanding",
    "formal constraint used naturally rather than mechanically",
    "conversational contemporary voice",
    "surreal but internally coherent imagery",
    "philosophical pressure grounded in an ordinary object",
    "place-based writing with specific physical movement",
    "compressed lyric with an emotional reversal",
)

SETTINGS = (
    "a laundromat during its final wash cycle",
    "a municipal swimming pool before opening",
    "the loading bay behind a grocery store",
    "a ferry terminal during a service delay",
    "a suburban garage during a power cut",
    "an all-night pharmacy near shift change",
    "a school gym after a local election count",
    "a repair cafe on a rainy Saturday",
    "a hospital car park at visiting-hour changeover",
    "a roadside fruit stall packing up",
    "a public library while the returns chute is jammed",
    "a train carriage being cleaned at the terminus",
    "a community garden after hail",
    "a takeaway kitchen just after closing",
    "a hardware shop during inventory",
    "a motel walkway before checkout",
)

OBJECT_PAIRS = (
    "one payment object and one item recovered from a dryer",
    "one piece of pool safety equipment and one forgotten personal item",
    "one damaged shipping container and one bruised perishable food",
    "one travel-information object and one rain-wet item of clothing",
    "one portable light source and one unfinished repair",
    "one dispensing-machine component and one packaged medicine",
    "one election-counting object and one piece of stored sports equipment",
    "one damaged fastener and one well-used drinking vessel",
    "one parking-related document and one item brought for a patient",
    "one object used to take payment and one packing material",
    "one library-account document and one object from the lost-property shelf",
    "one cleaning tool and one item a passenger left behind",
    "one storm-damaged plant support and one object holding melting hail",
    "one piece of closing equipment and one portion of unsold food",
    "one inventory tool and one mismatched piece of hardware",
    "one room-access object and one vending-machine component",
)

ACTIONS = (
    "someone decides whether to return an object",
    "two strangers cooperate without introducing themselves",
    "a routine task exposes a small lie",
    "someone repairs the wrong thing first",
    "a worker notices evidence left by the previous shift",
    "a child interprets an adult procedure literally",
    "someone rehearses a sentence and then says something else",
    "an interruption changes who is helping whom",
    "a minor spill forces a private decision into public view",
    "someone counts items to avoid answering a question",
    "an object changes hands twice",
    "a delayed departure becomes a deliberate choice",
    "someone follows a rule past the point where it helps",
    "an ordinary sound reveals that a person has returned",
    "a practical kindness is almost mistaken for criticism",
    "someone finishes a task another person abandoned",
)

PRESSURES = (
    "embarrassment without confession",
    "relief mixed with resentment",
    "care expressed as competent work",
    "a disagreement about what counts as waste",
    "the gap between being useful and being wanted",
    "impatience that gradually becomes attention",
    "an obligation that was never spoken aloud",
    "the cost of correcting someone in public",
    "a habit surviving after its reason is gone",
    "gratitude that cannot be comfortably voiced",
    "a private fear of becoming unreliable",
    "the difference between replacing and mending",
    "a promise inferred from repeated actions",
    "the awkwardness of accepting help",
    "a change noticed only through procedure",
    "affection hidden inside precise instructions",
)

FORMS = (
    "free verse with varied sentence lengths and no end rhyme",
    "a scene in tercets with restrained enjambment",
    "a compact dramatic monologue",
    "a list poem whose final item changes the meaning of the earlier items",
    "two unequal stanzas separated by a factual one-line hinge",
    "a narrative lyric with one brief line of dialogue",
    "a prose poem broken into 8 to 12 deliberate lines",
    "a poem organized by repeated physical actions, not repeated phrases",
)

PARTICIPANTS = (
    "a worker and a customer deciding who owns the found item",
    "two strangers of noticeably different ages",
    "coworkers who normally trust each other's records",
    "siblings who disagree about the practical task",
    "workers from consecutive shifts who do not overlap",
    "a child and an unrelated adult following a public procedure",
    "coworkers near the end of a shared shift",
    "a volunteer and someone reluctant to receive help",
    "a supervisor and a new employee",
    "a couple communicating mostly through tasks",
    "neighbors who know each other only by routine",
    "two former friends meeting unexpectedly",
    "a supervisor and a rule-conscious new employee",
    "one person alone, reacting to evidence that another has returned",
    "neighbors who often misread each other's tone",
    "workers from consecutive shifts who have never met",
)

STOCK_PHRASES = (
    "city exhales",
    "dance of",
    "echoes of",
    "for a heartbeat",
    "golden light",
    "heart of",
    "held its breath",
    "hidden life",
    "like a sigh",
    "promise of",
    "quiet whisper",
    "silver thread",
    "soft whisper",
    "stands still",
    "symphony of",
    "tapestry of",
    "time stood still",
    "world whispers",
)

BANNED_SYNTHETIC_WORDS = frozenset(
    {
        "breathes",
        "echo",
        "echoes",
        "flicker",
        "flickers",
        "ghost",
        "heartbeat",
        "hush",
        "lingering",
        "sigh",
        "sighing",
        "sighs",
        "silence",
        "soul",
        "symphony",
        "tapestry",
        "timeless",
        "whisper",
        "whispering",
        "whispers",
    }
)

GENERATOR_SYSTEM_PROMPT = """\
Create original, prompt-conditioned English poetry training examples.

Every example must:
- use an original short poem, never a quotation or continuation of a known poem;
- avoid imitating or naming any real author;
- contain 8 to 20 non-empty lines and roughly 70 to 180 words;
- respond concretely to its prompt rather than defaulting to generic stars, sea, dawn, or longing;
- use grammatical language, intentional line breaks, and no invented malformed words;
- avoid repeated lines, stock filler, explanatory notes, and title-only conditioning;
- never use these mode-collapse words: whisper, echo, sigh, silence, soul, symphony,
  tapestry, timeless, ghost, heartbeat, hush, lingering, flicker, or breathes;
- avoid prefab lyric language such as silver threads, hearts, the world holding its
  breath, a city exhaling, or an object making a promise;
- earn its emotional turn through observed action; never finish by explaining a lesson;
- contain no Markdown, bullets, decorative symbols, or backslashes at line endings;
- keep quotation marks balanced and every physical action plausible in the stated setting;
- provide three genuinely different prompts for the same poem: theme, imagery, and paraphrase;
- keep each prompt self-contained and suitable for a user asking a small poetry model.

The user supplies one concrete brief per requested example. Follow the briefs in
order, instantiate both object categories as specific setting-native items, make those
items affect the action, and do not swap settings between examples.
Return only the strict JSON object requested by the schema."""

CRITIC_SYSTEM_PROMPT = """\
You are the final gatekeeper for a small language model's training corpus, not a
supportive workshop reader. Judge only the supplied example. Most competent first
drafts should score 3 or lower. A 4 must be specific, controlled, memorable, and free
of prefab lyric language. A 5 is rare and publication-ready.

Reject generic emotional summaries, moral explanations, arbitrary metaphor stacking,
decorative surrealism without causal sense, repeated image families, mechanical rhyme,
Markdown artifacts, unbalanced quotation marks, or prohibited mode-collapse language.
Reject if the poem merely names the requested objects, places an object somewhere
implausible, or jumps between actions without a physically intelligible scene.
Also reject incoherence, degeneration, suspicious quotation, or named-author imitation.
Set decision to accept only when prompt adherence, coherence, craft, and originality
are all at least 4 and no rejection concern applies. Return only the strict JSON object."""


def _exact_object(value: object, *, name: str, required: set[str]) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be an object with string keys")
    actual = set(value)
    if actual != required:
        raise ValueError(f"{name} must contain exactly {sorted(required)}")
    return cast(dict[str, object], value)


def _required_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _required_integer(value: object, *, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _required_number(value: object, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _string_tuple(value: object, *, name: str, minimum: int = 1) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TypeError(f"{name} must be an array")
    result = tuple(_required_string(item, name=f"{name} item") for item in value)
    if len(result) < minimum:
        raise ValueError(f"{name} must contain at least {minimum} items")
    return result


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_bytes(path: Path, payload: bytes) -> None:
    def write(handle: BinaryIO) -> None:
        handle.write(payload)

    atomic_write(path, write)


def _write_json(path: Path, value: object) -> None:
    _write_bytes(path, (_canonical_json(value) + "\n").encode("utf-8"))


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    payload = "".join(f"{_canonical_json(record)}\n" for record in records).encode("utf-8")
    _write_bytes(path, payload)


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from error
            if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
                raise TypeError(f"{path}:{line_number} must be an object")
            records.append(cast(dict[str, object], value))
    return tuple(records)


@dataclass(frozen=True, slots=True)
class GeneratorLane:
    model: GeneratorModel
    weight: int
    temperature: float
    reasoning_effort: ReasoningEffort

    @classmethod
    def from_mapping(cls, value: object) -> GeneratorLane:
        data = _exact_object(
            value,
            name="generator lane",
            required={"model", "weight", "temperature", "reasoning_effort"},
        )
        model = _required_string(data["model"], name="generator model")
        effort = _required_string(data["reasoning_effort"], name="reasoning effort")
        if effort not in {"low", "medium", "high"}:
            raise ValueError(f"unsupported reasoning effort {effort}")
        temperature = _required_number(data["temperature"], name="temperature")
        if not 0.0 < temperature <= 2.0:
            raise ValueError("temperature must be in (0, 2]")
        return cls(
            model,
            _required_integer(data["weight"], name="generator weight"),
            temperature,
            cast(ReasoningEffort, effort),
        )


@dataclass(frozen=True, slots=True)
class QualityConfig:
    minimum_prompt_adherence: int
    minimum_coherence: int
    minimum_craft: int
    minimum_originality: int
    minimum_word_count: int
    maximum_word_count: int
    minimum_line_count: int
    maximum_line_count: int
    maximum_repeated_bigram_rate: float
    maximum_banned_word_count: int
    maximum_stock_phrase_count: int
    dedup_ngram_size: int

    @classmethod
    def from_mapping(cls, value: object) -> QualityConfig:
        fields = {
            "minimum_prompt_adherence",
            "minimum_coherence",
            "minimum_craft",
            "minimum_originality",
            "minimum_word_count",
            "maximum_word_count",
            "minimum_line_count",
            "maximum_line_count",
            "maximum_repeated_bigram_rate",
            "maximum_banned_word_count",
            "maximum_stock_phrase_count",
            "dedup_ngram_size",
        }
        data = _exact_object(value, name="quality config", required=fields)
        config = cls(
            minimum_prompt_adherence=_required_integer(
                data["minimum_prompt_adherence"], name="minimum_prompt_adherence"
            ),
            minimum_coherence=_required_integer(
                data["minimum_coherence"], name="minimum_coherence"
            ),
            minimum_craft=_required_integer(data["minimum_craft"], name="minimum_craft"),
            minimum_originality=_required_integer(
                data["minimum_originality"], name="minimum_originality"
            ),
            minimum_word_count=_required_integer(
                data["minimum_word_count"], name="minimum_word_count"
            ),
            maximum_word_count=_required_integer(
                data["maximum_word_count"], name="maximum_word_count"
            ),
            minimum_line_count=_required_integer(
                data["minimum_line_count"], name="minimum_line_count"
            ),
            maximum_line_count=_required_integer(
                data["maximum_line_count"], name="maximum_line_count"
            ),
            maximum_repeated_bigram_rate=_required_number(
                data["maximum_repeated_bigram_rate"],
                name="maximum_repeated_bigram_rate",
            ),
            maximum_banned_word_count=_required_integer(
                data["maximum_banned_word_count"],
                name="maximum_banned_word_count",
                minimum=0,
            ),
            maximum_stock_phrase_count=_required_integer(
                data["maximum_stock_phrase_count"],
                name="maximum_stock_phrase_count",
                minimum=0,
            ),
            dedup_ngram_size=_required_integer(
                data["dedup_ngram_size"], name="dedup_ngram_size", minimum=2
            ),
        )
        for name in (
            "minimum_prompt_adherence",
            "minimum_coherence",
            "minimum_craft",
            "minimum_originality",
        ):
            if getattr(config, name) > 5:
                raise ValueError(f"{name} must be <= 5")
        if config.maximum_word_count < config.minimum_word_count:
            raise ValueError("maximum_word_count must be >= minimum_word_count")
        if config.maximum_line_count < config.minimum_line_count:
            raise ValueError("maximum_line_count must be >= minimum_line_count")
        if not 0.0 <= config.maximum_repeated_bigram_rate <= 1.0:
            raise ValueError("maximum_repeated_bigram_rate must be in [0, 1]")
        return config


@dataclass(frozen=True, slots=True)
class SyntheticCorpusConfig:
    format_version: int
    seed: int
    examples_per_request: int
    generator_lanes: tuple[GeneratorLane, ...]
    critic_model: GeneratorModel
    critic_reasoning_effort: ReasoningEffort
    max_completion_tokens: int
    quality: QualityConfig

    @classmethod
    def load(cls, path: Path) -> SyntheticCorpusConfig:
        data = _exact_object(
            load_mapping(path),
            name="synthetic corpus config",
            required={
                "format_version",
                "seed",
                "examples_per_request",
                "generator_lanes",
                "critic_model",
                "critic_reasoning_effort",
                "max_completion_tokens",
                "quality",
            },
        )
        if data["format_version"] != 1:
            raise ValueError("synthetic corpus config format_version must be 1")
        lanes_value = data["generator_lanes"]
        if not isinstance(lanes_value, list):
            raise TypeError("generator_lanes must be an array")
        lanes = tuple(GeneratorLane.from_mapping(item) for item in lanes_value)
        if not lanes:
            raise ValueError("generator_lanes must not be empty")
        critic_model = _required_string(data["critic_model"], name="critic_model")
        critic_effort = _required_string(
            data["critic_reasoning_effort"], name="critic_reasoning_effort"
        )
        if critic_effort not in {"low", "medium", "high"}:
            raise ValueError(f"unsupported critic reasoning effort {critic_effort}")
        seed = _required_integer(data["seed"], name="seed", minimum=0)
        return cls(
            1,
            seed,
            _required_integer(data["examples_per_request"], name="examples_per_request"),
            lanes,
            critic_model,
            cast(ReasoningEffort, critic_effort),
            _required_integer(data["max_completion_tokens"], name="max_completion_tokens"),
            QualityConfig.from_mapping(data["quality"]),
        )

    def lane_for_request(self, index: int) -> GeneratorLane:
        schedule = tuple(lane for lane in self.generator_lanes for _ in range(lane.weight))
        return schedule[index % len(schedule)]


@dataclass(frozen=True, slots=True)
class CandidatePrompt:
    text: str
    method: Literal["theme", "imagery", "paraphrase"]

    @classmethod
    def from_mapping(cls, value: object) -> CandidatePrompt:
        data = _exact_object(value, name="candidate prompt", required={"text", "method"})
        method = _required_string(data["method"], name="candidate prompt method")
        if method not in {"theme", "imagery", "paraphrase"}:
            raise ValueError(f"unsupported candidate prompt method {method}")
        return cls(
            _required_string(data["text"], name="candidate prompt text"),
            cast(Literal["theme", "imagery", "paraphrase"], method),
        )


@dataclass(frozen=True, slots=True)
class SyntheticCandidate:
    candidate_id: str
    request_id: str
    generator_model: GeneratorModel
    title: str
    prompts: tuple[CandidatePrompt, ...]
    poem: str
    themes: tuple[str, ...]
    imagery: tuple[str, ...]
    mood: str
    form: str

    @classmethod
    def from_generation(
        cls,
        value: object,
        *,
        request_id: str,
        generator_model: GeneratorModel,
        ordinal: int,
    ) -> SyntheticCandidate:
        data = _exact_object(
            value,
            name="synthetic candidate",
            required={"title", "prompts", "poem", "themes", "imagery", "mood", "form"},
        )
        prompts_value = data["prompts"]
        if not isinstance(prompts_value, list):
            raise TypeError("candidate prompts must be an array")
        prompts = tuple(CandidatePrompt.from_mapping(item) for item in prompts_value)
        if len(prompts) != 3 or {prompt.method for prompt in prompts} != {
            "theme",
            "imagery",
            "paraphrase",
        }:
            raise ValueError(
                "candidate must contain exactly one theme, imagery, and paraphrase prompt"
            )
        poem = _required_string(data["poem"], name="candidate poem").replace("\r\n", "\n")
        identity = sha256(f"{request_id}\0{ordinal}\0{poem}".encode()).hexdigest()[:24]
        return cls(
            f"synthetic-{identity}",
            request_id,
            generator_model,
            _required_string(data["title"], name="candidate title"),
            prompts,
            poem,
            _string_tuple(data["themes"], name="candidate themes"),
            _string_tuple(data["imagery"], name="candidate imagery"),
            _required_string(data["mood"], name="candidate mood"),
            _required_string(data["form"], name="candidate form"),
        )

    @classmethod
    def from_mapping(cls, value: object) -> SyntheticCandidate:
        data = _exact_object(
            value,
            name="stored synthetic candidate",
            required={
                "candidate_id",
                "request_id",
                "generator_model",
                "title",
                "prompts",
                "poem",
                "themes",
                "imagery",
                "mood",
                "form",
            },
        )
        model = _required_string(data["generator_model"], name="generator_model")
        prompts_value = data["prompts"]
        if not isinstance(prompts_value, list):
            raise TypeError("stored candidate prompts must be an array")
        return cls(
            _required_string(data["candidate_id"], name="candidate_id"),
            _required_string(data["request_id"], name="request_id"),
            model,
            _required_string(data["title"], name="title"),
            tuple(CandidatePrompt.from_mapping(item) for item in prompts_value),
            _required_string(data["poem"], name="poem"),
            _string_tuple(data["themes"], name="themes"),
            _string_tuple(data["imagery"], name="imagery"),
            _required_string(data["mood"], name="mood"),
            _required_string(data["form"], name="form"),
        )

    def to_mapping(self) -> dict[str, object]:
        return cast(dict[str, object], asdict(self))


@dataclass(frozen=True, slots=True)
class Critique:
    candidate_id: str
    prompt_adherence: int
    coherence: int
    craft: int
    originality: int
    degeneration: bool
    named_author_imitation: bool
    suspected_quote: bool
    decision: Decision
    reasons: tuple[str, ...]

    @classmethod
    def from_mapping(cls, value: object, *, candidate_id: str) -> Critique:
        data = _exact_object(
            value,
            name="critique",
            required={
                "prompt_adherence",
                "coherence",
                "craft",
                "originality",
                "degeneration",
                "named_author_imitation",
                "suspected_quote",
                "decision",
                "reasons",
            },
        )
        for name in ("degeneration", "named_author_imitation", "suspected_quote"):
            if not isinstance(data[name], bool):
                raise TypeError(f"{name} must be boolean")
        decision = _required_string(data["decision"], name="decision")
        if decision not in {"accept", "reject"}:
            raise ValueError(f"unsupported critique decision {decision}")
        scores = tuple(
            _required_integer(data[name], name=name)
            for name in ("prompt_adherence", "coherence", "craft", "originality")
        )
        if any(score > 5 for score in scores):
            raise ValueError("critique scores must be <= 5")
        return cls(
            candidate_id=candidate_id,
            prompt_adherence=scores[0],
            coherence=scores[1],
            craft=scores[2],
            originality=scores[3],
            degeneration=cast(bool, data["degeneration"]),
            named_author_imitation=cast(bool, data["named_author_imitation"]),
            suspected_quote=cast(bool, data["suspected_quote"]),
            decision=cast(Decision, decision),
            reasons=_string_tuple(data["reasons"], name="critique reasons"),
        )


def _generation_schema() -> dict[str, object]:
    prompt_schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string"},
            "method": {"type": "string", "enum": ["theme", "imagery", "paraphrase"]},
        },
        "required": ["text", "method"],
        "additionalProperties": False,
    }
    candidate_schema = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "prompts": {
                "type": "array",
                "items": prompt_schema,
            },
            "poem": {"type": "string"},
            "themes": {
                "type": "array",
                "items": {"type": "string"},
            },
            "imagery": {
                "type": "array",
                "items": {"type": "string"},
            },
            "mood": {"type": "string"},
            "form": {"type": "string"},
        },
        "required": ["title", "prompts", "poem", "themes", "imagery", "mood", "form"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "examples": {
                "type": "array",
                "items": candidate_schema,
            }
        },
        "required": ["examples"],
        "additionalProperties": False,
    }


def _critic_schema() -> dict[str, object]:
    return {
        "type": "object",
        "properties": {
            "prompt_adherence": {"type": "integer", "minimum": 1, "maximum": 5},
            "coherence": {"type": "integer", "minimum": 1, "maximum": 5},
            "craft": {"type": "integer", "minimum": 1, "maximum": 5},
            "originality": {"type": "integer", "minimum": 1, "maximum": 5},
            "degeneration": {"type": "boolean"},
            "named_author_imitation": {"type": "boolean"},
            "suspected_quote": {"type": "boolean"},
            "decision": {"type": "string", "enum": ["accept", "reject"]},
            "reasons": {
                "type": "array",
                "items": {"type": "string"},
            },
        },
        "required": [
            "prompt_adherence",
            "coherence",
            "craft",
            "originality",
            "degeneration",
            "named_author_imitation",
            "suspected_quote",
            "decision",
            "reasons",
        ],
        "additionalProperties": False,
    }


def _response_format(name: str, schema: Mapping[str, object]) -> dict[str, object]:
    return {
        "type": "json_schema",
        "json_schema": {"name": name, "strict": True, "schema": dict(schema)},
    }


def _batch_request(custom_id: str, body: Mapping[str, object]) -> dict[str, object]:
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": dict(body),
    }


def _generation_briefs(index: int, count: int, seed: int) -> tuple[dict[str, str], ...]:
    briefs: list[dict[str, str]] = []
    for ordinal in range(count):
        digest = sha256(f"{seed}:{index}:{ordinal}:brief".encode()).digest()
        scenario_index = digest[0] % len(SETTINGS)
        action_index = digest[1] % len(ACTIONS)
        briefs.append(
            {
                "setting": SETTINGS[scenario_index],
                "required_objects": OBJECT_PAIRS[scenario_index],
                "physical_event": ACTIONS[action_index],
                "emotional_pressure": PRESSURES[digest[2] % len(PRESSURES)],
                "participants": PARTICIPANTS[action_index],
                "form": FORMS[digest[4] % len(FORMS)],
            }
        )
    return tuple(briefs)


def plan_generation(
    config_path: Path,
    *,
    request_count: int,
    output_directory: Path,
    model_override: str | None = None,
    openai_compatible: bool = False,
    response_format_mode: ResponseFormat = "json-schema",
    max_tokens_field: Literal["max_completion_tokens", "max_tokens"] = "max_completion_tokens",
) -> tuple[Path, Path]:
    config = SyntheticCorpusConfig.load(config_path)
    if request_count < 1:
        raise ValueError("request_count must be positive")
    if model_override is not None:
        _required_string(model_override, name="model_override")
    if response_format_mode not in {"json-schema", "json-object", "none"}:
        raise ValueError(f"unsupported response format mode: {response_format_mode}")
    if max_tokens_field not in {"max_completion_tokens", "max_tokens"}:
        raise ValueError(f"unsupported max token field: {max_tokens_field}")
    requests: list[dict[str, object]] = []
    assignments: list[dict[str, object]] = []
    for index in range(request_count):
        lane = config.lane_for_request(index)
        model = model_override or lane.model
        custom_id = f"poetry-synthetic-{index:08d}"
        diversity_lane = GENERATION_LANES[index % len(GENERATION_LANES)]
        diversity_seed = sha256(f"{config.seed}:{index}".encode()).hexdigest()
        briefs = _generation_briefs(index, config.examples_per_request, config.seed)
        body: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": GENERATOR_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        f"Generate {config.examples_per_request} mutually distinct examples. "
                        f"Creative lane: {diversity_lane}. Diversity seed: {diversity_seed}. "
                        "The seed is only an identity marker; do not include it in the output. "
                        f"Briefs in required output order: {_canonical_json(briefs)}"
                    ),
                },
            ],
            "temperature": lane.temperature,
            max_tokens_field: config.max_completion_tokens,
        }
        if not openai_compatible:
            body["seed"] = config.seed + index
            body["reasoning_effort"] = lane.reasoning_effort
        if response_format_mode == "json-schema":
            body["response_format"] = _response_format(
                "poetry_training_bundle",
                _generation_schema(),
            )
        elif response_format_mode == "json-object":
            body["response_format"] = {"type": "json_object"}
        requests.append(_batch_request(custom_id, body))
        assignments.append(
            {
                "custom_id": custom_id,
                "model": model,
                "diversity_lane": diversity_lane,
                "diversity_seed": diversity_seed,
                "briefs": briefs,
                "request_sha256": sha256(_canonical_json(body).encode()).hexdigest(),
            }
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    requests_path = output_directory / "generation.requests.jsonl"
    plan_path = output_directory / "generation.plan.json"
    _write_jsonl(requests_path, requests)
    _write_json(
        plan_path,
        {
            "format_version": 1,
            "config": str(config_path),
            "config_sha256": file_hash(config_path),
            "request_count": request_count,
            "candidate_capacity": request_count * config.examples_per_request,
            "openai_compatible": openai_compatible,
            "response_format_mode": response_format_mode,
            "max_tokens_field": max_tokens_field,
            "requests": str(requests_path),
            "requests_sha256": file_hash(requests_path),
            "assignments": assignments,
        },
    )
    return requests_path, plan_path


def _request_models(requests_path: Path) -> dict[str, GeneratorModel]:
    result: dict[str, GeneratorModel] = {}
    for record in _read_jsonl(requests_path):
        custom_id = _required_string(record.get("custom_id"), name="custom_id")
        body = record.get("body")
        if not isinstance(body, dict):
            raise TypeError("batch request body must be an object")
        model = _required_string(body.get("model"), name="batch request model")
        if custom_id in result:
            raise ValueError(f"duplicate custom_id {custom_id}")
        result[custom_id] = model
    return result


def _result_content(record: Mapping[str, object]) -> tuple[str, dict[str, object]]:
    custom_id = _required_string(record.get("custom_id"), name="result custom_id")
    error = record.get("error")
    if error is not None:
        raise RuntimeError(f"batch result {custom_id} failed: {error}")
    response = record.get("response")
    if not isinstance(response, dict):
        raise TypeError(f"batch result {custom_id} response must be an object")
    if response.get("status_code") != 200:
        raise RuntimeError(f"batch result {custom_id} status is {response.get('status_code')}")
    body = response.get("body")
    if not isinstance(body, dict):
        raise TypeError(f"batch result {custom_id} body must be an object")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError(f"batch result {custom_id} must contain one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise TypeError(f"batch result {custom_id} message must be an object")
    content = _required_string(message.get("content"), name=f"batch result {custom_id} content")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error_value:
        raise ValueError(f"batch result {custom_id} content is not JSON") from error_value
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"batch result {custom_id} content must decode to an object")
    return custom_id, cast(dict[str, object], value)


def _result_usage(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    totals = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "missing_usage_records": 0,
    }
    for record in records:
        response = record.get("response")
        body = response.get("body") if isinstance(response, dict) else None
        usage = body.get("usage") if isinstance(body, dict) else None
        if not isinstance(usage, dict):
            totals["missing_usage_records"] += 1
            continue
        for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
            value = usage.get(name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"result usage {name} must be a non-negative integer")
            totals[name] += value
    return totals


def ingest_generation_results(
    config_path: Path,
    *,
    requests_path: Path,
    results_path: Path,
    output_directory: Path,
    create_critic_requests: bool = True,
) -> tuple[Path, Path]:
    config = SyntheticCorpusConfig.load(config_path)
    request_models = _request_models(requests_path)
    seen_results: set[str] = set()
    candidates: list[SyntheticCandidate] = []
    generation_result_rejections: list[dict[str, str]] = []
    candidate_rejections: list[dict[str, str | int]] = []
    generation_results = _read_jsonl(results_path)
    for result in generation_results:
        custom_id = _required_string(result.get("custom_id"), name="result custom_id")
        if custom_id not in request_models:
            raise ValueError(f"unexpected generation result {custom_id}")
        if custom_id in seen_results:
            raise ValueError(f"duplicate generation result {custom_id}")
        seen_results.add(custom_id)
        try:
            parsed_custom_id, content = _result_content(result)
        except (RuntimeError, TypeError, ValueError) as error:
            generation_result_rejections.append(
                {
                    "custom_id": custom_id,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        if parsed_custom_id != custom_id:
            raise AssertionError("parsed result custom ID changed")
        examples = content.get("examples")
        if not isinstance(examples, list) or len(examples) != config.examples_per_request:
            generation_result_rejections.append(
                {
                    "custom_id": custom_id,
                    "reason": (
                        f"expected {config.examples_per_request} examples, "
                        f"got {len(examples) if isinstance(examples, list) else 'non-array'}"
                    ),
                }
            )
            continue
        for ordinal, example in enumerate(examples):
            try:
                candidate = SyntheticCandidate.from_generation(
                    example,
                    request_id=custom_id,
                    generator_model=request_models[custom_id],
                    ordinal=ordinal,
                )
            except (TypeError, ValueError) as error:
                candidate_rejections.append(
                    {
                        "custom_id": custom_id,
                        "ordinal": ordinal,
                        "reason": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            candidates.append(candidate)
    missing = set(request_models).difference(seen_results)
    if missing:
        raise ValueError(f"generation results are missing requests: {sorted(missing)}")
    if len({candidate.candidate_id for candidate in candidates}) != len(candidates):
        raise ValueError("generation produced duplicate candidate IDs")

    critic_requests = (
        [
            _batch_request(
                f"critic-{candidate.candidate_id}",
                {
                    "model": config.critic_model,
                    "messages": [
                        {"role": "system", "content": CRITIC_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": _canonical_json(
                                {
                                    "candidate_id": candidate.candidate_id,
                                    "title": candidate.title,
                                    "prompts": [asdict(prompt) for prompt in candidate.prompts],
                                    "poem": candidate.poem,
                                }
                            ),
                        },
                    ],
                    "temperature": 0.2,
                    "seed": int(sha256(candidate.candidate_id.encode()).hexdigest()[:8], 16),
                    "reasoning_effort": config.critic_reasoning_effort,
                    "max_completion_tokens": 1024,
                    "response_format": _response_format(
                        "poetry_training_critique",
                        _critic_schema(),
                    ),
                },
            )
            for candidate in candidates
        ]
        if create_critic_requests
        else []
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    candidates_path = output_directory / "candidates.jsonl"
    critic_requests_path = output_directory / "critic.requests.jsonl"
    _write_jsonl(candidates_path, (candidate.to_mapping() for candidate in candidates))
    _write_jsonl(critic_requests_path, critic_requests)
    _write_json(
        output_directory / "generation.ingest.receipt.json",
        {
            "format_version": 1,
            "request_count": len(request_models),
            "candidate_count": len(candidates),
            "critic_requests_enabled": create_critic_requests,
            "generation_result_rejections": generation_result_rejections,
            "candidate_rejections": candidate_rejections,
            "usage": _result_usage(generation_results),
            "requests_sha256": file_hash(requests_path),
            "results_sha256": file_hash(results_path),
            "candidates_sha256": file_hash(candidates_path),
            "critic_requests_sha256": file_hash(critic_requests_path),
        },
    )
    return candidates_path, critic_requests_path


def _words(text: str) -> tuple[str, ...]:
    return tuple(word.casefold() for word in WORD.findall(text))


def _ngrams(words: Sequence[str], n: int) -> set[tuple[str, ...]]:
    return {tuple(words[index : index + n]) for index in range(max(0, len(words) - n + 1))}


def _reference_ngram_matches(
    candidates: Sequence[SyntheticCandidate],
    *,
    reference_manifest: Path,
    ngram_size: int,
) -> dict[str, set[tuple[str, ...]]]:
    candidate_ngrams = {
        candidate.candidate_id: _ngrams(_words(candidate.poem), ngram_size)
        for candidate in candidates
    }
    query_ngrams = set().union(*candidate_ngrams.values()) if candidate_ngrams else set()
    matched: set[tuple[str, ...]] = set()
    if query_ngrams:
        for document in iter_manifest(reference_manifest, allow_synthetic=True):
            for block in document.blocks:
                matched.update(_ngrams(_words(block.text), ngram_size) & query_ngrams)
    return {candidate_id: ngrams & matched for candidate_id, ngrams in candidate_ngrams.items()}


def _local_quality_reasons(candidate: SyntheticCandidate, config: QualityConfig) -> tuple[str, ...]:
    words = _words(candidate.poem)
    lines = tuple(line.strip() for line in candidate.poem.splitlines() if line.strip())
    bigrams = tuple(zip(words, words[1:], strict=False))
    bigram_counts = Counter(bigrams)
    repeated_bigrams = sum(count - 1 for count in bigram_counts.values() if count > 1)
    repeated_bigram_rate = repeated_bigrams / max(1, len(bigrams))
    reasons: list[str] = []
    if not config.minimum_word_count <= len(words) <= config.maximum_word_count:
        reasons.append(f"word_count={len(words)}")
    if not config.minimum_line_count <= len(lines) <= config.maximum_line_count:
        reasons.append(f"line_count={len(lines)}")
    if len(set(lines)) != len(lines):
        reasons.append("repeated_lines")
    if any(line.endswith("\\") for line in lines):
        reasons.append("markdown_line_break")
    if candidate.poem.count('"') % 2 != 0 or candidate.poem.count("“") != candidate.poem.count("”"):
        reasons.append("unbalanced_quotation_marks")
    if repeated_bigram_rate > config.maximum_repeated_bigram_rate:
        reasons.append(f"repeated_bigram_rate={repeated_bigram_rate:.6f}")
    normalized_poem = " ".join(words)
    banned_word_count = sum(word in BANNED_SYNTHETIC_WORDS for word in words)
    if banned_word_count > config.maximum_banned_word_count:
        reasons.append(f"banned_word_count={banned_word_count}")
    stock_phrase_count = sum(normalized_poem.count(phrase) for phrase in STOCK_PHRASES)
    if stock_phrase_count > config.maximum_stock_phrase_count:
        reasons.append(f"stock_phrase_count={stock_phrase_count}")
    return tuple(reasons)


def _critique_reasons(critique: Critique, config: QualityConfig) -> tuple[str, ...]:
    reasons: list[str] = []
    if critique.decision != "accept":
        reasons.append("critic_rejected")
    for field, minimum in (
        ("prompt_adherence", config.minimum_prompt_adherence),
        ("coherence", config.minimum_coherence),
        ("craft", config.minimum_craft),
        ("originality", config.minimum_originality),
    ):
        if getattr(critique, field) < minimum:
            reasons.append(f"{field}={getattr(critique, field)}")
    if critique.degeneration:
        reasons.append("critic_detected_degeneration")
    if critique.named_author_imitation:
        reasons.append("critic_detected_named_author_imitation")
    if critique.suspected_quote:
        reasons.append("critic_suspected_quote")
    return tuple(reasons)


def _source_document(
    candidate: SyntheticCandidate,
    critique: Critique | None,
    *,
    critic_model: str | None,
) -> SourceDocument:
    document_id = candidate.candidate_id
    poem_id = f"{document_id}:poem"
    block_id = f"{poem_id}:full"
    critic_metadata = (
        {
            "critic_summary": _canonical_json(
                {
                    "prompt_adherence": critique.prompt_adherence,
                    "coherence": critique.coherence,
                    "craft": critique.craft,
                    "originality": critique.originality,
                }
            )
        }
        if critique is not None
        else {}
    )
    provenance = Provenance(
        work=candidate.title,
        author=candidate.generator_model,
        licence="synthetic-generated-output",
        source="OpenAI-compatible synthetic corpus pipeline",
        source_locator=candidate.request_id,
        rights_status="synthetic",
        rights_notes=(
            f"Generated by {candidate.generator_model}; "
            f"{'independently model-critiqued and ' if critique is not None else ''}"
            "screened by deterministic local gates; "
            "synthetic status does not assert non-memorization or exclusive copyright."
        ),
    )
    block = ContentBlock(
        block_id=block_id,
        kind="poem",
        text=candidate.poem,
        poem_id=poem_id,
        title=candidate.title,
        start_char=0,
        end_char=len(candidate.poem),
        metadata={
            "generator_model": candidate.generator_model,
            "themes": _canonical_json(candidate.themes),
            "imagery": _canonical_json(candidate.imagery),
            "mood": candidate.mood,
            "form": candidate.form,
            **critic_metadata,
        },
    )
    return SourceDocument(
        document_id=document_id,
        provenance=provenance,
        text=candidate.poem,
        blocks=(block,),
        source_path=f"synthetic-request:{candidate.request_id}",
        raw_text=candidate.poem,
        metadata={
            "generator_model": candidate.generator_model,
            **({"critic_model": critic_model} if critic_model is not None else {}),
        },
        transformation_lineage=(
            "openai_compatible_json_generation",
            *(("independent_model_critique",) if critique is not None else ()),
            "local_quality_gates",
        ),
    )


def finalize_synthetic_corpus(
    config_path: Path,
    *,
    candidates_path: Path,
    critic_results_path: Path | None,
    output_directory: Path,
    reference_manifest: Path | None = None,
) -> Path:
    config = SyntheticCorpusConfig.load(config_path)
    candidates = tuple(
        SyntheticCandidate.from_mapping(record) for record in _read_jsonl(candidates_path)
    )
    candidates_by_id = {candidate.candidate_id: candidate for candidate in candidates}
    if len(candidates_by_id) != len(candidates):
        raise ValueError("candidate file contains duplicate IDs")
    critiques: dict[str, Critique] = {}
    critic_results = _read_jsonl(critic_results_path) if critic_results_path is not None else ()
    for result in critic_results:
        custom_id, content = _result_content(result)
        if not custom_id.startswith("critic-"):
            raise ValueError(f"unexpected critic result ID {custom_id}")
        candidate_id = custom_id.removeprefix("critic-")
        if candidate_id not in candidates_by_id:
            raise ValueError(f"critic result references unknown candidate {candidate_id}")
        if candidate_id in critiques:
            raise ValueError(f"duplicate critique for {candidate_id}")
        critiques[candidate_id] = Critique.from_mapping(content, candidate_id=candidate_id)
    missing = set(candidates_by_id).difference(critiques)
    if critic_results_path is not None and missing:
        raise ValueError(f"critic results are missing candidates: {sorted(missing)}")
    reference_matches = (
        _reference_ngram_matches(
            candidates,
            reference_manifest=reference_manifest,
            ngram_size=config.quality.dedup_ngram_size,
        )
        if reference_manifest is not None
        else {}
    )

    accepted: list[SyntheticCandidate] = []
    quality_rows: list[dict[str, object]] = []
    exact_poems: set[str] = set()
    accepted_ngrams: set[tuple[str, ...]] = set()
    for candidate in sorted(candidates, key=lambda item: item.candidate_id):
        critique = critiques.get(candidate.candidate_id)
        reasons = list(_local_quality_reasons(candidate, config.quality))
        if critique is not None:
            reasons.extend(_critique_reasons(critique, config.quality))
        normalised = " ".join(_words(candidate.poem))
        candidate_ngrams = _ngrams(_words(candidate.poem), config.quality.dedup_ngram_size)
        if normalised in exact_poems:
            reasons.append("duplicate_poem")
        if reference_matches.get(candidate.candidate_id):
            reasons.append(f"reference_shared_{config.quality.dedup_ngram_size}_gram")
        if candidate_ngrams & accepted_ngrams:
            reasons.append(f"shared_{config.quality.dedup_ngram_size}_gram")
        is_accepted = not reasons
        if is_accepted:
            accepted.append(candidate)
            exact_poems.add(normalised)
            accepted_ngrams.update(candidate_ngrams)
        quality_rows.append(
            {
                "candidate_id": candidate.candidate_id,
                "accepted": is_accepted,
                "rejection_reasons": reasons,
                "critic": asdict(critique) if critique is not None else None,
            }
        )

    documents = tuple(
        _source_document(
            candidate,
            critiques.get(candidate.candidate_id),
            critic_model=config.critic_model if critic_results_path is not None else None,
        )
        for candidate in accepted
    )
    prompts = tuple(
        PromptRecord(
            prompt_id=sha256(
                f"{candidate.candidate_id}\0{prompt.method}\0{prompt.text}".encode()
            ).hexdigest()[:24],
            document_id=candidate.candidate_id,
            prompt=prompt.text,
            method=cast(PromptMethod, prompt.method),
            source_attribution=(f"synthetic:{candidate.generator_model}:{candidate.request_id}"),
            poem_id=f"{candidate.candidate_id}:poem",
        )
        for candidate in accepted
        for prompt in candidate.prompts
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.jsonl"
    prompts_path = output_directory / "prompts.jsonl"
    thoughts_path = output_directory / "thoughts.jsonl"
    quality_path = output_directory / "quality.jsonl"
    write_manifest(manifest_path, documents, allow_synthetic=True)
    write_prompt_records(prompts_path, prompts)
    write_thought_records(thoughts_path, ())
    _write_jsonl(quality_path, quality_rows)
    receipt_path = output_directory / "synthetic.receipt.json"
    _write_json(
        receipt_path,
        {
            "format_version": 1,
            "config_sha256": file_hash(config_path),
            "candidates_sha256": file_hash(candidates_path),
            "critic_enabled": critic_results_path is not None,
            "critic_results_sha256": (
                file_hash(critic_results_path) if critic_results_path is not None else None
            ),
            "reference_manifest_sha256": (
                file_hash(reference_manifest) if reference_manifest is not None else None
            ),
            "candidate_count": len(candidates),
            "accepted_count": len(accepted),
            "rejected_count": len(candidates) - len(accepted),
            "critic_usage": _result_usage(critic_results)
            if critic_results_path is not None
            else None,
            "manifest_sha256": file_hash(manifest_path),
            "prompts_sha256": file_hash(prompts_path),
            "thoughts_sha256": file_hash(thoughts_path),
            "quality_sha256": file_hash(quality_path),
            "requires_allow_synthetic": True,
        },
    )
    return receipt_path


def merge_corpus_artifacts(
    *,
    base_manifest: Path,
    base_prompts: Path,
    base_thoughts: Path,
    base_pairings: Path,
    synthetic_directory: Path,
    output_directory: Path,
) -> Path:
    synthetic_manifest = synthetic_directory / "manifest.jsonl"
    synthetic_prompts = synthetic_directory / "prompts.jsonl"
    synthetic_thoughts = synthetic_directory / "thoughts.jsonl"
    synthetic_receipt = synthetic_directory / "synthetic.receipt.json"
    prompts = (*read_prompt_records(base_prompts), *read_prompt_records(synthetic_prompts))
    thoughts = (*read_thought_records(base_thoughts), *read_thought_records(synthetic_thoughts))
    pairings = read_pairings(base_pairings)

    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "manifest.jsonl"
    prompts_path = output_directory / "prompts.jsonl"
    thoughts_path = output_directory / "thoughts.jsonl"
    pairings_path = output_directory / "pairings.jsonl"
    document_count, document_ids = _write_merged_manifest(
        manifest_path,
        manifests=(base_manifest, synthetic_manifest),
    )
    unknown_prompt_documents = sorted(
        {prompt.document_id for prompt in prompts}.difference(document_ids)
    )
    if unknown_prompt_documents:
        raise ValueError(f"merged prompts reference unknown documents: {unknown_prompt_documents}")
    unknown_thought_documents = sorted(
        {thought.document_id for thought in thoughts}.difference(document_ids)
    )
    if unknown_thought_documents:
        raise ValueError(
            f"merged thoughts reference unknown documents: {unknown_thought_documents}"
        )

    write_prompt_records(prompts_path, prompts)
    write_thought_records(thoughts_path, thoughts)
    write_pairings(pairings_path, pairings)
    receipt_path = output_directory / "merge.receipt.json"
    _write_json(
        receipt_path,
        {
            "format_version": 1,
            "base": {
                "manifest_sha256": file_hash(base_manifest),
                "prompts_sha256": file_hash(base_prompts),
                "thoughts_sha256": file_hash(base_thoughts),
                "pairings_sha256": file_hash(base_pairings),
            },
            "synthetic": {
                "receipt_sha256": file_hash(synthetic_receipt),
                "manifest_sha256": file_hash(synthetic_manifest),
                "prompts_sha256": file_hash(synthetic_prompts),
                "thoughts_sha256": file_hash(synthetic_thoughts),
            },
            "counts": {
                "documents": document_count,
                "prompts": len(prompts),
                "thoughts": len(thoughts),
                "pairings": len(pairings),
            },
            "outputs": {
                "manifest_sha256": file_hash(manifest_path),
                "prompts_sha256": file_hash(prompts_path),
                "thoughts_sha256": file_hash(thoughts_path),
                "pairings_sha256": file_hash(pairings_path),
            },
            "requires_allow_synthetic": True,
        },
    )
    return receipt_path


def _write_merged_manifest(output_path: Path, *, manifests: Sequence[Path]) -> tuple[int, set[str]]:
    def ordered_documents(path: Path) -> Iterable[SourceDocument]:
        previous_id: str | None = None
        for document in iter_manifest(path, allow_synthetic=True):
            if previous_id is not None and document.document_id <= previous_id:
                raise ValueError(f"manifest is not ordered by document_id: {path}")
            previous_id = document.document_id
            yield document

    document_ids: set[str] = set()
    document_count = 0

    def write(handle: BinaryIO) -> None:
        nonlocal document_count
        documents = merge_sorted(
            *(ordered_documents(path) for path in manifests),
            key=lambda document: document.document_id,
        )
        for document in documents:
            if document.document_id in document_ids:
                raise ValueError(
                    "base and synthetic corpora contain duplicate document IDs: "
                    f"{document.document_id}"
                )
            document_ids.add(document.document_id)
            document_count += 1
            line = json.dumps(
                document.to_mapping(),
                ensure_ascii=False,
                sort_keys=True,
            )
            handle.write(f"{line}\n".encode())

    atomic_write(output_path, write)
    return document_count, document_ids


def _request_token_estimate(body: Mapping[str, object]) -> int:
    maximum_value = body.get("max_completion_tokens", body.get("max_tokens"))
    maximum_completion = _required_integer(
        maximum_value,
        name="maximum completion tokens",
    )
    input_bytes = len(_canonical_json(body).encode("utf-8"))
    conservative_input_tokens = (input_bytes + 2) // 3
    return maximum_completion + conservative_input_tokens


def _final_text_only_response_body(
    response_body: Mapping[str, object],
    *,
    custom_id: str,
) -> dict[str, object]:
    choices = response_body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError(f"request {custom_id} must return exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise TypeError(f"request {custom_id} message must be an object")
    try:
        content = _required_string(
            message.get("content"),
            name=f"request {custom_id} final content",
        )
    except ValueError as error:
        if message.get("reasoning") is not None or message.get("reasoning_content") is not None:
            raise ValueError(
                f"request {custom_id} returned reasoning without final content"
            ) from error
        raise

    result: dict[str, object] = {
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }
    for key in ("model", "usage"):
        if key in response_body:
            result[key] = response_body[key]
    return result


def _sync_request(
    client: Cerebras,
    request: Mapping[str, object],
    limiter: DualTokenBucket,
) -> dict[str, object]:
    custom_id = _required_string(request.get("custom_id"), name="custom_id")
    body = request.get("body")
    if not isinstance(body, dict):
        raise TypeError(f"request {custom_id} body must be an object")
    reserved_tokens = _request_token_estimate(body)
    limiter.acquire(reserved_tokens)
    completion = client.post(
        "/v1/chat/completions",
        cast_to=ChatCompletionResponse,
        body=cast(dict[str, object], body),
    )
    response_body = completion.model_dump(mode="json")
    usage = response_body.get("usage")
    total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    if isinstance(total_tokens, int) and not isinstance(total_tokens, bool) and total_tokens >= 0:
        limiter.refund_model_tokens(reserved_tokens, total_tokens)
    return {
        "custom_id": custom_id,
        "response": {"status_code": 200, "body": response_body},
        "error": None,
    }


def _pending_requests(
    requests_path: Path,
    results_path: Path,
) -> tuple[dict[str, object], ...]:
    requests = _read_jsonl(requests_path)
    request_ids = [
        _required_string(request.get("custom_id"), name="custom_id") for request in requests
    ]
    if len(request_ids) != len(set(request_ids)):
        raise ValueError("request file contains duplicate custom IDs")
    existing_results = _read_jsonl(results_path) if results_path.exists() else ()
    completed_ids = [
        _required_string(record.get("custom_id"), name="existing result custom_id")
        for record in existing_results
    ]
    if len(completed_ids) != len(set(completed_ids)):
        raise ValueError("existing result file contains duplicate custom IDs")
    unexpected_results = set(completed_ids).difference(request_ids)
    if unexpected_results:
        raise ValueError(f"existing results contain unknown IDs: {sorted(unexpected_results)}")
    completed = set(completed_ids)
    return tuple(
        request
        for request in requests
        if _required_string(request.get("custom_id"), name="custom_id") not in completed
    )


def _execute_pending_requests(
    pending: Sequence[Mapping[str, object]],
    results_path: Path,
    *,
    concurrency: int,
    worker: Callable[[Mapping[str, object]], dict[str, object]],
    provider_name: str,
) -> None:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")
    results_path.parent.mkdir(parents=True, exist_ok=True)
    failures: list[Exception] = []
    with results_path.open("a", encoding="utf-8", newline="\n") as handle:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(worker, request): request for request in pending}
            for future in as_completed(futures):
                try:
                    result = future.result()
                except Exception as error:
                    failures.append(error)
                    continue
                handle.write(_canonical_json(result))
                handle.write("\n")
                handle.flush()
    if failures:
        raise RuntimeError(
            f"{len(failures)} {provider_name} requests failed; successful results were preserved"
        ) from failures[0]


def run_synchronous_batch(
    requests_path: Path,
    results_path: Path,
    *,
    concurrency: int = 8,
    requests_per_minute: int = 950,
    tokens_per_minute: int = 950_000,
) -> None:
    if not os.environ.get("CEREBRAS_API_KEY"):
        raise RuntimeError("CEREBRAS_API_KEY is required")
    pending = _pending_requests(requests_path, results_path)
    limiter = DualTokenBucket(
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
    )
    client = Cerebras(max_retries=0, timeout=180.0)

    def worker(request: Mapping[str, object]) -> dict[str, object]:
        return _sync_request(client, request, limiter)

    _execute_pending_requests(
        pending,
        results_path,
        concurrency=concurrency,
        worker=worker,
        provider_name="Cerebras",
    )


def _sync_openai_compatible_request(
    *,
    base_url: str,
    api_key: str,
    request: Mapping[str, object],
    limiter: DualTokenBucket,
    timeout_seconds: float,
    store_final_text_only: bool,
) -> dict[str, object]:
    custom_id = _required_string(request.get("custom_id"), name="custom_id")
    body = request.get("body")
    if not isinstance(body, dict):
        raise TypeError(f"request {custom_id} body must be an object")
    reserved_tokens = _request_token_estimate(body)
    limiter.acquire(reserved_tokens)
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    http_request = urllib.request.Request(
        endpoint,
        data=_canonical_json(body).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(http_request, timeout=timeout_seconds) as response:
            status_code = response.getcode()
            payload = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"request {custom_id} returned HTTP {error.code}: {detail}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"request {custom_id} failed: {error.reason}") from error
    if status_code != 200:
        raise RuntimeError(f"request {custom_id} returned HTTP {status_code}")
    if not isinstance(payload, dict) or any(not isinstance(key, str) for key in payload):
        raise TypeError(f"request {custom_id} response must be a JSON object")
    response_body = cast(dict[str, object], payload)
    usage = response_body.get("usage")
    total_tokens = usage.get("total_tokens") if isinstance(usage, dict) else None
    if isinstance(total_tokens, int) and not isinstance(total_tokens, bool) and total_tokens >= 0:
        limiter.refund_model_tokens(reserved_tokens, total_tokens)
    stored_body = (
        _final_text_only_response_body(response_body, custom_id=custom_id)
        if store_final_text_only
        else response_body
    )
    return {
        "custom_id": custom_id,
        "response": {"status_code": status_code, "body": stored_body},
        "error": None,
    }


def run_openai_compatible_batch(
    requests_path: Path,
    results_path: Path,
    *,
    base_url: str,
    api_key_environment_variable: str = "OPENAI_API_KEY",
    concurrency: int = 8,
    requests_per_minute: int = 60,
    tokens_per_minute: int = 100_000,
    timeout_seconds: float = 180.0,
    store_final_text_only: bool = False,
) -> None:
    _required_string(base_url, name="base_url")
    _required_string(api_key_environment_variable, name="api_key_environment_variable")
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    api_key = os.environ.get(api_key_environment_variable)
    if not api_key:
        raise RuntimeError(f"{api_key_environment_variable} is required")
    pending = _pending_requests(requests_path, results_path)
    limiter = DualTokenBucket(
        requests_per_minute=requests_per_minute,
        tokens_per_minute=tokens_per_minute,
    )

    def worker(request: Mapping[str, object]) -> dict[str, object]:
        return _sync_openai_compatible_request(
            base_url=base_url,
            api_key=api_key,
            request=request,
            limiter=limiter,
            timeout_seconds=timeout_seconds,
            store_final_text_only=store_final_text_only,
        )

    _execute_pending_requests(
        pending,
        results_path,
        concurrency=concurrency,
        worker=worker,
        provider_name="OpenAI-compatible endpoint",
    )
