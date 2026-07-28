"""Deterministic, chunked synthetic prompt-response generation for poetry SFT."""

from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path
from typing import BinaryIO, Literal, cast

from tokenizers import Tokenizer

from poetry50m.config import file_hash
from poetry50m.trajectory._persistence import atomic_write

from .tokenizer import load_tokenizer

MaxTokensField = Literal["max_completion_tokens", "max_tokens"]
ReasoningEffort = Literal["none", "low", "medium", "high"]
TargetMetric = Literal["formatted", "supervised"]

FORMAT_VERSION = 1
LEGACY_RECIPE_VERSION = "poetry-sft-prompts-v1"
RECIPE_VERSION = "poetry-sft-prompts-v2"
SUPPORTED_RECIPE_VERSIONS = frozenset((LEGACY_RECIPE_VERSION, RECIPE_VERSION))
DEFAULT_TARGET_TOKENS = 15_000_000
TRACK1_SFT_TOKENIZER_SHA256 = "f36d39162cb38a59a74f2e3b082b50711613deb2d826f2d57cd1b8542e05a84d"
WORD = re.compile(r"[\w']+", re.UNICODE)
LENGTH_LINE_BOUNDS = {
    "very short": (4, 7),
    "short": (8, 12),
    "medium": (13, 20),
    "long": (21, 32),
}
REFUSAL_MARKERS = (
    "as an ai",
    "i apologize, but",
    "i cannot comply",
    "i cannot fulfill",
    "i can't assist",
    "i’m sorry, but i can’t",
    "i'm sorry, but i can't",
    "unable to provide",
)

SYSTEM_PROMPT = """\
Write original English poems for supervised fine-tuning.

Follow the user's prompt closely. The poem must be original rather than a quotation
or continuation. Do not mention these instructions, explain or critique the poem,
add a title unless requested, or wrap it in Markdown. Preserve intentional line
breaks. Return only the poem text."""

GENERATION_OUTPUT_INSTRUCTION = (
    "RETURN ONLY THE POEM ITSELF. DO NOT INCLUDE A TITLE, EXPLANATION, MARKDOWN, OR EMOJIS."
)

SUBJECTS = (
    "a honey bee choosing between the last two lavender flowers",
    "a lighthouse keeper recording a storm that never arrives",
    "love expressed through repairing a cracked bowl",
    "a child finding a moth asleep inside a library book",
    "two neighbours carrying groceries through sudden rain",
    "an empty train platform just before sunrise",
    "an old dog waiting beside a garden gate",
    "the moon reflected in a bucket after farm work",
    "someone learning the name of a bird after hearing it for years",
    "a baker opening the shop during a power cut",
    "grief hidden inside the task of folding clean laundry",
    "a swimmer returning to the sea after a long absence",
    "a mechanic discovering wildflowers beside an abandoned car",
    "the first cold morning in a community garden",
    "a letter that was carried for years but never opened",
    "a night nurse watching snow gather in a hospital courtyard",
    "a fig tree growing through a damaged fence",
    "two siblings clearing their childhood bedroom",
    "a ferry crossing made in complete fog",
    "a blackbird stealing thread from a washing line",
    "patience as practised while sharpening a kitchen knife",
    "the difference between solitude and loneliness",
    "forgiveness without reconciliation",
    "courage during an ordinary medical appointment",
)

FORMS = (
    "free verse",
    "a compact lyric",
    "a narrative poem",
    "a prose-like free-verse poem with deliberate line breaks",
    "three restrained tercets",
    "a dramatic monologue",
    "a list poem whose final item changes the earlier items",
    "two unequal stanzas separated by a one-line turn",
)

TONES = (
    "tender and observant",
    "plainspoken and unsentimental",
    "quietly funny",
    "melancholic but not despairing",
    "wonder-struck without becoming grandiose",
    "intimate and restrained",
    "earthy and conversational",
    "meditative with a sharp final turn",
)

LENGTHS = (
    ("very short", "4 to 7 lines"),
    ("short", "8 to 12 lines"),
    ("medium", "13 to 20 lines"),
    ("long", "21 to 32 lines"),
)

FORM_LENGTH_LABELS = {
    "free verse": ("very short", "short", "medium", "long"),
    "a compact lyric": ("very short", "short", "medium"),
    "a narrative poem": ("short", "medium", "long"),
    "a prose-like free-verse poem with deliberate line breaks": ("short", "medium", "long"),
    "three restrained tercets": ("short",),
    "a dramatic monologue": ("short", "medium", "long"),
    "a list poem whose final item changes the earlier items": (
        "very short",
        "short",
        "medium",
        "long",
    ),
    "two unequal stanzas separated by a one-line turn": ("short", "medium", "long"),
}
LENGTH_INSTRUCTIONS = dict(LENGTHS)
FORM_LENGTHS = tuple(
    (
        form,
        length_label,
        (
            "exactly 9 lines"
            if form == "three restrained tercets"
            else LENGTH_INSTRUCTIONS[length_label]
        ),
    )
    for form in FORMS
    for length_label in FORM_LENGTH_LABELS[form]
)

VOICES = (
    "clear contemporary language",
    "spare imagist language",
    "a private Stoic meditation",
    "a folk-ballad directness without forced rhyme",
    "an attentive nature-poetry voice",
    "a precise urban observational voice",
    "a fable-like voice without stating a moral",
    "a lyrical voice grounded in physical action",
)

TECHNIQUES = (
    "Use one recurring concrete image, changed by the ending.",
    "Include one natural line of dialogue.",
    "Let sound and rhythm carry the feeling; avoid end rhyme.",
    "Move from close observation to one surprising thought.",
    "Keep every metaphor physically coherent.",
    "Use a repeated action, but do not repeat a full line.",
    "Make the emotional turn implicit in what the speaker does.",
    "Use specific sensory detail and avoid abstract explanation.",
)


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_bytes(path: Path, payload: bytes) -> None:
    def write(handle: BinaryIO) -> None:
        handle.write(payload)

    atomic_write(path, write)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_bytes(path, (_canonical_json(value) + "\n").encode("utf-8"))


def _write_jsonl(path: Path, records: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{_canonical_json(record)}\n" for record in records).encode("utf-8")
    _write_bytes(path, payload)


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise TypeError(f"{path} must contain a JSON object")
    return cast(dict[str, object], value)


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"blank JSONL record at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
                raise TypeError(f"{path}:{line_number} must be a JSON object")
            records.append(cast(dict[str, object], value))
    return tuple(records)


def _required_string(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _required_integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _normalized_text_hash(text: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return sha256(normalized.encode()).hexdigest()


PROMPT_CAPACITY = (
    len(SUBJECTS) * len(FORM_LENGTHS) * len(TONES) * len(VOICES) * len(TECHNIQUES)
)
PROMPT_BLOCK_SIZE = 2_048


@lru_cache(maxsize=32)
def _prompt_axis_shifts(seed: int) -> tuple[int, int, int, int, int]:
    digest = sha256(f"{RECIPE_VERSION}:{seed}".encode()).digest()
    return (
        digest[0] % len(SUBJECTS),
        digest[1] % len(FORM_LENGTHS),
        digest[2] % len(TONES),
        digest[3] % len(VOICES),
        digest[4] % len(TECHNIQUES),
    )


def _prompt_axis_indices(index: int, seed: int) -> tuple[int, int, int, int, int]:
    block, remainder = divmod(index, PROMPT_BLOCK_SIZE)
    tone_code = remainder & 7
    voice_code = (remainder >> 3) & 7
    technique_code = (remainder >> 6) & 7
    length_code = (remainder >> 9) & 3
    subject_block = block % len(SUBJECTS)
    form_length_block = block // len(SUBJECTS)
    subject_shift, form_length_shift, tone_shift, voice_shift, technique_shift = (
        _prompt_axis_shifts(seed)
    )

    # This reversible interleaver visits every prompt combination exactly once
    # while balancing every contiguous 2,048-example production block.
    subject_index = (subject_block + remainder % len(SUBJECTS) + subject_shift) % len(SUBJECTS)
    form_length_index = (
        form_length_block * 4
        + length_code
        + tone_code
        + 3 * voice_code
        + 7 * technique_code
        + form_length_shift
    ) % len(FORM_LENGTHS)
    tone_index = (tone_code + tone_shift) % len(TONES)
    voice_index = (voice_code + voice_shift) % len(VOICES)
    technique_index = (technique_code + technique_shift) % len(TECHNIQUES)
    return subject_index, form_length_index, tone_index, voice_index, technique_index


@dataclass(frozen=True, slots=True)
class PromptSpec:
    subject: str
    form: str
    tone: str
    length_label: str
    length_instruction: str
    minimum_lines: int
    maximum_lines: int
    voice: str
    technique: str

    @classmethod
    def for_index(cls, index: int, seed: int) -> PromptSpec:
        subject_index, form_length_index, tone_index, voice_index, technique_index = (
            _prompt_axis_indices(index, seed)
        )
        subject = SUBJECTS[subject_index]
        form, length_label, length_instruction = FORM_LENGTHS[form_length_index]
        tone = TONES[tone_index]
        voice = VOICES[voice_index]
        technique = TECHNIQUES[technique_index]
        minimum_lines, maximum_lines = (
            (9, 9)
            if form == "three restrained tercets"
            else LENGTH_LINE_BOUNDS[length_label]
        )
        return cls(
            subject=subject,
            form=form,
            tone=tone,
            length_label=length_label,
            length_instruction=length_instruction,
            minimum_lines=minimum_lines,
            maximum_lines=maximum_lines,
            voice=voice,
            technique=technique,
        )

    def render(self) -> str:
        return (
            f"Write {self.form} about {self.subject}. Make it {self.tone}, using "
            f"{self.voice}. Use {self.length_instruction}, counting only lines containing words. "
            f"{self.technique}"
        )


@dataclass(frozen=True, slots=True)
class PlannedExample:
    example_id: str
    global_index: int
    prompt: str
    prompt_spec: PromptSpec

    @classmethod
    def create(cls, index: int, seed: int) -> PlannedExample:
        spec = PromptSpec.for_index(index, seed)
        return cls(
            example_id=f"synthetic-sft-v2-{index:09d}",
            global_index=index,
            prompt=spec.render(),
            prompt_spec=spec,
        )


def _chunk_id(seed: int, start_index: int, example_count: int) -> str:
    stop_index = start_index + example_count
    return f"sft-v2-s{seed}-{start_index:09d}-{stop_index:09d}"


def plan_sft_chunk(
    *,
    output_directory: Path,
    model: str,
    provider: str,
    start_index: int,
    example_count: int,
    seed: int = 20260728,
    temperature: float = 0.9,
    max_completion_tokens: int = 1024,
    max_tokens_field: MaxTokensField = "max_completion_tokens",
    reasoning_effort: ReasoningEffort | None = None,
) -> tuple[Path, Path]:
    """Create an immutable request plan for one disjoint SFT chunk."""
    model = _required_string(model, name="model")
    provider = _required_string(provider, name="provider")
    _required_integer(start_index, name="start_index")
    _required_integer(example_count, name="example_count", minimum=1)
    _required_integer(seed, name="seed")
    _required_integer(max_completion_tokens, name="max_completion_tokens", minimum=1)
    if not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    if max_tokens_field not in {"max_completion_tokens", "max_tokens"}:
        raise ValueError(f"unsupported max token field: {max_tokens_field}")
    if reasoning_effort not in {None, "none", "low", "medium", "high"}:
        raise ValueError(f"unsupported reasoning effort: {reasoning_effort}")
    if start_index + example_count > PROMPT_CAPACITY:
        raise ValueError(
            f"chunk exceeds the {PROMPT_CAPACITY:,}-example collision-free prompt capacity"
        )

    examples = tuple(
        PlannedExample.create(index, seed)
        for index in range(start_index, start_index + example_count)
    )
    chunk_id = _chunk_id(seed, start_index, example_count)
    requests: list[dict[str, object]] = []
    request_assignments: list[dict[str, object]] = []
    for request_index, example in enumerate(examples):
        custom_id = f"{chunk_id}-r{request_index:06d}"
        body: dict[str, object] = {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": f"{example.prompt}\n\n{GENERATION_OUTPUT_INSTRUCTION}",
                },
            ],
            "temperature": temperature,
            max_tokens_field: max_completion_tokens,
        }
        if reasoning_effort is not None:
            body["reasoning"] = {"effort": reasoning_effort}
        requests.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": body,
            }
        )
        request_assignments.append(
            {
                "custom_id": custom_id,
                "example_id": example.example_id,
                "request_sha256": sha256(_canonical_json(body).encode()).hexdigest(),
            }
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    requests_path = output_directory / "requests.jsonl"
    plan_path = output_directory / "plan.json"
    if requests_path.exists() or plan_path.exists():
        raise FileExistsError(f"chunk plan already exists in {output_directory}")
    _write_jsonl(requests_path, requests)
    _write_json(
        plan_path,
        {
            "format_version": FORMAT_VERSION,
            "recipe_version": RECIPE_VERSION,
            "chunk_id": chunk_id,
            "model": model,
            "provider": provider,
            "seed": seed,
            "start_index": start_index,
            "example_count": example_count,
            "output_mode": "raw-text",
            "temperature": temperature,
            "max_completion_tokens": max_completion_tokens,
            "max_tokens_field": max_tokens_field,
            "reasoning_effort": reasoning_effort,
            "requests_filename": requests_path.name,
            "requests_sha256": file_hash(requests_path),
            "examples": [
                {
                    "example_id": example.example_id,
                    "global_index": example.global_index,
                    "prompt": example.prompt,
                    "prompt_spec": asdict(example.prompt_spec),
                }
                for example in examples
            ],
            "assignments": request_assignments,
        },
    )
    return requests_path, plan_path


def record_sft_dispatch(
    *,
    plan_path: Path,
    base_url: str,
    api_key_environment_variable: str,
    concurrency: int,
    requests_per_minute: int,
    tokens_per_minute: int,
    timeout_seconds: float,
) -> Path:
    """Bind a chunk to the non-secret endpoint and runtime settings used to dispatch it."""
    plan = _read_json(plan_path)
    requests_path = plan_path.parent / _required_string(
        plan.get("requests_filename"), name="requests_filename"
    )
    if file_hash(requests_path) != _required_string(
        plan.get("requests_sha256"), name="requests_sha256"
    ):
        raise ValueError("request file hash does not match the chunk plan")
    normalized_base_url = _required_string(base_url, name="base_url").rstrip("/")
    api_key_environment_variable = _required_string(
        api_key_environment_variable, name="api_key_environment_variable"
    )
    _required_integer(concurrency, name="concurrency", minimum=1)
    _required_integer(requests_per_minute, name="requests_per_minute", minimum=1)
    _required_integer(tokens_per_minute, name="tokens_per_minute", minimum=1)
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")
    dispatch_path = plan_path.parent / "dispatch.json"
    dispatch = {
        "format_version": FORMAT_VERSION,
        "chunk_id": _required_string(plan.get("chunk_id"), name="chunk_id"),
        "plan_sha256": file_hash(plan_path),
        "requests_sha256": file_hash(requests_path),
        "base_url": normalized_base_url,
        "api_key_environment_variable": api_key_environment_variable,
        "concurrency": concurrency,
        "requests_per_minute": requests_per_minute,
        "tokens_per_minute": tokens_per_minute,
        "timeout_seconds": timeout_seconds,
    }
    if dispatch_path.exists():
        if _read_json(dispatch_path) != dispatch:
            raise ValueError("chunk was already dispatched with different runtime settings")
        return dispatch_path
    _write_json(dispatch_path, dispatch)
    return dispatch_path


def _completion_text(record: Mapping[str, object]) -> tuple[str, str]:
    custom_id = _required_string(record.get("custom_id"), name="result custom_id")
    if record.get("error") is not None:
        raise RuntimeError(f"result {custom_id} failed: {record['error']}")
    response = record.get("response")
    if not isinstance(response, dict) or response.get("status_code") != 200:
        raise RuntimeError(f"result {custom_id} does not contain a successful response")
    body = response.get("body")
    if not isinstance(body, dict):
        raise TypeError(f"result {custom_id} body must be an object")
    choices = body.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise ValueError(f"result {custom_id} must contain exactly one choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise TypeError(f"result {custom_id} message must be an object")
    content = _required_string(message.get("content"), name=f"result {custom_id} content")
    return custom_id, content.replace("\r\n", "\n").replace("\r", "\n")


def _token_counts(tokenizer: Tokenizer, prompt: str, response: str) -> dict[str, int]:
    prompt_tokens = len(tokenizer.encode(prompt, add_special_tokens=False).ids)
    response_tokens = len(tokenizer.encode(response, add_special_tokens=False).ids)
    return {
        "prompt": prompt_tokens,
        "response": response_tokens,
        "formatted": prompt_tokens + response_tokens + 4,
        "training_input": prompt_tokens + response_tokens + 3,
        "supervised": response_tokens + 1,
    }


def _provider_usage(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
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
            totals[name] += _required_integer(usage.get(name), name=f"provider usage {name}")
    return totals


def _observed_models(records: Sequence[Mapping[str, object]]) -> tuple[tuple[str, ...], int]:
    models: set[str] = set()
    missing = 0
    for record in records:
        response = record.get("response")
        body = response.get("body") if isinstance(response, dict) else None
        model = body.get("model") if isinstance(body, dict) else None
        if isinstance(model, str) and model.strip():
            models.add(model.strip())
        else:
            missing += 1
    return tuple(sorted(models)), missing


def _local_rejection_reasons(response: str, prompt_spec: Mapping[str, object]) -> tuple[str, ...]:
    reasons: list[str] = []
    if "minimum_lines" in prompt_spec and "maximum_lines" in prompt_spec:
        minimum_lines = _required_integer(prompt_spec.get("minimum_lines"), name="minimum_lines")
        maximum_lines = _required_integer(prompt_spec.get("maximum_lines"), name="maximum_lines")
    else:
        length_label = _required_string(prompt_spec.get("length_label"), name="length_label")
        if length_label not in LENGTH_LINE_BOUNDS:
            raise ValueError(f"unsupported prompt length label: {length_label}")
        minimum_lines, maximum_lines = LENGTH_LINE_BOUNDS[length_label]
    if maximum_lines < minimum_lines:
        raise ValueError("maximum_lines must be greater than or equal to minimum_lines")
    nonempty_lines = tuple(line.strip() for line in response.splitlines() if line.strip())
    if not minimum_lines <= len(nonempty_lines) <= maximum_lines:
        reasons.append(f"line_count={len(nonempty_lines)}_expected={minimum_lines}-{maximum_lines}")
    word_count = len(WORD.findall(response))
    if word_count < len(nonempty_lines) or word_count > 600:
        reasons.append(f"word_count={word_count}")
    if "```" in response:
        reasons.append("markdown_fence")
    if any(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t"
        for character in response
    ):
        reasons.append("control_character")
    normalized = unicodedata.normalize("NFKC", response).casefold()
    if any(marker in normalized for marker in REFUSAL_MARKERS):
        reasons.append("refusal_boilerplate")
    normalized_lines = [" ".join(line.casefold().split()) for line in nonempty_lines]
    if sum(line in {"---", "***"} for line in normalized_lines) > 1:
        reasons.append("multiple_poem_separator")
    return tuple(reasons)


def _legacy_salvage_prompt(
    prompt_spec: Mapping[str, object],
    *,
    line_count: int,
) -> tuple[str, dict[str, object]]:
    subject = _required_string(prompt_spec.get("subject"), name="subject")
    tone = _required_string(prompt_spec.get("tone"), name="tone")
    voice = _required_string(prompt_spec.get("voice"), name="voice")
    prompt = (
        f"Write an original English poem about {subject}. Make it {tone}, using {voice}. "
        f"Use exactly {line_count} non-empty lines."
    )
    return prompt, {
        "subject": subject,
        "form": "an original English poem",
        "tone": tone,
        "length_label": "observed",
        "length_instruction": f"exactly {line_count} non-empty lines",
        "minimum_lines": line_count,
        "maximum_lines": line_count,
        "voice": voice,
        "technique": "Follow the requested subject, tone, voice, and line count.",
    }


def finalize_sft_chunk(
    *,
    plan_path: Path,
    results_path: Path,
    tokenizer_path: Path,
    output_directory: Path,
    expected_tokenizer_sha256: str = TRACK1_SFT_TOKENIZER_SHA256,
    allow_partial: bool = False,
) -> Path:
    """Validate a complete chunk and emit canonical SFT examples with exact token counts."""
    plan = _read_json(plan_path)
    recipe_version = _required_string(plan.get("recipe_version"), name="recipe_version")
    if (
        plan.get("format_version") != FORMAT_VERSION
        or recipe_version not in SUPPORTED_RECIPE_VERSIONS
    ):
        raise ValueError("unsupported SFT chunk plan format")
    if plan.get("output_mode") != "raw-text":
        raise ValueError("SFT chunk plan must use raw-text output")
    requests_path = plan_path.parent / _required_string(
        plan.get("requests_filename"), name="requests_filename"
    )
    if file_hash(requests_path) != _required_string(
        plan.get("requests_sha256"), name="requests_sha256"
    ):
        raise ValueError("request file hash does not match the chunk plan")
    dispatch_path = plan_path.parent / "dispatch.json"
    dispatch = _read_json(dispatch_path)
    if dispatch.get("chunk_id") != plan.get("chunk_id"):
        raise ValueError("dispatch does not match the chunk plan")
    if dispatch.get("plan_sha256") != file_hash(plan_path):
        raise ValueError("dispatch plan hash does not match the chunk plan")
    if dispatch.get("requests_sha256") != file_hash(requests_path):
        raise ValueError("dispatch request hash does not match the chunk plan")

    assignments_value = plan.get("assignments")
    examples_value = plan.get("examples")
    if not isinstance(assignments_value, list) or not isinstance(examples_value, list):
        raise TypeError("chunk plan assignments and examples must be arrays")
    planned_examples: dict[str, dict[str, object]] = {}
    for value in examples_value:
        if not isinstance(value, dict):
            raise TypeError("planned example must be an object")
        example = cast(dict[str, object], value)
        example_id = _required_string(example.get("example_id"), name="planned example_id")
        if example_id in planned_examples:
            raise ValueError(f"duplicate planned example {example_id}")
        planned_examples[example_id] = example

    expected_by_request: dict[str, str] = {}
    for value in assignments_value:
        if not isinstance(value, dict):
            raise TypeError("request assignment must be an object")
        assignment = cast(dict[str, object], value)
        custom_id = _required_string(assignment.get("custom_id"), name="assignment custom_id")
        example_id = _required_string(
            assignment.get("example_id"), name=f"assignment {custom_id} example_id"
        )
        if custom_id in expected_by_request:
            raise ValueError(f"duplicate assignment {custom_id}")
        expected_by_request[custom_id] = example_id

    generated: dict[str, tuple[str, str]] = {}
    seen_requests: set[str] = set()
    result_records = _read_jsonl(results_path)
    for record in result_records:
        custom_id, response = _completion_text(record)
        if custom_id not in expected_by_request:
            raise ValueError(f"unexpected result {custom_id}")
        if custom_id in seen_requests:
            raise ValueError(f"duplicate result {custom_id}")
        seen_requests.add(custom_id)
        example_id = expected_by_request[custom_id]
        if example_id in generated:
            raise ValueError(f"duplicate generated example {example_id}")
        generated[example_id] = (custom_id, response)

    missing_requests = set(expected_by_request).difference(seen_requests)
    missing_examples = set(planned_examples).difference(generated)
    if missing_requests and not allow_partial:
        raise ValueError(f"results are missing requests: {sorted(missing_requests)}")
    if missing_examples and not allow_partial:
        raise ValueError(f"results are missing examples: {sorted(missing_examples)}")

    tokenizer_hash = file_hash(tokenizer_path)
    if tokenizer_hash != expected_tokenizer_sha256:
        raise ValueError(
            "tokenizer does not match the frozen Track 1 SFT tokenizer: "
            f"expected {expected_tokenizer_sha256}, got {tokenizer_hash}"
        )
    tokenizer = load_tokenizer(tokenizer_path)
    chunk_id = _required_string(plan.get("chunk_id"), name="chunk_id")
    model = _required_string(plan.get("model"), name="model")
    provider = _required_string(plan.get("provider"), name="provider")
    seed = _required_integer(plan.get("seed"), name="seed")
    output_records: list[dict[str, object]] = []
    rejections: list[dict[str, object]] = []
    response_hashes: set[str] = set()
    totals = Counter[str]()
    adjusted_prompt_count = 0
    for example_id, planned in planned_examples.items():
        if example_id not in generated:
            continue
        custom_id, response = generated[example_id]
        prompt_spec = planned.get("prompt_spec")
        if not isinstance(prompt_spec, dict):
            raise TypeError(f"prompt_spec {example_id} must be an object")
        rejection_reasons = list(_local_rejection_reasons(response, prompt_spec))
        prompt_adjustment: dict[str, object] | None = None
        if recipe_version == LEGACY_RECIPE_VERSION:
            rejection_reasons = [
                reason for reason in rejection_reasons if not reason.startswith("line_count=")
            ]
            nonempty_line_count = sum(bool(line.strip()) for line in response.splitlines())
            prompt, training_prompt_spec = _legacy_salvage_prompt(
                prompt_spec,
                line_count=nonempty_line_count,
            )
            prompt_adjustment = {
                "kind": "legacy-v1-observed-shape",
                "source_prompt_sha256": sha256(
                    _required_string(
                        planned.get("prompt"),
                        name=f"prompt {example_id}",
                    ).encode()
                ).hexdigest(),
            }
        else:
            prompt = _required_string(planned.get("prompt"), name=f"prompt {example_id}")
            training_prompt_spec = prompt_spec
        response_hash = _normalized_text_hash(response)
        if response_hash in response_hashes:
            rejection_reasons.append("duplicate_response")
        if rejection_reasons:
            rejections.append(
                {
                    "example_id": example_id,
                    "request_id": custom_id,
                    "reasons": rejection_reasons,
                    "response_sha256": sha256(response.encode()).hexdigest(),
                }
            )
            continue
        response_hashes.add(response_hash)
        if prompt_adjustment is not None:
            adjusted_prompt_count += 1
        counts = _token_counts(tokenizer, prompt, response)
        totals.update(counts)
        provenance: dict[str, object] = {
            "kind": "synthetic",
            "recipe_version": recipe_version,
            "chunk_id": chunk_id,
            "request_id": custom_id,
            "generator_model": model,
            "provider": provider,
            "seed": seed,
        }
        if prompt_adjustment is not None:
            provenance["prompt_adjustment"] = prompt_adjustment
        output_records.append(
            {
                "format_version": FORMAT_VERSION,
                "example_id": example_id,
                "messages": [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": response},
                ],
                "prompt_spec": training_prompt_spec,
                "token_counts": counts,
                "provenance": provenance,
            }
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    examples_path = output_directory / "examples.jsonl"
    rejections_path = output_directory / "rejections.jsonl"
    receipt_path = output_directory / "receipt.json"
    if examples_path.exists() or rejections_path.exists() or receipt_path.exists():
        raise FileExistsError(f"finalized chunk already exists in {output_directory}")
    _write_jsonl(examples_path, output_records)
    _write_jsonl(rejections_path, rejections)
    observed_models, missing_observed_models = _observed_models(result_records)
    _write_json(
        receipt_path,
        {
            "format_version": FORMAT_VERSION,
            "recipe_version": recipe_version,
            "chunk_id": chunk_id,
            "model": model,
            "provider": provider,
            "seed": seed,
            "planned_example_count": len(planned_examples),
            "completed_result_count": len(generated),
            "missing_result_count": len(missing_requests),
            "partial": bool(missing_requests),
            "adjusted_prompt_count": adjusted_prompt_count,
            "example_count": len(output_records),
            "rejected_example_count": len(rejections),
            "token_counts": dict(totals),
            "provider_usage": _provider_usage(result_records),
            "observed_models": observed_models,
            "missing_observed_model_records": missing_observed_models,
            "plan_sha256": file_hash(plan_path),
            "dispatch_sha256": file_hash(dispatch_path),
            "requests_sha256": file_hash(requests_path),
            "results_sha256": file_hash(results_path),
            "tokenizer_sha256": tokenizer_hash,
            "examples_sha256": file_hash(examples_path),
            "examples_filename": examples_path.name,
            "rejections_sha256": file_hash(rejections_path),
            "rejections_filename": rejections_path.name,
        },
    )
    return receipt_path


def _validate_receipt_set(
    receipts: Sequence[Mapping[str, object]], *, expected_tokenizer_sha256: str
) -> tuple[str, int, tuple[str, ...]]:
    if not receipts:
        raise ValueError("at least one chunk receipt is required")
    for receipt in receipts:
        if receipt.get("format_version") != FORMAT_VERSION:
            raise ValueError("chunk receipts use an unsupported format or prompt recipe")
        recipe_version = _required_string(receipt.get("recipe_version"), name="recipe_version")
        if recipe_version not in SUPPORTED_RECIPE_VERSIONS:
            raise ValueError("chunk receipts use an unsupported format or prompt recipe")
    tokenizer_hashes = {
        _required_string(receipt.get("tokenizer_sha256"), name="tokenizer_sha256")
        for receipt in receipts
    }
    if tokenizer_hashes != {expected_tokenizer_sha256}:
        raise ValueError("chunk receipts do not use the frozen Track 1 SFT tokenizer")
    seeds = {_required_integer(receipt.get("seed"), name="seed") for receipt in receipts}
    if len(seeds) != 1:
        raise ValueError("chunk receipts use different corpus seeds")
    chunk_ids = [_required_string(receipt.get("chunk_id"), name="chunk_id") for receipt in receipts]
    if len(chunk_ids) != len(set(chunk_ids)):
        raise ValueError("duplicate chunk receipt")
    recipe_versions = tuple(
        sorted(
            {
                _required_string(receipt.get("recipe_version"), name="recipe_version")
                for receipt in receipts
            }
        )
    )
    return next(iter(tokenizer_hashes)), next(iter(seeds)), recipe_versions


def summarize_sft_chunks(
    receipt_paths: Sequence[Path],
    *,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    output_path: Path | None = None,
    expected_tokenizer_sha256: str = TRACK1_SFT_TOKENIZER_SHA256,
) -> dict[str, object]:
    """Verify compatible finalized chunks and report progress toward the token target."""
    _required_integer(target_tokens, name="target_tokens", minimum=1)
    if not receipt_paths:
        raise ValueError("at least one chunk receipt is required")
    receipts = [_read_json(path) for path in receipt_paths]
    tokenizer_hash, seed, recipe_versions = _validate_receipt_set(
        receipts, expected_tokenizer_sha256=expected_tokenizer_sha256
    )
    chunk_ids = [_required_string(receipt.get("chunk_id"), name="chunk_id") for receipt in receipts]

    totals = Counter[str]()
    total_examples = 0
    by_model: Counter[str] = Counter()
    for receipt in receipts:
        total_examples += _required_integer(receipt.get("example_count"), name="example_count")
        model = _required_string(receipt.get("model"), name="model")
        by_model[model] += _required_integer(receipt.get("example_count"), name="example_count")
        counts = receipt.get("token_counts")
        if not isinstance(counts, dict):
            raise TypeError("receipt token_counts must be an object")
        for name in ("prompt", "response", "formatted", "training_input", "supervised"):
            totals[name] += _required_integer(counts.get(name), name=f"token_counts.{name}")

    formatted_tokens = totals["formatted"]
    summary: dict[str, object] = {
        "format_version": FORMAT_VERSION,
        "recipe_versions": recipe_versions,
        "target_formatted_tokens": target_tokens,
        "formatted_tokens": formatted_tokens,
        "remaining_formatted_tokens": max(0, target_tokens - formatted_tokens),
        "target_fraction": formatted_tokens / target_tokens,
        "example_count": total_examples,
        "token_counts": dict(totals),
        "examples_by_model": dict(sorted(by_model.items())),
        "tokenizer_sha256": tokenizer_hash,
        "seed": seed,
        "chunks": chunk_ids,
    }
    if output_path is not None:
        _write_json(output_path, summary)
    return summary


def assemble_sft_dataset(
    receipt_paths: Sequence[Path],
    *,
    output_directory: Path,
    target_tokens: int = DEFAULT_TARGET_TOKENS,
    target_metric: TargetMetric = "formatted",
    allow_under_target: bool = False,
    expected_tokenizer_sha256: str = TRACK1_SFT_TOKENIZER_SHA256,
) -> Path:
    """Verify, deduplicate, order, and budget finalized chunks into one SFT dataset."""
    _required_integer(target_tokens, name="target_tokens", minimum=1)
    if target_metric not in {"formatted", "supervised"}:
        raise ValueError(f"unsupported target metric: {target_metric}")
    receipt_entries: list[tuple[Path, dict[str, object]]] = [
        (path, _read_json(path)) for path in receipt_paths
    ]
    tokenizer_hash, seed, recipe_versions = _validate_receipt_set(
        [receipt for _, receipt in receipt_entries],
        expected_tokenizer_sha256=expected_tokenizer_sha256,
    )

    candidates: list[dict[str, object]] = []
    source_receipts: list[dict[str, str]] = []
    for receipt_path, receipt in receipt_entries:
        chunk_id = _required_string(receipt.get("chunk_id"), name="chunk_id")
        receipt_recipe = _required_string(receipt.get("recipe_version"), name="recipe_version")
        receipt_model = _required_string(receipt.get("model"), name="model")
        receipt_provider = _required_string(receipt.get("provider"), name="provider")
        examples_path = receipt_path.parent / _required_string(
            receipt.get("examples_filename"), name="examples_filename"
        )
        expected_hash = _required_string(receipt.get("examples_sha256"), name="examples_sha256")
        if file_hash(examples_path) != expected_hash:
            raise ValueError(f"examples hash does not match receipt: {examples_path}")
        records = _read_jsonl(examples_path)
        expected_count = _required_integer(receipt.get("example_count"), name="example_count")
        if len(records) != expected_count:
            raise ValueError(f"example count does not match receipt: {examples_path}")
        rejections_path = receipt_path.parent / _required_string(
            receipt.get("rejections_filename"), name="rejections_filename"
        )
        if file_hash(rejections_path) != _required_string(
            receipt.get("rejections_sha256"), name="rejections_sha256"
        ):
            raise ValueError(f"rejections hash does not match receipt: {rejections_path}")
        for record in records:
            provenance = record.get("provenance")
            if not isinstance(provenance, dict):
                raise TypeError("SFT example provenance must be an object")
            if (
                provenance.get("chunk_id") != chunk_id
                or provenance.get("recipe_version") != receipt_recipe
                or provenance.get("generator_model") != receipt_model
                or provenance.get("provider") != receipt_provider
                or provenance.get("seed") != seed
            ):
                raise ValueError(f"example provenance does not match receipt: {examples_path}")
        candidates.extend(records)
        source_receipts.append(
            {
                "chunk_id": chunk_id,
                "recipe_version": receipt_recipe,
                "receipt_sha256": file_hash(receipt_path),
                "examples_sha256": expected_hash,
            }
        )

    def global_index(record: Mapping[str, object]) -> tuple[int, int]:
        example_id = _required_string(record.get("example_id"), name="example_id")
        prefixes = (("synthetic-sft-", 1), ("synthetic-sft-v2-", 2))
        for prefix, recipe_order in reversed(prefixes):
            suffix = example_id.removeprefix(prefix)
            if suffix != example_id and suffix.isdigit():
                return recipe_order, int(suffix)
        raise ValueError(f"invalid synthetic SFT example ID: {example_id}")

    candidates.sort(key=global_index)
    validated: list[tuple[dict[str, object], dict[str, int], str]] = []
    seen_ids: set[str] = set()
    seen_responses: dict[str, str] = {}
    duplicate_responses: list[dict[str, str]] = []
    for record in candidates:
        example_id = _required_string(record.get("example_id"), name="example_id")
        if example_id in seen_ids:
            raise ValueError(f"overlapping chunks contain example ID {example_id}")
        seen_ids.add(example_id)
        messages = record.get("messages")
        if not isinstance(messages, list) or len(messages) != 2:
            raise ValueError(f"example {example_id} must contain two messages")
        user = messages[0]
        assistant = messages[1]
        if not isinstance(user, dict) or user.get("role") != "user":
            raise ValueError(f"example {example_id} lacks its user prompt")
        if not isinstance(assistant, dict) or assistant.get("role") != "assistant":
            raise ValueError(f"example {example_id} lacks its assistant response")
        response = _required_string(assistant.get("content"), name=f"response {example_id}")
        response_hash = _normalized_text_hash(response)
        first_example_id = seen_responses.get(response_hash)
        if first_example_id is not None:
            duplicate_responses.append(
                {
                    "example_id": example_id,
                    "duplicate_of": first_example_id,
                    "response_sha256": response_hash,
                }
            )
            continue
        seen_responses[response_hash] = example_id
        counts = record.get("token_counts")
        if not isinstance(counts, dict):
            raise TypeError(f"example {example_id} token_counts must be an object")
        parsed_counts = {
            name: _required_integer(counts.get(name), name=f"token_counts.{name}")
            for name in ("prompt", "response", "formatted", "training_input", "supervised")
        }
        if (
            parsed_counts["formatted"] != parsed_counts["prompt"] + parsed_counts["response"] + 4
            or parsed_counts["training_input"] != parsed_counts["formatted"] - 1
            or parsed_counts["supervised"] != parsed_counts["response"] + 1
        ):
            raise ValueError(f"example {example_id} has inconsistent token counts")
        provenance = record.get("provenance")
        if not isinstance(provenance, dict):
            raise TypeError(f"example {example_id} provenance must be an object")
        model = _required_string(provenance.get("generator_model"), name="generator_model")
        validated.append((record, parsed_counts, model))

    accepted: list[dict[str, object]] = []
    totals = Counter[str]()
    by_model: Counter[str] = Counter()
    reached_target = False
    for record, parsed_counts, model in validated:
        totals.update(parsed_counts)
        by_model[model] += 1
        accepted.append(record)
        if totals[target_metric] >= target_tokens:
            reached_target = True
            break

    if not reached_target and not allow_under_target:
        raise ValueError(
            f"assembled data has {totals[target_metric]:,} {target_metric} tokens, "
            f"below the required {target_tokens:,}; use summarize to check progress"
        )

    output_directory.mkdir(parents=True, exist_ok=True)
    dataset_path = output_directory / "dataset.jsonl"
    receipt_path = output_directory / "receipt.json"
    duplicates_path = output_directory / "duplicate_responses.jsonl"
    if dataset_path.exists() or receipt_path.exists() or duplicates_path.exists():
        raise FileExistsError(f"assembled SFT dataset already exists in {output_directory}")
    _write_jsonl(dataset_path, accepted)
    _write_jsonl(duplicates_path, duplicate_responses)
    actual_target_tokens = totals[target_metric]
    _write_json(
        receipt_path,
        {
            "format_version": FORMAT_VERSION,
            "recipe_versions": recipe_versions,
            "target_metric": target_metric,
            "target_tokens": target_tokens,
            "target_reached": reached_target,
            "actual_target_tokens": actual_target_tokens,
            "target_overshoot": max(0, actual_target_tokens - target_tokens),
            "token_counts": dict(totals),
            "example_count": len(accepted),
            "duplicate_response_count": len(duplicate_responses),
            "examples_by_model": dict(sorted(by_model.items())),
            "tokenizer_sha256": tokenizer_hash,
            "seed": seed,
            "source_receipts": sorted(source_receipts, key=lambda value: value["chunk_id"]),
            "dataset_sha256": file_hash(dataset_path),
            "duplicate_responses_sha256": file_hash(duplicates_path),
        },
    )
    return receipt_path
