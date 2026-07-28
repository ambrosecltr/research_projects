"""Build actual prompt/thought/poem continuation examples from corpus blocks."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from hashlib import sha256
from typing import TypeVar

from .schema import (
    ConditionalExample,
    ContentBlock,
    CrossDocumentPairing,
    PoetryNTPExample,
    PromptRecord,
    ProseNTPExample,
    SourceDocument,
    ThoughtRecord,
)

T = TypeVar("T")


def _unique(records: Iterable[T], identifier: Callable[[T], str], label: str) -> dict[str, T]:
    result: dict[str, T] = {}
    for record in records:
        key = identifier(record)
        if key in result:
            raise ValueError(f"duplicate {label} {key!r}")
        result[key] = record
    return result


def _example_id(
    document_id: str, poem_id: str | None, prompt: str, thought: str | None, target: str
) -> str:
    content = "\0".join((document_id, poem_id or "", prompt, thought or "", target))
    return sha256(content.encode()).hexdigest()[:24]


def _same_training_relation(left: ConditionalExample, right: ConditionalExample) -> bool:
    """Compare the conditioning relation while ignoring optional pairing provenance."""
    return (
        left.document_id,
        left.poem_id,
        left.prompt,
        left.thought,
        left.poem_target,
        left.prompt_id,
        left.thought_id,
        left.prompt_document_id,
        left.thought_document_id,
        left.loss_on_poem_only,
        left.split,
    ) == (
        right.document_id,
        right.poem_id,
        right.prompt,
        right.thought,
        right.poem_target,
        right.prompt_id,
        right.thought_id,
        right.prompt_document_id,
        right.thought_document_id,
        right.loss_on_poem_only,
        right.split,
    )


def build_conditional_examples(
    documents: Iterable[SourceDocument],
    *,
    prompts: Iterable[PromptRecord],
    thoughts: Iterable[ThoughtRecord] = (),
    pairings: Iterable[CrossDocumentPairing] = (),
) -> tuple[ConditionalExample, ...]:
    """Build prompt-only relations plus optional thought-conditioned variants."""

    prompt_by_id = _unique(prompts, lambda item: item.prompt_id, "prompt_id")
    thought_by_id = _unique(thoughts, lambda item: item.thought_id, "thought_id")
    documents_by_id = _unique(documents, lambda item: item.document_id, "document_id")
    prompt_positions = {prompt_id: position for position, prompt_id in enumerate(prompt_by_id)}
    mutable_prompts_by_relation: dict[tuple[str, str | None], list[PromptRecord]] = {}
    for indexed_prompt in prompt_by_id.values():
        mutable_prompts_by_relation.setdefault(
            (indexed_prompt.document_id, indexed_prompt.poem_id), []
        ).append(indexed_prompt)
    prompts_by_relation = {
        relation: tuple(records) for relation, records in mutable_prompts_by_relation.items()
    }
    mutable_thoughts_by_document: dict[str, list[ThoughtRecord]] = {}
    for indexed_thought in thought_by_id.values():
        mutable_thoughts_by_document.setdefault(indexed_thought.document_id, []).append(
            indexed_thought
        )
    thoughts_by_document = {
        document_id: tuple(records) for document_id, records in mutable_thoughts_by_document.items()
    }
    examples: dict[str, ConditionalExample] = {}

    def add(example: ConditionalExample) -> None:
        existing = examples.get(example.example_id)
        if existing is None or existing == example:
            examples[example.example_id] = example
            return
        if _same_training_relation(existing, example):
            if existing.pairing_id is not None and example.pairing_id is None:
                return
            if existing.pairing_id is None and example.pairing_id is not None:
                examples[example.example_id] = example
                return
            if existing.pairing_id is not None and example.pairing_id is not None:
                raise ValueError(
                    f"explicit pairings {existing.pairing_id!r} and {example.pairing_id!r} "
                    "produce the same conditional example"
                )
        raise ValueError(f"conflicting conditional examples share ID {example.example_id}")

    def make_example(
        document: SourceDocument,
        block: ContentBlock,
        prompt_record: PromptRecord,
        thought_record: ThoughtRecord | None,
        pairing: CrossDocumentPairing | None = None,
    ) -> ConditionalExample:
        thought_text = thought_record.text if thought_record is not None else None
        return ConditionalExample(
            example_id=_example_id(
                document.document_id,
                block.poem_id,
                prompt_record.prompt,
                thought_text,
                block.text,
            ),
            document_id=document.document_id,
            poem_id=block.poem_id,
            prompt=prompt_record.prompt,
            thought=thought_text,
            poem_target=block.text,
            prompt_id=prompt_record.prompt_id,
            thought_id=thought_record.thought_id if thought_record is not None else None,
            prompt_document_id=prompt_record.document_id,
            thought_document_id=(
                thought_record.document_id if thought_record is not None else None
            ),
            pairing_id=pairing.pairing_id if pairing is not None else None,
            transformation_lineage=(pairing.transformation_lineage if pairing is not None else ()),
        )

    pairing_records = tuple(
        sorted(
            _unique(pairings, lambda item: item.pairing_id, "pairing_id").values(),
            key=lambda item: item.pairing_id,
        )
    )
    blocks_by_target = (
        {
            (document.document_id, block.block_id): block
            for document in documents_by_id.values()
            for block in document.blocks
        }
        if pairing_records
        else {}
    )
    for pairing in pairing_records:
        document = documents_by_id.get(pairing.target_document_id)
        prompt_record = prompt_by_id.get(pairing.prompt_id)
        thought_record = thought_by_id.get(pairing.thought_id) if pairing.thought_id else None
        if (
            document is None
            or prompt_record is None
            or (pairing.thought_id and thought_record is None)
        ):
            raise ValueError(
                f"pairing {pairing.pairing_id} refers to a missing document, prompt, or thought"
            )
        block = blocks_by_target.get((pairing.target_document_id, pairing.target_block_id))
        if block is None or block.kind not in {"poem", "stanza"}:
            raise ValueError(f"pairing {pairing.pairing_id} requires a poem or stanza target block")
        add(make_example(document, block, prompt_record, None, pairing))
        if thought_record is not None:
            add(make_example(document, block, prompt_record, thought_record, pairing))
    for document in sorted(documents_by_id.values(), key=lambda item: item.document_id):
        for block in document.blocks:
            if block.kind not in {"poem", "stanza"}:
                continue
            general_prompts = prompts_by_relation.get((document.document_id, None), ())
            specific_prompts = (
                prompts_by_relation.get((document.document_id, block.poem_id), ())
                if block.poem_id is not None
                else ()
            )
            matching_prompts = tuple(
                sorted(
                    (*general_prompts, *specific_prompts),
                    key=lambda record: prompt_positions[record.prompt_id],
                )
            )
            matching_thoughts = thoughts_by_document.get(document.document_id, ())
            if not matching_prompts:
                continue
            for prompt_record in matching_prompts:
                add(make_example(document, block, prompt_record, None))
                for thought_record in matching_thoughts:
                    add(make_example(document, block, prompt_record, thought_record))
    return tuple(sorted(examples.values(), key=lambda item: item.example_id))


def build_auxiliary_prose_ntp_examples(
    documents: Iterable[SourceDocument],
) -> tuple[ProseNTPExample, ...]:
    """Expose a separately named prose objective; it is never passed off as poetry."""
    examples: list[ProseNTPExample] = []
    for document in documents:
        for block in document.blocks:
            if block.kind in {"paragraph", "document"}:
                example_id = sha256(
                    f"{document.document_id}\0{block.block_id}".encode()
                ).hexdigest()[:24]
                examples.append(
                    ProseNTPExample(example_id, document.document_id, block.block_id, block.text)
                )
    return tuple(sorted(examples, key=lambda item: item.example_id))


def build_poetry_ntp_examples(documents: Iterable[SourceDocument]) -> tuple[PoetryNTPExample, ...]:
    """Expose Gutenberg book verse as raw NTP without inventing poem-level prompts."""
    examples: list[PoetryNTPExample] = []
    for document in documents:
        if document.metadata.get("training_role") != "unconditional_book_verse_ntp":
            continue
        for block in document.blocks:
            if block.kind != "verse_document":
                continue
            example_id = sha256(f"{document.document_id}\0{block.block_id}".encode()).hexdigest()[
                :24
            ]
            examples.append(
                PoetryNTPExample(example_id, document.document_id, block.block_id, block.text)
            )
    return tuple(sorted(examples, key=lambda item: item.example_id))
