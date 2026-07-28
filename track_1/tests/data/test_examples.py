from __future__ import annotations

from typing import ClassVar

from poetry50m.data.examples import build_conditional_examples
from poetry50m.data.schema import (
    ConditionalExample,
    ContentBlock,
    PromptRecord,
    Provenance,
    SourceDocument,
    ThoughtRecord,
)


class _ComparisonBoundId(str):
    comparisons: ClassVar[int] = 0
    comparison_limit: ClassVar[int] = 100_000

    def __eq__(self, other: object) -> bool:
        type(self).comparisons += 1
        if type(self).comparisons > type(self).comparison_limit:
            raise AssertionError("conditional example construction scanned all prompt records")
        return super().__eq__(other)

    __hash__ = str.__hash__


def _document(
    document_id: str,
    blocks: tuple[ContentBlock, ...],
) -> SourceDocument:
    return SourceDocument(
        document_id=document_id,
        provenance=Provenance(
            work=f"Work {document_id}",
            author="Fixture Author",
            licence="synthetic",
            source="fixture",
            rights_status="synthetic",
        ),
        text="\n\n".join(block.text for block in blocks),
        blocks=blocks,
    )


def _relation(example: ConditionalExample) -> tuple[str, str | None, str | None, str | None, str]:
    return (
        example.document_id,
        example.poem_id,
        example.prompt_id,
        example.thought_id,
        example.poem_target,
    )


def test_indexed_conditioning_relations_match_pairwise_selection_semantics():
    documents = (
        _document(
            "document-a",
            (
                ContentBlock("a:poem:1", "poem", "First complete poem.", poem_id="poem-a"),
                ContentBlock(
                    "a:stanza:1",
                    "stanza",
                    "First stanza.",
                    poem_id="poem-a",
                    stanza_index=0,
                ),
                ContentBlock("a:poem:2", "poem", "Second complete poem.", poem_id="poem-b"),
                ContentBlock(
                    "a:paragraph",
                    "paragraph",
                    "Excluded prose paragraph.",
                    paragraph_index=0,
                ),
            ),
        ),
        _document(
            "document-b",
            (ContentBlock("b:poem", "poem", "Third complete poem.", poem_id="poem-c"),),
        ),
    )
    prompts = (
        PromptRecord(
            "prompt-b", "document-b", "Write the third poem.", "title", "fixture", "poem-c"
        ),
        PromptRecord(
            "prompt-a-specific",
            "document-a",
            "Write the first poem.",
            "title",
            "fixture",
            "poem-a",
        ),
        PromptRecord(
            "prompt-a-general",
            "document-a",
            "Write a poem from this work.",
            "theme",
            "fixture",
        ),
        PromptRecord(
            "prompt-a-second",
            "document-a",
            "Write the second poem.",
            "title",
            "fixture",
            "poem-b",
        ),
    )
    thoughts = (
        ThoughtRecord("thought-b", "document-b", "Third thought.", "editorial", "fixture"),
        ThoughtRecord("thought-a-1", "document-a", "First thought.", "editorial", "fixture"),
        ThoughtRecord("thought-a-2", "document-a", "Second thought.", "editorial", "fixture"),
    )

    actual = build_conditional_examples(documents, prompts=prompts, thoughts=thoughts)
    expected_relations = {
        (
            document.document_id,
            block.poem_id,
            prompt.prompt_id,
            thought.thought_id if thought is not None else None,
            block.text,
        )
        for document in sorted(documents, key=lambda item: item.document_id)
        for block in document.blocks
        if block.kind in {"poem", "stanza"}
        for prompt in prompts
        if prompt.document_id == document.document_id and prompt.poem_id in {None, block.poem_id}
        for thought in (
            None,
            *(record for record in thoughts if record.document_id == document.document_id),
        )
    }

    assert {_relation(example) for example in actual} == expected_relations
    assert tuple(example.example_id for example in actual) == tuple(
        sorted(example.example_id for example in actual)
    )


def test_thousands_of_documents_do_not_scan_all_prompt_records():
    document_count = 5_000
    documents = tuple(
        _document(
            f"document-{index}",
            (
                ContentBlock(
                    f"document-{index}:poem",
                    "poem",
                    f"Unique poem text {index}.",
                    poem_id=f"poem-{index}",
                ),
            ),
        )
        for index in range(document_count)
    )
    prompts = tuple(
        PromptRecord(
            prompt_id=f"prompt-{index}",
            document_id=_ComparisonBoundId(f"document-{index}"),
            prompt=f"Write unique poem {index}.",
            method="title",
            source_attribution="fixture",
            poem_id=f"poem-{index}",
        )
        for index in range(document_count)
    )
    _ComparisonBoundId.comparisons = 0

    examples = build_conditional_examples(documents, prompts=prompts)

    assert len(examples) == document_count
    assert _ComparisonBoundId.comparisons < document_count * 10
