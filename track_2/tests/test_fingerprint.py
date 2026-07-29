from __future__ import annotations

import torch

from genome.fingerprint import corpus_fingerprint, merge_fingerprints


def test_corpus_fingerprint_is_deterministic_and_semantic() -> None:
    first = corpus_fingerprint([[1, 2, 3], [3, 4]], raw_texts=["hello", "world"])
    second = corpus_fingerprint([[1, 2, 3], [3, 4]], raw_texts=["hello", "world"])
    changed = corpus_fingerprint([[1, 2, 4], [3, 4]], raw_texts=["hello", "world"])
    assert first.fingerprint_id == second.fingerprint_id
    assert first.fingerprint_id != changed.fingerprint_id
    assert torch.equal(first.tensors["corpus.unigram"], second.tensors["corpus.unigram"])


def test_merge_preserves_parts_without_hash_vectors() -> None:
    a = corpus_fingerprint([[1, 2]])
    b = corpus_fingerprint([[3, 4]])
    merged = merge_fingerprints(a, b)
    assert all("sha" not in name.lower() and "hash" not in name.lower() for name in merged.tensors)
