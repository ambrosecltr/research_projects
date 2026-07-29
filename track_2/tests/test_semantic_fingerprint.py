from __future__ import annotations

import torch

from genome.semantic_fingerprint import (
    CorpusFingerprintBuilder,
    SemanticFingerprintConfig,
    activation_probe_fingerprint,
    gradient_probe_fingerprint,
)


def build_fixture(
    sequences: list[list[int]],
    *,
    provenance_sha256: str = "a" * 64,
) -> tuple[torch.Tensor, dict]:
    builder = CorpusFingerprintBuilder(
        vocab_size=32,
        config=SemanticFingerprintConfig(
            token_sketch_dim=32,
            bigram_sketch_dim=32,
            gradient_sketch_dim_per_role=16,
            length_bin_edges=(2, 4, 8),
            seed=17,
        ),
    )
    for index, sequence in enumerate(sequences):
        builder.update_tokens(sequence)
        builder.update_bytes(f"document {index}: {sequence}")
    result = builder.finalize(
        provenance={"repository": "org/corpus", "sha256": provenance_sha256}
    )
    return result.flattened(), result.manifest


def test_corpus_fingerprint_is_deterministic_and_semantic() -> None:
    first, first_manifest = build_fixture([[1, 2, 3], [3, 4, 5, 6]])
    repeated, repeated_manifest = build_fixture([[1, 2, 3], [3, 4, 5, 6]])
    changed_provenance, changed_provenance_manifest = build_fixture(
        [[1, 2, 3], [3, 4, 5, 6]],
        provenance_sha256="b" * 64,
    )
    different, _ = build_fixture([[1, 2, 3], [9, 9, 9, 9]])

    torch.testing.assert_close(first, repeated, rtol=0, atol=0)
    torch.testing.assert_close(first, changed_provenance, rtol=0, atol=0)
    assert first_manifest["content_sha256"] == repeated_manifest["content_sha256"]
    assert (
        first_manifest["content_sha256"]
        != changed_provenance_manifest["content_sha256"]
    )
    assert not torch.equal(first, different)
    assert first_manifest["provenance_is_model_input"] is False
    assert all(
        "sha" not in name.lower()
        and "provenance" not in name.lower()
        and "revision" not in name.lower()
        for name in first_manifest["semantic_tensor_order"]
    )


def test_supervised_fraction_is_recorded_from_content() -> None:
    builder = CorpusFingerprintBuilder(vocab_size=8)
    builder.update_tokens([1, 2, 3, 4], supervised_mask=[False, True, True, False])
    result = builder.finalize()
    scalars = result.tensors["corpus.scalar_statistics"]
    assert float(scalars[6].item()) == 0.5


def test_gradient_probe_is_role_conditioned_and_changes_with_gradients() -> None:
    config = SemanticFingerprintConfig(gradient_sketch_dim_per_role=8, seed=23)
    roles = {"a": "attention_q", "b": "mlp_up"}
    first, manifest = gradient_probe_fingerprint(
        {
            "a": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "b": torch.linspace(-1, 1, 10),
        },
        roles,
        config=config,
    )
    repeated, _ = gradient_probe_fingerprint(
        {
            "a": torch.arange(12, dtype=torch.float32).reshape(3, 4),
            "b": torch.linspace(-1, 1, 10),
        },
        roles,
        config=config,
    )
    changed, _ = gradient_probe_fingerprint(
        {
            "a": torch.arange(12, dtype=torch.float32).reshape(3, 4).roll(1),
            "b": torch.linspace(-1, 1, 10),
        },
        roles,
        config=config,
    )

    assert manifest["contains_endpoint_data"] is False
    assert set(manifest["roles"]) == {"attention_q", "mlp_up"}
    for name in first:
        torch.testing.assert_close(first[name], repeated[name], rtol=0, atol=0)
    assert not torch.equal(
        first["w0_gradient.attention_q.countsketch"],
        changed["w0_gradient.attention_q.countsketch"],
    )


def test_activation_probe_summarizes_actual_values() -> None:
    tensors, manifest = activation_probe_fingerprint(
        {
            "residual.0": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
            "attention.0": torch.tensor([-2.0, 0.0, 2.0]),
        }
    )
    assert manifest["contains_endpoint_data"] is False
    assert tensors["w0_activation.residual.0.summary"].shape == (9,)
    assert torch.isfinite(torch.cat(list(tensors.values()))).all()
