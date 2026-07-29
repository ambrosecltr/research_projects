from __future__ import annotations

import pytest
import torch

from genome.codecs.common import make_manifest
from genome.mgp.interpreter import decode_program
from genome.mgp.opcodes import DENSE_DELTA, LOW_RANK, QUANTIZED_DELTA
from genome.program_compiler import PROGRAM_TOKEN_TO_ID
from genome.program_tokens import (
    ProgramTokenizationConfig,
    collate_program_sequences,
    program_to_sequence,
    sequence_to_program,
    teacher_forcing_shift,
)
from genome.types import GenomeComponent, GenomeProgram, TensorGenomeRecord, TensorSpec


def fixture() -> tuple[GenomeProgram, list[TensorSpec]]:
    specification = TensorSpec(
        canonical_index=0,
        name="weight",
        role="mlp_up",
        layer_index=0,
        shape=(4, 4),
        dtype="float32",
        numel=16,
        nbytes=64,
    )
    u = torch.tensor([[1.0], [0.0], [-1.0], [0.5]], dtype=torch.float32)
    s = torch.tensor([2.0])
    vh = torch.tensor([[0.5, -1.0, 0.0, 1.0]], dtype=torch.float32)
    keys = ["u", "s", "vh"]
    program = GenomeProgram(
        manifest=make_manifest(candidate_id="token-test", codec="compact"),
        records=[
            TensorGenomeRecord(
                tensor_name="weight",
                canonical_index=0,
                role="mlp_up",
                layer_index=0,
                shape=(4, 4),
                output_dtype="float32",
                components=[
                    GenomeComponent(
                        LOW_RANK,
                        payload_keys=keys,
                        arguments={"rank": 1},
                    )
                ],
            )
        ],
        payload_tensors={"u": u, "s": s, "vh": vh},
    )
    return program, [specification]


def vector_fixture() -> tuple[GenomeProgram, list[TensorSpec]]:
    specification = TensorSpec(
        canonical_index=0,
        name="norm.weight",
        role="norm_scale",
        layer_index=0,
        shape=(4,),
        dtype="float32",
        numel=4,
        nbytes=16,
    )
    program = GenomeProgram(
        manifest=make_manifest(candidate_id="vector-test", codec="compact"),
        records=[
            TensorGenomeRecord(
                tensor_name="norm.weight",
                canonical_index=0,
                role="norm_scale",
                layer_index=0,
                shape=(4,),
                output_dtype="float32",
                components=[
                    GenomeComponent(
                        QUANTIZED_DELTA,
                        payload_keys=["values", "scale"],
                        arguments={"bits": 8, "shape": [4], "small_vector": True},
                    )
                ],
            )
        ],
        payload_tensors={
            "values": torch.tensor([1, -2, 3, 0], dtype=torch.int8),
            "scale": torch.tensor(0.25, dtype=torch.float32),
        },
    )
    return program, [specification]


def test_program_tokenization_round_trips_through_runtime() -> None:
    original, inventory = fixture()
    config = ProgramTokenizationConfig(coefficient_chunk_dim=4, factor_dtype="float16")
    sequence = program_to_sequence(original, inventory, config=config)
    restored = sequence_to_program(sequence, inventory, config=config)

    base = {"weight": torch.zeros(4, 4)}
    expected = decode_program(original, base, inventory)["weight"]
    actual = decode_program(restored, base, inventory)["weight"]
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)
    assert restored.manifest["created_unix"] == 0.0
    assert restored.manifest["contains_exact_residual"] is False


def test_quantized_vector_token_round_trips_through_runtime() -> None:
    original, inventory = vector_fixture()
    config = ProgramTokenizationConfig(coefficient_chunk_dim=4)
    sequence = program_to_sequence(original, inventory, config=config)
    assert PROGRAM_TOKEN_TO_ID["QUANTIZED_VECTOR"] in sequence.token_ids.tolist()
    restored = sequence_to_program(sequence, inventory, config=config)

    base = {"norm.weight": torch.ones(4)}
    expected = decode_program(original, base, inventory)["norm.weight"]
    actual = decode_program(restored, base, inventory)["norm.weight"]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert restored.payload_tensors["t00000.vector_q8.values"].dtype == torch.int8


def test_program_batch_collation_and_teacher_shift() -> None:
    original, inventory = fixture()
    sequence = program_to_sequence(
        original,
        inventory,
        config=ProgramTokenizationConfig(coefficient_chunk_dim=8),
    )
    batch = collate_program_sequences((sequence, sequence))
    decoder_tokens, decoder_numeric, targets = teacher_forcing_shift(batch)
    assert decoder_tokens.shape == targets.token_ids.shape
    assert decoder_numeric.shape == targets.numeric_values.shape
    assert batch.token_mask.all()


def test_tokenizer_rejects_dense_endpoint_payload() -> None:
    original, inventory = fixture()
    original.records[0].components = [GenomeComponent(DENSE_DELTA, payload_keys=["dense"])]
    original.payload_tensors = {"dense": torch.zeros(4, 4)}
    with pytest.raises(ValueError, match="only canonical LOW_RANK"):
        program_to_sequence(original, inventory)
