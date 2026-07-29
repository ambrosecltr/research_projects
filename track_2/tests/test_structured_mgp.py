from __future__ import annotations

import torch

from genome.codecs.common import make_manifest
from genome.mgp.interpreter import decode_program
from genome.mgp.opcodes import CODEBOOK_BLOCKS, KRONECKER, SHARED_BASIS, SPECTRAL_DCT
from genome.types import GenomeComponent, GenomeProgram, TensorGenomeRecord, TensorSpec


def spec(shape: tuple[int, ...] = (4, 4)) -> TensorSpec:
    numel = int(torch.tensor(shape).prod().item())
    return TensorSpec(
        canonical_index=0,
        name="weight",
        role="test_matrix",
        layer_index=0,
        shape=shape,
        dtype="float32",
        numel=numel,
        nbytes=numel * 4,
    )


def program(component: GenomeComponent, payload: dict[str, torch.Tensor]) -> GenomeProgram:
    return GenomeProgram(
        manifest=make_manifest(candidate_id="structured-test", codec="test"),
        records=[
            TensorGenomeRecord(
                tensor_name="weight",
                canonical_index=0,
                role="test_matrix",
                layer_index=0,
                shape=(4, 4),
                output_dtype="float32",
                components=[component],
            )
        ],
        payload_tensors=payload,
    )


def decode(candidate: GenomeProgram) -> torch.Tensor:
    return decode_program(
        candidate,
        {"weight": torch.zeros(4, 4)},
        [spec()],
    )["weight"]


def test_kronecker_primitive() -> None:
    left = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    right = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
    candidate = program(
        GenomeComponent(
            KRONECKER,
            payload_keys=["left", "right", "coefficient"],
            arguments={"terms": 1, "matrix_shape": [4, 4]},
        ),
        {
            "left": left,
            "right": right,
            "coefficient": torch.tensor(0.5),
        },
    )
    torch.testing.assert_close(decode(candidate), 0.5 * torch.kron(left, right))


def test_spectral_dct_dc_mode_produces_constant_matrix() -> None:
    candidate = program(
        GenomeComponent(
            SPECTRAL_DCT,
            payload_keys=["frequencies", "coefficients"],
            arguments={"matrix_shape": [4, 4]},
        ),
        {
            "frequencies": torch.tensor([[0, 0]], dtype=torch.int16),
            "coefficients": torch.tensor([4.0]),
        },
    )
    torch.testing.assert_close(decode(candidate), torch.ones(4, 4), rtol=1e-6, atol=1e-6)


def test_shared_basis_primitive() -> None:
    basis = torch.stack(
        [
            torch.eye(4).flatten(),
            torch.ones(4, 4).triu().flatten(),
        ]
    )
    coefficients = torch.tensor([2.0, -0.5])
    candidate = program(
        GenomeComponent(
            SHARED_BASIS,
            payload_keys=["basis", "coefficients"],
        ),
        {"basis": basis, "coefficients": coefficients},
    )
    expected = (coefficients @ basis).reshape(4, 4)
    torch.testing.assert_close(decode(candidate), expected)


def test_codebook_blocks_primitive() -> None:
    codebook = torch.stack([torch.ones(2, 2), torch.full((2, 2), 2.0)])
    indices = torch.tensor([0, 1, 1, 0], dtype=torch.int16)
    scales = torch.tensor([1.0, 0.5, 1.5, 2.0])
    candidate = program(
        GenomeComponent(
            CODEBOOK_BLOCKS,
            payload_keys=["codebook", "indices", "scales"],
            arguments={"matrix_shape": [4, 4], "block_rows": 2, "block_cols": 2},
        ),
        {"codebook": codebook, "indices": indices, "scales": scales},
    )
    expected = torch.tensor(
        [
            [1.0, 1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
            [3.0, 3.0, 2.0, 2.0],
            [3.0, 3.0, 2.0, 2.0],
        ]
    )
    torch.testing.assert_close(decode(candidate), expected)
