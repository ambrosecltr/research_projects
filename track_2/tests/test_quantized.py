from __future__ import annotations

import torch

from genome.codecs import QuantizedDeltaCodec
from genome.codecs.quantized import pack_int4
from genome.metrics import parameter_metrics
from genome.mgp.interpreter import decode_program, unpack_int4
from genome.mgp.serializer import load_program, save_program


def test_int4_pack_roundtrip():
    values = torch.tensor([-7, -3, 0, 1, 7, 2, -1], dtype=torch.int8)
    packed = pack_int4(values)
    decoded = unpack_int4(packed, values.numel())
    assert torch.equal(decoded, values)
    assert packed.numel() == (values.numel() + 1) // 2


def test_int8_and_int4_candidates_are_valid(tiny_artifacts):
    specimen = tiny_artifacts["specimen"]
    base = specimen.load_base()
    target = specimen.load_target()
    errors = {}
    sizes = {}
    for bits in (8, 4):
        program = QuantizedDeltaCodec(bits, candidate_id=f"q{bits}").fit(
            base,
            target,
            specimen.inventory,
            tied_groups=specimen.tied_groups,
        )
        path = tiny_artifacts["root"] / f"q{bits}.mgp"
        info = save_program(program, path)
        decoded = decode_program(
            load_program(path),
            base,
            specimen.inventory,
            tied_groups=specimen.tied_groups,
        )
        assert all(torch.isfinite(value).all() for value in decoded.values())
        errors[bits] = parameter_metrics(decoded, target, specimen.inventory)["relative_l2"]
        sizes[bits] = info["mgp_bytes"]
    assert errors[8] < errors[4]
    assert sizes[4] < sizes[8]
