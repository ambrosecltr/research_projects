from __future__ import annotations

from genome.bit_accounting import account_mgp
from genome.codecs import DenseDeltaCodec
from genome.mgp.serializer import save_program


def test_accounting_uses_only_decode_contract_files(tiny_artifacts):
    specimen = tiny_artifacts["specimen"]
    path = tiny_artifacts["root"] / "accounting.mgp"
    program = DenseDeltaCodec(candidate_id="accounting").fit(
        specimen.load_base(),
        specimen.load_target(),
        specimen.inventory,
        tied_groups=specimen.tied_groups,
    )
    save_program(program, path)
    (path / "unrelated_evaluation.json").write_text("x" * 5000, encoding="utf-8")
    result = account_mgp(path, base_path=specimen.base_path)
    expected = sum(
        (path / name).stat().st_size for name in ["manifest.json", "genome.safetensors"]
    )
    assert result["mgp_bytes"] == expected
    assert result["artifact_directory_bytes"] > result["mgp_bytes"]
