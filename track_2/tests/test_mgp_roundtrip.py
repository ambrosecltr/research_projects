from __future__ import annotations

import json
import shutil

import pytest
import torch

from genome.bit_accounting import account_mgp
from genome.codecs import DenseDeltaCodec
from genome.mgp.interpreter import decode_program
from genome.mgp.serializer import load_program, save_program
from genome.tensor_inventory import assert_tied_equal


def contract(specimen):
    return {
        "architecture_manifest_sha256": specimen.manifest["contract_hashes"]["architecture"],
        "tensor_inventory_sha256": specimen.manifest["contract_hashes"]["tensor_inventory"],
        "base_state_sha256": specimen.manifest["state_hashes"]["W0"],
    }


def metadata(specimen):
    return {**contract(specimen), "research_level": "G0"}


def test_dense_mgp_roundtrip_is_exact(tiny_artifacts):
    specimen = tiny_artifacts["specimen"]
    base = specimen.load_base()
    target = specimen.load_target()
    program = DenseDeltaCodec(candidate_id="test_dense").fit(
        base,
        target,
        specimen.inventory,
        tied_groups=specimen.tied_groups,
        manifest_metadata=metadata(specimen),
    )
    path = tiny_artifacts["root"] / "test_dense.mgp"
    save_program(program, path)
    del program
    loaded = load_program(path)
    first = decode_program(
        loaded,
        base,
        specimen.inventory,
        tied_groups=specimen.tied_groups,
        contract=contract(specimen),
    )
    second = decode_program(
        loaded,
        base,
        specimen.inventory,
        tied_groups=specimen.tied_groups,
        contract=contract(specimen),
    )
    for name in target:
        assert torch.equal(first[name], target[name]), name
        assert torch.equal(first[name], second[name]), name
    assert_tied_equal(first, specimen.tied_groups)


def test_payload_corruption_is_detected(tiny_artifacts):
    specimen = tiny_artifacts["specimen"]
    path = tiny_artifacts["root"] / "corrupt_source.mgp"
    program = DenseDeltaCodec(candidate_id="corrupt_source").fit(
        specimen.load_base(),
        specimen.load_target(),
        specimen.inventory,
        tied_groups=specimen.tied_groups,
        manifest_metadata=metadata(specimen),
    )
    save_program(program, path)
    corrupt = tiny_artifacts["root"] / "corrupt_copy.mgp"
    shutil.copytree(path, corrupt)
    payload = corrupt / "genome.safetensors"
    data = bytearray(payload.read_bytes())
    data[-1] ^= 0x01
    payload.write_bytes(data)
    with pytest.raises(ValueError, match="hash mismatch"):
        load_program(corrupt)


def test_mgp_rejects_path_traversal(tiny_artifacts):
    specimen = tiny_artifacts["specimen"]
    source = tiny_artifacts["root"] / "path_source.mgp"
    program = DenseDeltaCodec(candidate_id="path_source").fit(
        specimen.load_base(),
        specimen.load_target(),
        specimen.inventory,
        tied_groups=specimen.tied_groups,
        manifest_metadata=metadata(specimen),
    )
    save_program(program, source)
    escaped = tiny_artifacts["root"] / "path_escape.mgp"
    shutil.copytree(source, escaped)
    manifest_path = escaped / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["payload_file"] = "../genome.safetensors"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="payload_file"):
        load_program(escaped)
    with pytest.raises(ValueError, match="payload_file"):
        account_mgp(escaped)


def test_mgp_rejects_unsupported_version(tiny_artifacts):
    specimen = tiny_artifacts["specimen"]
    source = tiny_artifacts["root"] / "version_source.mgp"
    program = DenseDeltaCodec(candidate_id="version_source").fit(
        specimen.load_base(),
        specimen.load_target(),
        specimen.inventory,
        tied_groups=specimen.tied_groups,
        manifest_metadata=metadata(specimen),
    )
    save_program(program, source)
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["version"] = "0.99.0"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="unsupported MGP version"):
        load_program(source)
