from __future__ import annotations

import json
import shutil

import pytest
import torch

from genome.hashing import sha256_json
from genome.metrics import parameter_metrics
from genome.mgp.interpreter import decode_program
from genome.mgp.serializer import load_program, save_program
from genome.neural import (
    AutodecoderTrainingConfig,
    BlockDecoderConfig,
    fit_autodecoder,
    load_interpreter,
)
from genome.types import TensorSpec


def test_neural_block_field_can_overfit_one_tensor(tmp_path):
    torch.manual_seed(4)
    base = {"matrix.weight": torch.zeros(8, 8)}
    target = {"matrix.weight": torch.randn(8, 8) * 0.2}
    spec = TensorSpec(
        canonical_index=0,
        name="matrix.weight",
        role="other",
        layer_index=0,
        shape=(8, 8),
        dtype="float32",
        numel=64,
        nbytes=256,
    )
    interpreter_path = tmp_path / "interpreter"
    result = fit_autodecoder(
        base,
        target,
        [spec],
        interpreter_path=interpreter_path,
        decoder_config=BlockDecoderConfig(
            block_rows=4,
            block_cols=4,
            global_code_dim=16,
            layer_code_dim=8,
            tensor_code_dim=8,
            role_embedding_dim=8,
            hidden_dim=64,
            depth=2,
        ),
        training_config=AutodecoderTrainingConfig(
            updates=350,
            batch_size=4,
            learning_rate=0.003,
            log_every=100,
        ),
    )
    mgp_path = tmp_path / "candidate.mgp"
    save_program(result.program, mgp_path)
    decoded = decode_program(
        load_program(mgp_path),
        base,
        [spec],
        interpreter=load_interpreter(interpreter_path),
        verify_checksums=False,
    )
    metrics = parameter_metrics(decoded, target, [spec])
    assert metrics["relative_l2"] < 0.08

    escaped = tmp_path / "interpreter_escape"
    shutil.copytree(interpreter_path, escaped)
    manifest_path = escaped / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["decoder_file"] = "../decoder.safetensors"
    content = dict(manifest)
    content.pop("manifest_content_sha256", None)
    manifest["manifest_content_sha256"] = sha256_json(content)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="decoder_file"):
        load_interpreter(escaped)
