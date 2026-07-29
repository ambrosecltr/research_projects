from __future__ import annotations

import json
import shutil

import pytest
import torch
from safetensors.torch import save_file

from genome.hashing import sha256_json
from genome.neural.compiler import GenomeCodeLayout
from genome.neural.compiler_training import (
    CompilerTrainingConfig,
    load_compiler,
    train_compiler,
)

pytestmark = pytest.mark.legacy


def test_compiler_training_artifact_roundtrip(tmp_path):
    layout = GenomeCodeLayout(
        global_code_dim=2,
        n_layers=2,
        layer_code_dim=2,
        n_tensors=3,
        tensor_code_dim=2,
    )
    paths = []
    torch.manual_seed(9)
    for index in range(4):
        path = tmp_path / f"life_{index}.safetensors"
        save_file(
            {
                "architecture_features": torch.randn(5),
                "dataset_fingerprint": torch.randn(7),
                "trajectory_fingerprint": torch.randn(6),
                "target_flat_codes": torch.randn(layout.total_dim),
            },
            str(path),
        )
        paths.append(path)
    output = tmp_path / "compiler"
    train_compiler(
        paths,
        layout=layout,
        output_path=output,
        config=CompilerTrainingConfig(
            epochs=2,
            batch_size=2,
            hidden_dim=16,
            depth=1,
            learning_rate=1e-3,
        ),
    )
    model = load_compiler(output)
    distribution = model(torch.randn(1, 5), torch.randn(1, 7), torch.randn(1, 6))
    assert distribution.mean.shape == (1, layout.total_dim)

    escaped = tmp_path / "compiler_escape"
    shutil.copytree(output, escaped)
    manifest_path = escaped / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["weights_file"] = "../compiler.safetensors"
    content = dict(manifest)
    content.pop("manifest_content_sha256", None)
    manifest["manifest_content_sha256"] = sha256_json(content)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="weights_file"):
        load_compiler(escaped)
