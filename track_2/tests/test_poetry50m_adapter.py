from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch
import yaml

from genome.adapters.poetry50m import Poetry50MAdapter
from genome.evaluator import evaluate_model_state
from genome.hashing import sha256_file
from genome.specimen import freeze_specimen
from genome.tensor_inventory import infer_role

pytestmark = pytest.mark.track1_evaluation


def _write_fake_track1(root: Path) -> None:
    package = root / "src/poetry50m"
    (package / "model").mkdir(parents=True)
    (package / "training").mkdir(parents=True)
    (package / "data").mkdir(parents=True)
    (root / "src/tokenizers.py").write_text(
        '''import json\nclass Tokenizer:\n    def __init__(self, value): self.value=value\n    @classmethod\n    def from_file(cls, path):\n        with open(path, encoding="utf-8") as handle: return cls(json.load(handle))\n    def token_to_id(self, token): return self.value.get("vocab", {}).get(token)\n    def get_vocab_size(self, with_added_tokens=True): return len(self.value.get("vocab", {}))\n''',
        encoding="utf-8",
    )
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "model/__init__.py").write_text(
        '''from dataclasses import dataclass\nimport torch\nfrom torch import nn\n\n@dataclass(frozen=True)\nclass ModelConfig:\n    architecture: str\n    vocab_size: int\n    max_seq_len: int\n    d_model: int\n    n_layers: int\n    n_heads: int\n    ffn_dim: int\n    @classmethod\n    def from_mapping(cls, value): return cls(**value)\n\nclass Output:\n    def __init__(self, logits, loss=None, token_count=0):\n        self.logits, self.loss, self.token_count = logits, loss, token_count\n\nclass DecoderOnlyTransformer(nn.Module):\n    def __init__(self, config):\n        super().__init__(); self.config=config\n        self.token_embedding=nn.Embedding(config.vocab_size, config.d_model)\n        self.output_projection=nn.Linear(config.d_model, config.vocab_size, bias=False)\n    def forward(self, input_ids, targets=None, loss_mask=None):\n        logits=self.output_projection(self.token_embedding(input_ids))\n        if targets is None: return Output(logits)\n        mask=torch.ones_like(targets, dtype=torch.bool) if loss_mask is None else loss_mask.bool()\n        losses=torch.nn.functional.cross_entropy(logits.flatten(0,1), targets.flatten(), reduction='none').view_as(targets)\n        count=int(mask.sum().item()); loss=(losses*mask).sum()/count\n        return Output(logits, loss, count)\n''',
        encoding="utf-8",
    )
    (package / "training/__init__.py").write_text("", encoding="utf-8")
    (package / "training/engine.py").write_text(
        '''import hashlib\nimport json\nimport random\nimport torch\ndef seed_everything(seed, deterministic):\n    random.seed(seed); torch.manual_seed(seed); torch.use_deterministic_algorithms(deterministic)\ndef mapping_hash(values):\n    encoded=json.dumps(dict(values), sort_keys=True, separators=(",", ":"), default=str).encode()\n    return hashlib.sha256(encoded).hexdigest()\n''',
        encoding="utf-8",
    )
    (package / "data/__init__.py").write_text("", encoding="utf-8")
    (package / "data/tokenizer.py").write_text(
        'SPECIAL_TOKENS=("<|pad|>", "<|bos|>", "<|eos|>")\n',
        encoding="utf-8",
    )
    (package / "data/artifacts.py").write_text(
        '''import json\nfrom types import SimpleNamespace\ndef read_packed_sequences(path):\n    rows=[]\n    with open(path, encoding="utf-8") as handle:\n        for line in handle:\n            if line.strip(): rows.append(SimpleNamespace(**json.loads(line)))\n    return tuple(rows)\n''',
        encoding="utf-8",
    )
    (root / "configs/model").mkdir(parents=True)
    (root / "configs/training").mkdir(parents=True)
    (root / "configs/model/track1_50m.yaml").write_text(
        yaml.safe_dump(
            {
                "architecture": "gpt",
                "vocab_size": 16,
                "max_seq_len": 8,
                "d_model": 4,
                "n_layers": 1,
                "n_heads": 1,
                "ffn_dim": 8,
            }
        ),
        encoding="utf-8",
    )
    (root / "configs/training/baseline.yaml").write_text(
        yaml.safe_dump({"seed": 17, "deterministic": True, "max_steps": 10}),
        encoding="utf-8",
    )


def _clear_fake_modules() -> None:
    for name in tuple(sys.modules):
        if name == "tokenizers" or name == "poetry50m" or name.startswith("poetry50m."):
            del sys.modules[name]


def test_poetry50m_compact_anchor_loss_matches_model_loss(tmp_path: Path) -> None:
    root = tmp_path / "track_1"
    _write_fake_track1(root)
    _clear_fake_modules()
    adapter = Poetry50MAdapter(track1_root=root, require_complete_endpoint=False)
    model = adapter.build_model().eval()
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
        "targets": torch.tensor([[2, 3, 4, 5]], dtype=torch.long),
        "loss_mask": torch.tensor([[False, True, True, True]]),
    }
    with torch.inference_mode():
        expected_loss, expected_count = adapter.batch_loss(model, batch)
        actual_loss, actual_count, anchors = adapter.evaluate_batch(
            model, batch, capture_anchors=True, anchor_positions=2
        )
    torch.testing.assert_close(actual_loss, expected_loss)
    assert actual_count == expected_count == 3
    assert anchors is not None and anchors.shape == (2, 16)
    _clear_fake_modules()


def test_poetry50m_role_inference_handles_exact_track1_names() -> None:
    matrix = torch.zeros(4, 4)
    vector = torch.zeros(4)
    assert infer_role("blocks.0.attention.output.weight", matrix) == "o_proj"
    assert infer_role("blocks.0.mlp.in_projection.weight", matrix) == "gate_up_proj"
    assert infer_role("blocks.0.mlp.out_projection.weight", matrix) == "down_proj"
    assert infer_role("blocks.0.attention.qkv.weight", matrix) == "qkv_proj"
    assert infer_role("blocks.0.attention.qk_scale", vector) == "attention_scale"
    assert infer_role("blocks.0.mlp.uv_scale", vector) == "mlp_scale"
    assert infer_role("blocks.0.attention_rate", vector) == "residual_rate"
    assert infer_role("output_projection.weight", matrix) == "lm_head"
    assert infer_role("logit_scale", vector) == "logit_scale"


def test_poetry50m_adapter_replays_w0_and_rejects_partial_endpoint(tmp_path: Path) -> None:
    root = tmp_path / "track_1"
    _write_fake_track1(root)
    _clear_fake_modules()
    adapter = Poetry50MAdapter(
        track1_root=root,
        require_complete_endpoint=True,
    )
    first = adapter.initial_state()
    second = adapter.initial_state()
    assert all(torch.equal(first[name], second[name]) for name in first)

    adapter.run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    adapter.run_manifest_path.write_text(
        json.dumps({"run_id": "fake-r0"}),
        encoding="utf-8",
    )
    initial = tmp_path / "initial.pt"
    torch.save(
        {
            "format": "poetry50m.weights.v1",
            "metadata": {
                "run_id": "fake-r0",
                "step": 0,
                "model_config_hash": adapter.expected_model_config_hash,
                "training_config_hash": adapter.expected_train_config_hash,
            },
            "state_dict": first,
        },
        initial,
    )

    checkpoint = tmp_path / "partial.pt"
    torch.save(
        {
            "format_version": 2,
            "model": first,
            "training_state": {"global_step": 9},
            "model_config": adapter.model_config_mapping,
            "train_config": adapter.train_config_mapping,
            "run_metadata": {"run_id": "fake-r0"},
        },
        checkpoint,
    )
    summary = adapter.checkpoint_summary(checkpoint)
    assert summary["global_step"] == 9
    assert summary["max_steps"] == 10
    assert not summary["complete"]
    with pytest.raises(ValueError, match="not a complete endpoint"):
        adapter.validate_endpoint_checkpoint(checkpoint)

    complete = tmp_path / "complete.pt"
    torch.save(
        {
            "format_version": 2,
            "model": first,
            "training_state": {"global_step": 10},
            "model_config": adapter.model_config_mapping,
            "train_config": adapter.train_config_mapping,
            "run_metadata": {"run_id": "fake-r0"},
        },
        complete,
    )
    adapter.final_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "poetry50m.weights.v1",
            "metadata": {
                "run_id": "fake-r0",
                "step": 10,
                "model_config_hash": adapter.expected_model_config_hash,
                "training_config_hash": adapter.expected_train_config_hash,
            },
            "state_dict": first,
        },
        adapter.final_snapshot_path,
    )
    adapter.train_receipt_path.write_text(
        json.dumps(
            {
                "run_id": "fake-r0",
                "global_step": 10,
                "checkpoint_sha256": sha256_file(complete),
                "snapshot_sha256": sha256_file(adapter.final_snapshot_path),
                "run_manifest_sha256": sha256_file(adapter.run_manifest_path),
            }
        ),
        encoding="utf-8",
    )
    assert adapter.validate_endpoint_checkpoint(complete)["complete"]
    base_summary = adapter.validate_base_checkpoint(initial, endpoint_checkpoint=complete)
    assert base_summary["valid_base"] is True
    assert base_summary["run_manifest_matches"] is True
    model = adapter.build_model()
    adapter.load_checkpoint(model, complete)
    assert all(torch.equal(model.state_dict()[name], first[name]) for name in first)
    candidate = {name: tensor.clone() for name, tensor in first.items()}
    first_name = next(iter(candidate))
    candidate[first_name].view(-1)[0] += 0.01
    candidate = dict(reversed(tuple(candidate.items())))
    export = adapter.export_evaluation_checkpoint(
        candidate,
        template_checkpoint=complete,
        output=tmp_path / "candidate-eval.pt",
        candidate_id="candidate-test",
        provenance={"test": True},
    )
    exported = torch.load(export["path"], map_location="cpu", weights_only=True)
    assert exported["genome_evaluation"]["evaluation_only"] is True
    assert exported["genome_evaluation"]["resume_forbidden"] is True
    assert all(torch.equal(exported["model"][name], candidate[name]) for name in candidate)
    receipt = json.loads(adapter.train_receipt_path.read_text(encoding="utf-8"))
    receipt["checkpoint_sha256"] = "0" * 64
    adapter.train_receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(ValueError, match="completion artifacts failed validation"):
        adapter.validate_endpoint_checkpoint(complete)
    _clear_fake_modules()


def test_poetry50m_adapter_accepts_verified_sft_parent_checkpoint(tmp_path: Path) -> None:
    root = tmp_path / "track_1"
    _write_fake_track1(root)
    _clear_fake_modules()
    sft_config_path = root / "configs/training/sft.yaml"
    sft_config_path.write_text(
        yaml.safe_dump(
            {
                "seed": 17,
                "deterministic": True,
                "max_steps": 20,
                "learning_rate": 0.001,
            }
        ),
        encoding="utf-8",
    )
    adapter = Poetry50MAdapter(
        track1_root=root,
        train_config=sft_config_path,
        require_complete_endpoint=True,
    )
    state = adapter.initial_state()
    base_run_id = "fake-pretrain"
    sft_run_id = "fake-sft"
    base = tmp_path / "pretrain-final.pt"
    torch.save(
        {
            "format_version": 2,
            "model": state,
            "training_state": {"global_step": 10},
            "model_config": adapter.model_config_mapping,
            "train_config": {
                "seed": 17,
                "deterministic": True,
                "max_steps": 10,
            },
            "run_metadata": {"run_id": base_run_id},
        },
        base,
    )
    adapter.run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    adapter.run_manifest_path.write_text(
        json.dumps({"run_id": sft_run_id}),
        encoding="utf-8",
    )
    endpoint = tmp_path / "sft-final.pt"
    torch.save(
        {
            "format_version": 2,
            "model": state,
            "training_state": {"global_step": 20},
            "model_config": adapter.model_config_mapping,
            "train_config": adapter.train_config_mapping,
            "run_metadata": {
                "run_id": sft_run_id,
                "mode": "supervised_fine_tuning",
                "base_run_id": base_run_id,
                "base_checkpoint_sha256": sha256_file(base),
            },
        },
        endpoint,
    )
    adapter.train_receipt_path.write_text(
        json.dumps(
            {
                "run_id": sft_run_id,
                "global_step": 20,
                "checkpoint_sha256": sha256_file(endpoint),
                "mode": "supervised_fine_tuning",
            }
        ),
        encoding="utf-8",
    )

    endpoint_summary = adapter.validate_endpoint_checkpoint(endpoint)
    assert endpoint_summary["completion_artifacts"]["valid"] is True
    assert endpoint_summary["completion_artifacts"]["final_snapshot"] is None
    base_summary = adapter.validate_base_checkpoint(
        base,
        endpoint_checkpoint=endpoint,
    )
    assert base_summary["valid_base"] is True
    assert base_summary["mode"] == "supervised_fine_tuning_parent"
    _clear_fake_modules()


def test_poetry50m_adapter_freezes_and_evaluates_complete_fake_lineage(
    tmp_path: Path,
) -> None:
    root = tmp_path / "track_1"
    _write_fake_track1(root)
    _clear_fake_modules()
    adapter = Poetry50MAdapter(track1_root=root, require_complete_endpoint=True)

    adapter.prepared_dir.mkdir(parents=True)
    vocab = {"<|pad|>": 0, "<|bos|>": 1, "<|eos|>": 2}
    vocab.update({f"token-{index}": index for index in range(3, 16)})
    (adapter.prepared_dir / "tokenizer.json").write_text(
        json.dumps({"vocab": vocab}),
        encoding="utf-8",
    )
    (adapter.prepared_dir / "metadata.json").write_text(
        json.dumps(
            {
                "tokenizer_hash": "fake-tokenizer",
                "config": {
                    "objective_mix": {
                        "conditional_poetry": 1.0,
                        "auxiliary_prose_ntp": 0.0,
                    }
                },
                "split_counts": {"train": 1, "validation": 1, "test": 1},
            }
        ),
        encoding="utf-8",
    )
    pack = {
        "objective": "conditional_poetry",
        "pack_id": "fake-pack",
        "input_ids": [1, 3, 4, 5, 2],
        "loss_mask": [False, True, True, True, True],
    }
    for split in ("train", "validation", "test"):
        (adapter.prepared_dir / f"{split}.packed.jsonl").write_text(
            json.dumps(pack) + "\n",
            encoding="utf-8",
        )
    adapter.corpus_manifest_path.parent.mkdir(parents=True)
    adapter.corpus_manifest_path.write_text('{"document_id":"fake"}\n', encoding="utf-8")

    base = adapter.initial_state()
    target = {name: tensor.clone() for name, tensor in base.items()}
    target["output_projection.weight"].view(-1)[0] += 0.02
    adapter.run_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    adapter.run_manifest_path.write_text(json.dumps({"run_id": "fake-r0"}), encoding="utf-8")
    adapter.initial_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "poetry50m.weights.v1",
            "metadata": {
                "run_id": "fake-r0",
                "step": 0,
                "model_config_hash": adapter.expected_model_config_hash,
                "training_config_hash": adapter.expected_train_config_hash,
            },
            "state_dict": base,
        },
        adapter.initial_snapshot_path,
    )
    endpoint = adapter.run_dir / "checkpoints/final.pt"
    endpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": 2,
            "model": target,
            "training_state": {"global_step": 10},
            "model_config": adapter.model_config_mapping,
            "train_config": adapter.train_config_mapping,
            "run_metadata": {"run_id": "fake-r0"},
        },
        endpoint,
    )
    adapter.final_snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "poetry50m.weights.v1",
            "metadata": {
                "run_id": "fake-r0",
                "step": 10,
                "model_config_hash": adapter.expected_model_config_hash,
                "training_config_hash": adapter.expected_train_config_hash,
            },
            "state_dict": target,
        },
        adapter.final_snapshot_path,
    )
    adapter.train_receipt_path.write_text(
        json.dumps(
            {
                "run_id": "fake-r0",
                "global_step": 10,
                "checkpoint_sha256": sha256_file(endpoint),
                "snapshot_sha256": sha256_file(adapter.final_snapshot_path),
                "run_manifest_sha256": sha256_file(adapter.run_manifest_path),
            }
        ),
        encoding="utf-8",
    )

    specimen = freeze_specimen(
        adapter,
        output_dir=tmp_path / "specimen",
        specimen_id="fake-r0",
        final_checkpoint=endpoint,
        base_checkpoint=adapter.initial_snapshot_path,
    )
    assert specimen.manifest["base_validation"]["valid_base"] is True
    assert specimen.manifest["endpoint_validation"]["completion_artifacts"]["valid"] is True
    metrics = evaluate_model_state(
        adapter,
        specimen.load_target(),
        split="validation",
        max_batches=1,
        capture_logits=True,
    )
    assert metrics["batches"] == 1
    assert metrics["items"] == 4
    assert torch.isfinite(torch.tensor(metrics["mean_loss"]))
    _clear_fake_modules()
