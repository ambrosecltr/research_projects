from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM, PreTrainedTokenizerFast

from genome.adapters.gpt_neox import (
    assert_native_canonical_roundtrip,
    canonicalize_gpt_neox_state,
    load_canonical_state_into_model,
    model_from_canonical_state,
    nativeize_gpt_neox_state,
)
from genome.hashing import sha256_file, sha256_json, sha256_state_dict
from genome.io import load_tensor_file, save_tensor_file, write_json
from genome.mgp.interpreter import _apply_component, decode_program
from genome.neural.autodecoder import AutodecoderTrainingConfig, fit_autodecoder
from genome.neural.block_decoder import (
    BlockDecoderConfig,
    RoleConditionedBlockDecoder,
    load_interpreter,
    make_block_features,
)
from genome.neural.multilife_decoder import (
    LatentCodeFitConfig,
    MultiLifeBlockSampler,
    SharedDecoderTrainingConfig,
    fit_genome_code_with_frozen_decoder,
    masked_block_mse,
    train_shared_decoder,
)
from genome.neural.predictive_compiler import (
    PredictiveCompilerTrainingConfig,
    predict_hidden_genome,
    train_predictive_compiler,
)
from genome.legacy.polypythia_v4.evaluate import (
    evaluate_shared_decoder_corpus,
    execute_hidden_prediction,
)
from genome.polypythia.catalog import load_round_one_catalog
from genome.polypythia.evidence import EvidenceConfig, build_compiler_evidence
from genome.polypythia.hub import (
    CheckpointSource,
    DatasetOrderPlan,
    HubFile,
    LifeSourcePlan,
    RoundOneSourcePlan,
    TokenizerSourcePlan,
    _resolve_lfs_weight,
    iter_materializable_checkpoints,
    load_source_plan,
    materialize_source_plan,
    save_source_plan,
)
from genome.polypythia.lives import CanonicalModelLife
from genome.tensor_inventory import build_tensor_inventory_from_state
from genome.types import GenomeComponent

pytestmark = pytest.mark.legacy


def _round_one_config() -> Path:
    return Path(__file__).parents[2] / "configs" / "polypythia_14m_round1.yaml"


def test_round_one_catalog_seals_ten_complete_lives():
    catalog = load_round_one_catalog(_round_one_config())
    assert len(catalog.checkpoints.steps) == 154
    assert catalog.checkpoints.steps[0] == 0
    assert catalog.checkpoints.steps[-1] == 143000
    assert [life.run_id for life in catalog.lives_for("training")] == [
        f"pythia-14m-seed{seed}" for seed in range(8)
    ]
    assert catalog.lives_for("development")[0].run_id == "pythia-14m-seed8"
    assert catalog.hidden_life.run_id == "pythia-14m-seed9"
    assert catalog.lives[0].repository == "EleutherAI/pythia-14m"
    assert all("deduped" not in life.repository for life in catalog.lives)


def test_gpt_neox_native_canonical_roundtrip_is_exact():
    state = {
        "gpt_neox.embed_in.weight": torch.randn(9, 4),
        "gpt_neox.layers.0.input_layernorm.weight": torch.randn(4),
        "gpt_neox.layers.0.input_layernorm.bias": torch.randn(4),
        "gpt_neox.layers.0.post_attention_layernorm.weight": torch.randn(4),
        "gpt_neox.layers.0.post_attention_layernorm.bias": torch.randn(4),
        "gpt_neox.layers.0.attention.query_key_value.weight": torch.randn(12, 4),
        "gpt_neox.layers.0.attention.query_key_value.bias": torch.randn(12),
        "gpt_neox.layers.0.attention.dense.weight": torch.randn(4, 4),
        "gpt_neox.layers.0.attention.dense.bias": torch.randn(4),
        "gpt_neox.layers.0.mlp.dense_h_to_4h.weight": torch.randn(16, 4),
        "gpt_neox.layers.0.mlp.dense_h_to_4h.bias": torch.randn(16),
        "gpt_neox.layers.0.mlp.dense_4h_to_h.weight": torch.randn(4, 16),
        "gpt_neox.layers.0.mlp.dense_4h_to_h.bias": torch.randn(4),
        "gpt_neox.final_layer_norm.weight": torch.randn(4),
        "gpt_neox.final_layer_norm.bias": torch.randn(4),
        "embed_out.weight": torch.randn(9, 4),
    }
    assert_native_canonical_roundtrip(state)
    canonical = canonicalize_gpt_neox_state(state)
    assert "layers.0.attention.qkv_proj.weight" in canonical
    assert "layers.0.mlp.up_proj.weight" in canonical
    assert set(nativeize_gpt_neox_state(canonical)) == set(state)
    inventory, _ = build_tensor_inventory_from_state(canonical)
    roles = {spec.name: spec.role for spec in inventory}
    assert roles["layers.0.attention.qkv_proj.weight"] == "qkv_proj"
    assert roles["layers.0.attention.qkv_proj.bias"] == "qkv_proj"


def test_gpt_neox_canonical_state_excludes_regenerated_attention_buffers():
    state = {
        "gpt_neox.embed_in.weight": torch.randn(9, 4),
        "gpt_neox.layers.0.attention.bias": torch.ones(
            1,
            1,
            16,
            16,
            dtype=torch.bool,
        ).tril(),
        "gpt_neox.layers.0.attention.masked_bias": torch.tensor(-65_504.0),
        "gpt_neox.layers.0.attention.rotary_emb.inv_freq": torch.randn(2),
        "gpt_neox.layers.0.attention.query_key_value.weight": torch.randn(12, 4),
        "gpt_neox.layers.0.attention.query_key_value.bias": torch.randn(12),
        "gpt_neox.layers.0.attention.dense.weight": torch.randn(4, 4),
        "gpt_neox.layers.0.attention.dense.bias": torch.randn(4),
        "gpt_neox.layers.0.input_layernorm.weight": torch.randn(4),
        "gpt_neox.layers.0.input_layernorm.bias": torch.randn(4),
        "gpt_neox.layers.0.post_attention_layernorm.weight": torch.randn(4),
        "gpt_neox.layers.0.post_attention_layernorm.bias": torch.randn(4),
        "gpt_neox.layers.0.mlp.dense_h_to_4h.weight": torch.randn(16, 4),
        "gpt_neox.layers.0.mlp.dense_h_to_4h.bias": torch.randn(16),
        "gpt_neox.layers.0.mlp.dense_4h_to_h.weight": torch.randn(4, 16),
        "gpt_neox.layers.0.mlp.dense_4h_to_h.bias": torch.randn(4),
        "gpt_neox.final_layer_norm.weight": torch.randn(4),
        "gpt_neox.final_layer_norm.bias": torch.randn(4),
        "embed_out.weight": torch.randn(9, 4),
    }
    assert_native_canonical_roundtrip(state)
    canonical = canonicalize_gpt_neox_state(state)
    assert all("rotary" not in name and "masked_bias" not in name for name in canonical)
    assert len(canonical) == len(state) - 3


def test_real_transformers_gpt_neox_roundtrip_preserves_logits():
    torch.manual_seed(9)
    config = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )
    source = GPTNeoXForCausalLM(config).eval()
    canonical = canonicalize_gpt_neox_state(source.state_dict())
    restored = GPTNeoXForCausalLM(config).eval()
    load_canonical_state_into_model(restored, canonical)
    tokens = torch.tensor([[1, 2, 3, 4]], dtype=torch.long)
    with torch.inference_mode():
        expected = source(tokens).logits
        actual = restored(tokens).logits
    assert torch.equal(actual, expected)


def test_neural_genome_decodes_vectors_without_dense_endpoint_payload(tmp_path):
    torch.manual_seed(17)
    base = {
        "layers.0.attention.qkv_proj.weight": torch.zeros(4, 4),
        "layers.0.attention.qkv_proj.bias": torch.zeros(4),
    }
    target = {name: tensor + torch.randn_like(tensor) * 0.1 for name, tensor in base.items()}
    inventory, ties = build_tensor_inventory_from_state(base)
    result = fit_autodecoder(
        base,
        target,
        inventory,
        tied_groups=ties,
        interpreter_path=tmp_path / "interpreter",
        decoder_config=BlockDecoderConfig(
            block_rows=4,
            block_cols=4,
            global_code_dim=16,
            layer_code_dim=8,
            tensor_code_dim=8,
            role_embedding_dim=8,
            feature_dim=23,
            hidden_dim=64,
            depth=2,
        ),
        training_config=AutodecoderTrainingConfig(
            seed=2,
            updates=500,
            batch_size=4,
            learning_rate=0.003,
            log_every=250,
        ),
    )
    assert all(
        component.opcode == "NEURAL_BLOCK_FIELD"
        for record in result.program.records
        for component in record.components
    )
    decoded = decode_program(
        result.program,
        base,
        inventory,
        interpreter=load_interpreter(tmp_path / "interpreter"),
        verify_checksums=False,
    )
    assert decoded["layers.0.attention.qkv_proj.bias"].shape == (4,)
    relative_error = torch.linalg.vector_norm(
        decoded["layers.0.attention.qkv_proj.bias"] - target["layers.0.attention.qkv_proj.bias"]
    ) / torch.linalg.vector_norm(target["layers.0.attention.qkv_proj.bias"])
    assert relative_error < 0.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_neural_runtime_returns_cuda_decoder_output_to_cpu_base():
    class CudaInterpreter:
        def decode_tensor_from_component(self, *args, **kwargs):
            return torch.ones(2, 2, device="cuda")

    decoded = _apply_component(
        torch.zeros(2, 2),
        GenomeComponent("NEURAL_BLOCK_FIELD"),
        None,
        record_name="test.weight",
        interpreter=CudaInterpreter(),
    )
    assert decoded.device.type == "cpu"
    assert torch.equal(decoded, torch.ones(2, 2))


def _fake_source_plan() -> RoundOneSourcePlan:
    catalog = load_round_one_catalog(_round_one_config())
    file = HubFile(name="pytorch_model.bin", size=4, sha256="a" * 64)
    lives = tuple(
        LifeSourcePlan(
            run_id=life.run_id,
            seed=life.seed,
            data_order_seed=life.data_order_seed,
            repository=life.repository,
            split=life.split,
            main_commit="b" * 40,
            checkpoints=tuple(
                CheckpointSource(
                    step=step,
                    branch=f"step{step}",
                    commit=f"{step:040x}"[-40:],
                    weight=file,
                )
                for step in catalog.checkpoints.steps
            ),
        )
        for life in catalog.lives
    )
    order_file = HubFile(name="order.npy", size=8, sha256="c" * 64)
    return RoundOneSourcePlan(
        catalog=catalog.to_dict(),
        lives=lives,
        dataset_order=DatasetOrderPlan(
            repository=catalog.dataset_repository,
            commit="d" * 40,
            seed_files={str(seed): (order_file, order_file, order_file) for seed in range(10)},
        ),
        tokenizer=TokenizerSourcePlan(
            repository=catalog.tokenizer_source,
            commit="e" * 40,
            files=(
                HubFile(
                    name="config.json",
                    size=2,
                    sha256=None,
                    git_blob_id="f" * 40,
                ),
            ),
        ),
        catalogued_checkpoint_bytes=4 * 154 * 10,
        sealed_materialization_bytes=4 * 19,
        revealed_materialization_bytes=4 * 20,
    )


def test_hidden_source_plan_exposes_only_step_zero_before_prediction(tmp_path):
    plan = _fake_source_plan()
    path = tmp_path / "plan.json"
    save_source_plan(plan, path)
    loaded = load_source_plan(path)
    hidden = [
        checkpoint
        for life, checkpoint in iter_materializable_checkpoints(
            loaded,
            splits=("hidden",),
            reveal_hidden=False,
        )
    ]
    assert [checkpoint.step for checkpoint in hidden] == [0]
    revealed = [
        checkpoint
        for life, checkpoint in iter_materializable_checkpoints(
            loaded,
            splits=("hidden",),
            reveal_hidden=True,
        )
    ]
    assert [checkpoint.step for checkpoint in revealed] == [0, 143000]


def test_hidden_reveal_requires_pre_reveal_runtime_execution(tmp_path):
    plan = _fake_source_plan()
    seal = {
        "format": "GENOME_HIDDEN_PREDICTION_SEAL",
        "version": "0.1.0",
        "hidden_run_id": "pythia-14m-seed9",
        "created_unix": 1.0,
        "prediction_manifest_sha256": "1" * 64,
        "predicted_mgp_sha256": "2" * 64,
        "compiler_manifest_sha256": "3" * 64,
        "shared_decoder_manifest_sha256": "4" * 64,
        "base_state_sha256": "5" * 64,
        "source_plan_content_sha256": plan.to_dict()["content_sha256"],
        "target_endpoint_seen": False,
    }
    seal["content_sha256"] = sha256_json(seal)
    seal_path = tmp_path / "prediction-seal.json"
    write_json(seal_path, seal, canonical=True)
    with pytest.raises(ValueError, match="runtime execution manifest"):
        materialize_source_plan(
            plan,
            output_root=tmp_path / "downloads",
            reveal_hidden=True,
            prediction_seal=seal_path,
        )


def test_lfs_resolver_reads_exact_weight_identity_without_downloading_body():
    commit = "a" * 40
    sha256 = "b" * 64

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "HEAD"
        return httpx.Response(
            302,
            headers={
                "x-repo-commit": commit,
                "x-linked-size": "53331592",
                "x-linked-etag": f'"{sha256}"',
                "location": "https://example.test/weight",
            },
        )

    with httpx.Client(
        transport=httpx.MockTransport(handler),
        follow_redirects=False,
    ) as client:
        weight = _resolve_lfs_weight(
            client,
            endpoint="https://huggingface.co",
            repository="EleutherAI/pythia-14m-seed1",
            commit=commit,
            filename="pytorch_model.bin",
        )
    assert weight == HubFile(
        name="pytorch_model.bin",
        size=53_331_592,
        sha256=sha256,
    )


def test_compiler_evidence_contains_w0_but_no_endpoint_data():
    base = {
        "token_embedding.weight": torch.randn(8, 4),
        "layers.0.attention.qkv_proj.weight": torch.randn(12, 4),
        "lm_head.weight": torch.randn(8, 4),
    }
    inventory, _ = build_tensor_inventory_from_state(base)
    order_file = HubFile(name="order.npy", size=8, sha256="a" * 64)
    tensors, manifest = build_compiler_evidence(
        base_state=base,
        inventory=inventory,
        architecture={
            "hidden_size": 4,
            "intermediate_size": 16,
            "num_hidden_layers": 1,
            "num_attention_heads": 1,
            "vocab_size": 8,
            "max_position_embeddings": 16,
            "rotary_pct": 0.25,
            "layer_norm_eps": 1e-5,
            "initializer_range": 0.02,
            "use_parallel_residual": True,
            "tie_word_embeddings": False,
        },
        dataset_order=DatasetOrderPlan(
            repository="EleutherAI/pile-preshuffled-seeds",
            commit="b" * 40,
            seed_files={"0": (order_file, order_file, order_file)},
        ),
        data_order_seed=0,
        tokenizer_identity={"sha256": "c" * 64},
        training_recipe={"steps": 143000},
        config=EvidenceConfig(
            initialization_sketch_dim_per_role=8,
            digest_vector_dim=8,
        ),
    )
    assert "initialization_fingerprint" in tensors
    assert manifest["forbidden_endpoint_inputs"][0] == "WT_values"
    serialized = json.dumps(manifest)
    assert "target_state" not in serialized
    assert "endpoint_hash" in serialized


def test_multi_life_sampler_uses_independent_life_indices():
    base = [
        {"layers.0.attention.qkv_proj.weight": torch.zeros(4, 4)},
        {"layers.0.attention.qkv_proj.weight": torch.ones(4, 4) * 0.1},
    ]
    target = [
        {"layers.0.attention.qkv_proj.weight": torch.ones(4, 4) * 0.2},
        {"layers.0.attention.qkv_proj.weight": torch.ones(4, 4) * 0.4},
    ]
    inventory, ties = build_tensor_inventory_from_state(base[0])
    sampler = MultiLifeBlockSampler(
        base_states=base,
        target_states=target,
        tensor_specs=inventory,
        tied_groups=ties,
        decoder_config=BlockDecoderConfig(
            block_rows=4,
            block_cols=4,
            feature_dim=23,
        ),
    )
    batch = sampler.make_batch(
        batch_size=64,
        generator=torch.Generator().manual_seed(5),
        device=torch.device("cpu"),
    )
    assert set(batch.life_indices.tolist()) == {0, 1}
    assert batch.features.shape == (64, 23)
    assert batch.valid_masks.shape == batch.targets.shape
    assert torch.equal(batch.valid_masks, torch.ones_like(batch.valid_masks))


def test_masked_block_mse_ignores_vector_padding():
    base = {"layers.0.attention.qkv_proj.bias": torch.zeros(4)}
    target = {"layers.0.attention.qkv_proj.bias": torch.ones(4)}
    inventory, ties = build_tensor_inventory_from_state(base)
    sampler = MultiLifeBlockSampler(
        base_states=[base],
        target_states=[target],
        tensor_specs=inventory,
        tied_groups=ties,
        decoder_config=BlockDecoderConfig(
            block_rows=4,
            block_cols=4,
            feature_dim=23,
        ),
    )
    batch = sampler.make_batch(
        batch_size=1,
        generator=torch.Generator().manual_seed(7),
        device=torch.device("cpu"),
    )
    assert batch.valid_masks.sum().item() == 4
    prediction = batch.targets.clone()
    prediction[batch.valid_masks == 0] = 100.0
    assert masked_block_mse(prediction, batch.targets, batch.valid_masks).item() == 0.0


def test_multi_life_sampler_uses_per_tensor_scales():
    base = {
        "layers.0.attention.qkv_proj.weight": torch.zeros(4, 4),
        "layers.0.attention.qkv_proj.bias": torch.zeros(4),
    }
    target = {
        "layers.0.attention.qkv_proj.weight": torch.full((4, 4), 0.1),
        "layers.0.attention.qkv_proj.bias": torch.full((4,), 10.0),
    }
    inventory, ties = build_tensor_inventory_from_state(base)
    sampler = MultiLifeBlockSampler(
        base_states=[base],
        target_states=[target],
        tensor_specs=inventory,
        tied_groups=ties,
        decoder_config=BlockDecoderConfig(
            block_rows=4,
            block_cols=4,
            feature_dim=23,
        ),
    )
    assert sampler.tensor_scales["layers.0.attention.qkv_proj.weight"] == pytest.approx(0.1)
    assert sampler.tensor_scales["layers.0.attention.qkv_proj.bias"] == pytest.approx(10.0)


def test_fourier_block_features_are_deterministic():
    config = BlockDecoderConfig(
        block_rows=4,
        block_cols=4,
        feature_dim=31,
    )
    metadata = torch.tensor([0.0, 0.25, 0.75, 0.1, 0.1, 0.0, 0.0])
    block = torch.arange(16, dtype=torch.float32).reshape(4, 4)
    first = make_block_features(metadata, block, config)
    second = make_block_features(metadata, block, config)
    assert first.shape == (31,)
    assert torch.equal(first, second)
    assert torch.equal(first[-16:], block.flatten())


def test_block_decoder_config_loads_pre_block_code_manifests():
    legacy = BlockDecoderConfig().to_dict()
    legacy.pop("block_code_dim")
    legacy.pop("block_code_mode")
    legacy.pop("block_code_storage_dtype")
    loaded = BlockDecoderConfig.from_dict(legacy)
    assert loaded.block_code_dim == 0
    assert loaded.block_code_mode == "network"
    assert loaded.block_code_storage_dtype == "float32"


def test_residual_block_codes_are_a_canonical_decoder_residual():
    config = BlockDecoderConfig(
        block_rows=2,
        block_cols=2,
        global_code_dim=2,
        layer_code_dim=2,
        tensor_code_dim=2,
        block_code_dim=4,
        block_code_mode="residual",
        block_code_storage_dtype="float16",
        role_embedding_dim=2,
        feature_dim=7,
        hidden_dim=4,
        depth=0,
    )
    decoder = RoleConditionedBlockDecoder(1, config)
    for parameter in decoder.parameters():
        torch.nn.init.zeros_(parameter)
    block_codes = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float16)

    output = decoder(
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, 2),
        torch.zeros(1, dtype=torch.long),
        torch.zeros(1, 7),
        block_codes,
    )

    assert torch.equal(output, block_codes.to(torch.float32).reshape(1, 2, 2))


def _write_synthetic_life(
    root: Path,
    *,
    run_id: str,
    split: str,
    base: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor] | None,
    evidence_value: float,
) -> CanonicalModelLife:
    life_root = root / run_id
    life_root.mkdir(parents=True)
    save_tensor_file(life_root / "W0.safetensors", base)
    if target is not None:
        save_tensor_file(life_root / "WT.safetensors", target)
    evidence = {
        "architecture_features": torch.tensor([1.0, 0.5]),
        "initialization_fingerprint": torch.tensor([evidence_value, -evidence_value]),
        "dataset_fingerprint": torch.tensor([evidence_value, evidence_value**2]),
        "tokenizer_fingerprint": torch.tensor([0.25, -0.25]),
        "training_recipe_fingerprint": torch.tensor([0.75, -0.75]),
    }
    save_file(evidence, str(life_root / "compiler_evidence.safetensors"))
    manifest = {
        "run_id": run_id,
        "split": split,
        "W0": {
            "canonical_file": "W0.safetensors",
            "canonical_file_sha256": sha256_file(life_root / "W0.safetensors"),
            "canonical_state_sha256": sha256_state_dict(base),
        },
        "WT": (
            {"access": "hidden", "canonical_file": None}
            if target is None
            else {
                "access": "available",
                "canonical_file": "WT.safetensors",
                "canonical_file_sha256": sha256_file(life_root / "WT.safetensors"),
                "canonical_state_sha256": sha256_state_dict(target),
            }
        ),
        "compiler_evidence": {
            "tensor_file": "compiler_evidence.safetensors",
            "tensor_file_sha256": sha256_file(life_root / "compiler_evidence.safetensors"),
            "contains_endpoint_data": False,
        },
        "source_plan_content_sha256": "f" * 64,
    }
    return CanonicalModelLife(root=life_root, manifest=manifest)


def test_complete_learned_round_one_path_runs_without_hidden_endpoint(tmp_path):
    model_config = GPTNeoXConfig(
        vocab_size=32,
        hidden_size=8,
        intermediate_size=16,
        num_hidden_layers=1,
        num_attention_heads=2,
        max_position_embeddings=16,
        tie_word_embeddings=False,
    )
    config_path = tmp_path / "config.json"
    write_json(config_path, model_config.to_dict(), canonical=True)
    lives = []
    for index, (run_id, split) in enumerate(
        (
            ("train-0", "training"),
            ("train-1", "training"),
            ("development-0", "development"),
            ("hidden-0", "hidden"),
        )
    ):
        torch.manual_seed(100 + index)
        native = GPTNeoXForCausalLM(model_config).state_dict()
        base = canonicalize_gpt_neox_state(native)
        target = {
            name: tensor
            + (index + 1) * 0.001 * torch.sin(torch.arange(tensor.numel()).reshape(tensor.shape))
            for name, tensor in base.items()
        }
        if split == "hidden":
            target = None
        lives.append(
            _write_synthetic_life(
                tmp_path / "lives",
                run_id=run_id,
                split=split,
                base=base,
                target=target,
                evidence_value=(index + 1) / 10,
            )
        )
    inventory, ties = build_tensor_inventory_from_state(lives[0].load_base())
    decoder_config = BlockDecoderConfig(
        block_rows=4,
        block_cols=4,
        global_code_dim=8,
        layer_code_dim=4,
        tensor_code_dim=4,
        block_code_dim=16,
        block_code_mode="residual",
        block_code_storage_dtype="float16",
        role_embedding_dim=4,
        feature_dim=31,
        hidden_dim=32,
        depth=1,
    )
    decoder_path = tmp_path / "shared-decoder"
    shared_manifest = train_shared_decoder(
        lives[:2],
        tensor_specs=inventory,
        tied_groups=ties,
        output_path=decoder_path,
        decoder_config=decoder_config,
        training_config=SharedDecoderTrainingConfig(
            seed=3,
            updates=8,
            batch_size=16,
            learning_rate=0.003,
            device="cpu",
            log_every=4,
        ),
    )
    assert shared_manifest["block_layout"]["block_code_dim"] == 16
    assert shared_manifest["block_layout"]["block_count"] > 0
    fit_genome_code_with_frozen_decoder(
        lives[2],
        tensor_specs=inventory,
        tied_groups=ties,
        shared_decoder_path=decoder_path,
        output_path=tmp_path / "development-code",
        config=LatentCodeFitConfig(
            seed=4,
            updates=4,
            batch_size=16,
            learning_rate=0.003,
            device="cpu",
            log_every=2,
        ),
    )
    tokenizer_backend = Tokenizer(
        models.WordLevel(
            vocab={
                "<unk>": 0,
                "<eos>": 1,
                "the": 2,
                "model": 3,
                "learns": 4,
                "a": 5,
                "small": 6,
                "test": 7,
            },
            unk_token="<unk>",
        )
    )
    tokenizer_backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        unk_token="<unk>",
        eos_token="<eos>",
    )
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer.save_pretrained(tokenizer_path)
    evaluation_texts_path = tmp_path / "evaluation-texts.jsonl"
    evaluation_texts_path.write_text(
        json.dumps({"text": "the model learns a small test"}) + "\n",
        encoding="utf-8",
    )
    decoder_evaluation = evaluate_shared_decoder_corpus(
        training_lives=lives[:2],
        development_life=lives[2],
        tensor_specs=inventory,
        tied_groups=ties,
        shared_decoder_path=decoder_path,
        development_code_path=tmp_path / "development-code",
        config_path=config_path,
        tokenizer_path=tokenizer_path,
        evaluation_texts_path=evaluation_texts_path,
        output_path=tmp_path / "decoder-evaluation.json",
        device="cpu",
        sequence_length=4,
        batch_size=1,
        max_batches=1,
        anchors_per_batch=1,
    )
    assert decoder_evaluation["summary"]["life_count"] == 3
    assert decoder_evaluation["hidden_endpoints_seen"] is False
    compiler_path = tmp_path / "compiler"
    compiler_manifest = train_predictive_compiler(
        lives[:2],
        lives[2],
        tensor_specs=inventory,
        tied_groups=ties,
        shared_decoder_path=decoder_path,
        output_path=compiler_path,
        config=PredictiveCompilerTrainingConfig(
            seed=5,
            updates=8,
            batch_size=16,
            learning_rate=0.003,
            hidden_dim=32,
            depth=1,
            device="cpu",
            log_every=4,
            development_batches=1,
        ),
    )
    assert compiler_manifest["compiler_kind"] == "blockwise"
    prediction_path = tmp_path / "prediction"
    prediction = predict_hidden_genome(
        lives[3],
        tensor_specs=inventory,
        tied_groups=ties,
        shared_decoder_path=decoder_path,
        compiler_path=compiler_path,
        output_path=prediction_path,
        device="cpu",
    )
    assert prediction["seal"]["target_endpoint_seen"] is False
    predicted_codes = load_file(str(prediction_path / prediction["prediction"]["code_file"]))
    assert predicted_codes["block_codes"].shape == (
        shared_manifest["block_layout"]["block_count"],
        16,
    )
    assert predicted_codes["block_codes"].dtype == torch.float16
    runtime_path = tmp_path / "runtime"
    runtime = execute_hidden_prediction(
        lives[3],
        tensor_specs=inventory,
        tied_groups=ties,
        shared_decoder_path=decoder_path,
        prediction_path=prediction_path,
        config_path=config_path,
        output_path=runtime_path,
        device="cpu",
    )
    assert runtime["target_endpoint_seen"] is False
    candidate = load_tensor_file(runtime_path / "predicted_WT.safetensors")
    model = model_from_canonical_state(config_path, candidate)
    with torch.inference_mode():
        logits = model(torch.tensor([[0, 1]])).logits
    assert logits.shape == (1, 2, 32)
    assert torch.isfinite(logits).all()


def test_lm_eval_task_definitions_pin_every_dataset_revision():
    task_root = _round_one_config().parent / "lm_eval_round1"
    expected = {
        "lambada_openai.yaml": (
            "genome_lambada_openai",
            "900124bf3b8235c6daf21033af9948b3f07346c4",
        ),
        "piqa.yaml": ("genome_piqa", "2e8ac2dffd59bac8c3c6714948f4c551a0848bb0"),
        "winogrande.yaml": (
            "genome_winogrande",
            "01e74176c63542e6b0bcb004dcdea22d94fb67b5",
        ),
        "arc_easy.yaml": (
            "genome_arc_easy",
            "210d026faf9955653af8916fad021475a3f00453",
        ),
        "sciq.yaml": ("genome_sciq", "2c94ad3e1aafab77146f384e23536f97a4849815"),
        "wikitext.yaml": (
            "genome_wikitext",
            "647234772b9554e208af6c826f23b99e3cac88c8",
        ),
    }
    for filename, (task, revision) in expected.items():
        value = (task_root / filename).read_text(encoding="utf-8")
        assert f"task: {task}\n" in value
        assert f"  revision: {revision}\n" in value
