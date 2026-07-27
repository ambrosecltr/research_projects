from __future__ import annotations

from pathlib import Path

import pytest
import torch

from poetry50m.config import load_mapping
from poetry50m.model import DecoderOnlyTransformer, ModelConfig, count_parameters
from poetry50m.model.config import Architecture
from poetry50m.model.transformer import UnitEmbedding, UnitLinear


def tiny_config(architecture: Architecture = "gpt", dropout: float = 0.0) -> ModelConfig:
    return ModelConfig(
        architecture=architecture,
        vocab_size=32,
        max_seq_len=16,
        d_model=16,
        n_layers=2,
        n_heads=4,
        ffn_dim=32,
        dropout=dropout,
    )


@pytest.mark.parametrize("architecture", ["gpt", "ngpt"])
def test_forward_and_backward(architecture: Architecture) -> None:
    model = DecoderOnlyTransformer(tiny_config(architecture))
    input_ids = torch.randint(0, 32, (3, 8))
    targets = torch.randint(0, 32, (3, 8))
    output = model(input_ids, targets)
    assert output.logits.shape == (3, 8, 32)
    assert output.loss is not None
    assert output.per_example_loss is not None
    output.loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_causality() -> None:
    torch.manual_seed(7)
    model = DecoderOnlyTransformer(tiny_config()).eval()
    prefix = torch.tensor([[1, 2, 3, 4, 5, 6]])
    altered = prefix.clone()
    altered[:, 4:] = torch.tensor([[9, 10]])
    with torch.no_grad():
        original_logits = model(prefix).logits
        altered_logits = model(altered).logits
    torch.testing.assert_close(original_logits[:, :4], altered_logits[:, :4], rtol=0, atol=0)


@pytest.mark.parametrize("architecture", ["gpt", "ngpt"])
def test_cached_logits_match_full_prefix_at_every_position(architecture: Architecture) -> None:
    torch.manual_seed(29)
    model = DecoderOnlyTransformer(tiny_config(architecture, dropout=0.2)).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 9))
    with torch.inference_mode():
        full_prefill = model(input_ids[:, :3]).logits
        cached = model.forward_cached(input_ids[:, :3])
        torch.testing.assert_close(cached.logits, full_prefill, rtol=1e-5, atol=1e-6)
        for position in range(3, input_ids.shape[1]):
            cached = model.forward_cached(input_ids[:, position : position + 1], cached.cache)
            full = model(input_ids[:, : position + 1]).logits[:, -1:]
            torch.testing.assert_close(cached.logits, full, rtol=1e-5, atol=1e-6)
            assert cached.cache.sequence_length == position + 1


def test_cache_is_request_local_and_rejects_invalid_context_use() -> None:
    model = DecoderOnlyTransformer(tiny_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 4))
    with pytest.raises(RuntimeError, match="inference_mode"):
        model.forward_cached(input_ids)
    model.train()
    with torch.inference_mode(), pytest.raises(RuntimeError, match="evaluation mode"):
        model.forward_cached(input_ids)
    model.eval()
    with torch.inference_mode():
        first = model.forward_cached(input_ids)
        second = model.forward_cached(input_ids)
        assert first.cache is not second.cache
        assert first.cache.layers[0].keys.data_ptr() != second.cache.layers[0].keys.data_ptr()
        with pytest.raises(ValueError, match="exactly one token"):
            model.forward_cached(input_ids[:, :2], first.cache)
        with pytest.raises(ValueError, match="batch size"):
            model.forward_cached(torch.ones((2, 1), dtype=torch.long), first.cache)
        full = model.forward_cached(torch.ones((1, model.config.max_seq_len), dtype=torch.long))
        with pytest.raises(ValueError, match="max_seq_len"):
            model.forward_cached(torch.ones((1, 1), dtype=torch.long), full.cache)
        with pytest.raises(ValueError, match="max_seq_len"):
            model.forward_cached(torch.ones((1, model.config.max_seq_len + 1), dtype=torch.long))


@pytest.mark.parametrize("architecture", ["gpt", "ngpt"])
def test_anchor_features_match_selected_full_logits(architecture: Architecture) -> None:
    model = DecoderOnlyTransformer(tiny_config(architecture)).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (2, 7))
    positions = torch.tensor([0, 3, 6])
    with torch.inference_mode():
        expected = model(input_ids).logits.index_select(1, positions)
        logits, residuals = model.anchor_features(input_ids, positions)
    torch.testing.assert_close(logits, expected, rtol=0, atol=0)
    assert residuals.shape == (2, 3, model.config.d_model)


def test_anchor_features_validate_mode_and_positions() -> None:
    model = DecoderOnlyTransformer(tiny_config()).eval()
    input_ids = torch.randint(0, model.config.vocab_size, (1, 5))
    with pytest.raises(RuntimeError, match="inference_mode"):
        model.anchor_features(input_ids, torch.tensor([1]))
    with torch.inference_mode():
        with pytest.raises(ValueError, match="unique"):
            model.anchor_features(input_ids, torch.tensor([1, 1]))
        with pytest.raises(ValueError, match="within"):
            model.anchor_features(input_ids, torch.tensor([5]))
        with pytest.raises(TypeError, match="positions must use"):
            model.anchor_features(input_ids, torch.tensor([1.0]))


def test_ngpt_normalization_invariants() -> None:
    model = DecoderOnlyTransformer(tiny_config("ngpt"))
    normalized_modules = [
        module for module in model.modules() if isinstance(module, (UnitLinear, UnitEmbedding))
    ]
    assert normalized_modules
    for module in normalized_modules:
        expected_axis = module.normalization_axis if isinstance(module, UnitLinear) else 1
        norms = module.normalized_weight().norm(dim=expected_axis)
        torch.testing.assert_close(norms, torch.ones_like(norms), rtol=1e-5, atol=1e-5)
    hidden = model.token_embedding(torch.randint(0, 32, (2, 5)))
    for block in model.blocks:
        hidden = block(hidden)
        torch.testing.assert_close(
            hidden.norm(dim=-1), torch.ones_like(hidden[..., 0]), rtol=1e-5, atol=1e-5
        )


def test_ngpt_uses_paper_projection_axes_and_post_step_retraction() -> None:
    model = DecoderOnlyTransformer(tiny_config("ngpt"))
    attention = model.blocks[0].attention
    mlp = model.blocks[0].mlp
    assert attention.query.normalization_axis == 1
    assert attention.key.normalization_axis == 1
    assert attention.value.normalization_axis == 1
    assert attention.output.normalization_axis == 0
    assert mlp.in_projection.normalization_axis == 1
    assert mlp.out_projection.normalization_axis == 0
    with torch.no_grad():
        attention.output.weight.mul_(3.0)
    model.retract_normalized_parameters_()
    torch.testing.assert_close(
        attention.output.weight.norm(dim=0),
        torch.ones(attention.output.weight.shape[1]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_parameter_count_matches_production_shape() -> None:
    config = ModelConfig.from_mapping(
        load_mapping(Path(__file__).parents[2] / "configs/model/track1_8m.yaml")
    )
    model = DecoderOnlyTransformer(config)
    assert config.max_seq_len == 1024
    assert count_parameters(model, trainable_only=True) == 8_335_008


def test_rejects_invalid_loss_mask() -> None:
    model = DecoderOnlyTransformer(tiny_config())
    with pytest.raises(ValueError, match="at least one"):
        model(
            torch.ones((1, 4), dtype=torch.long),
            torch.ones((1, 4), dtype=torch.long),
            torch.zeros((1, 4)),
        )
