from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch
from torch import nn

from genome.adapters.base import Track1Adapter


@dataclass(frozen=True)
class TinyConfig:
    vocab_size: int = 32
    sequence_length: int = 16
    d_model: int = 24
    n_heads: int = 4
    n_layers: int = 2
    mlp_dim: int = 48
    init_seed: int = 1337
    data_seed: int = 2026


class TinyCausalSelfAttention(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        if config.d_model % config.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.q_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.o_proj = nn.Linear(config.d_model, config.d_model, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        batch, length, width = hidden.shape
        q = self.q_proj(hidden).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        output = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        output = output.transpose(1, 2).contiguous().view(batch, length, width)
        return self.o_proj(output)


class TinyBlock(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(config.d_model)
        self.attn = TinyCausalSelfAttention(config)
        self.mlp_norm = nn.LayerNorm(config.d_model)
        self.mlp = nn.ModuleDict(
            {
                "up_proj": nn.Linear(config.d_model, config.mlp_dim, bias=False),
                "down_proj": nn.Linear(config.mlp_dim, config.d_model, bias=False),
            }
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = hidden + self.attn(self.attention_norm(hidden))
        mlp_input = self.mlp_norm(hidden)
        hidden = hidden + self.mlp["down_proj"](
            torch.nn.functional.silu(self.mlp["up_proj"](mlp_input))
        )
        return hidden


class TinyDecoderLM(nn.Module):
    def __init__(self, config: TinyConfig) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embedding = nn.Embedding(config.sequence_length, config.d_model)
        self.blocks = nn.ModuleList([TinyBlock(config) for _ in range(config.n_layers)])
        self.final_norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding.weight
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        if isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        length = input_ids.shape[1]
        if length > self.config.sequence_length:
            raise ValueError("sequence exceeds configured length")
        positions = torch.arange(length, device=input_ids.device)
        hidden = self.token_embedding(input_ids) + self.position_embedding(positions)[None, :, :]
        for block in self.blocks:
            hidden = block(hidden)
        return self.lm_head(self.final_norm(hidden))


def _make_sequences(config: TinyConfig, count: int, offset: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(config.data_seed + offset)
    starts = torch.randint(0, config.vocab_size, (count,), generator=generator)
    modes = torch.randint(0, 4, (count,), generator=generator)
    sequences = torch.empty(count, config.sequence_length, dtype=torch.long)
    for row in range(count):
        value = int(starts[row])
        mode = int(modes[row])
        for column in range(config.sequence_length):
            sequences[row, column] = value
            if mode == 0:
                value = (value + 1) % config.vocab_size
            elif mode == 1:
                value = (value + 3 + column % 2) % config.vocab_size
            elif mode == 2:
                value = (value * 2 + 1) % config.vocab_size
            else:
                value = (value + (column % 5) + 1) % config.vocab_size
    return sequences


class TinyTrack1Adapter(Track1Adapter):
    adapter_id = "examples.tiny_track1.v1"

    def __init__(self, config: TinyConfig | None = None, batch_size: int = 16) -> None:
        self.config = config or TinyConfig()
        self.batch_size = batch_size
        self._splits = {
            "train": _make_sequences(self.config, 256, 0),
            "genome_fit": _make_sequences(self.config, 64, 1),
            "fingerprint": _make_sequences(self.config, 64, 2),
            "probe": _make_sequences(self.config, 64, 3),
            "development": _make_sequences(self.config, 64, 4),
            "verifier_hidden": _make_sequences(self.config, 64, 5),
        }

    def build_model(self) -> nn.Module:
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.config.init_seed)
            return TinyDecoderLM(self.config)

    def architecture_manifest(self, model: nn.Module) -> dict[str, Any]:
        value = super().architecture_manifest(model)
        value.update({"tiny_config": asdict(self.config), "tied_embedding_lm_head": True})
        return value

    def tokenizer_manifest(self) -> dict[str, Any]:
        return {
            "type": "integer_fixture_tokenizer",
            "vocab_size": self.config.vocab_size,
            "special_tokens": {},
        }

    def corpus_manifest(self) -> dict[str, Any]:
        return {
            "type": "deterministic_synthetic_sequences",
            "purpose": "GENOME implementation smoke test only",
            "data_seed": self.config.data_seed,
        }

    def training_recipe(self) -> dict[str, Any]:
        return {
            "optimizer": "AdamW",
            "learning_rate": 0.003,
            "batch_size": self.batch_size,
            "objective": "next-token prediction",
        }

    def split_manifest(self) -> dict[str, Any]:
        return {
            "splits": {
                name: {
                    "count": len(values),
                    "record_ids": [f"{name}:{index:04d}" for index in range(len(values))],
                }
                for name, values in self._splits.items()
            }
        }

    def evaluation_batches(self, split: str, max_batches: int | None = None) -> Iterable[Any]:
        aliases = {"D_genome_fit": "genome_fit", "D_fingerprint": "fingerprint", "D_probe": "probe"}
        split = aliases.get(split, split)
        if split not in self._splits:
            raise KeyError(f"unknown tiny split: {split}")
        values = self._splits[split]
        emitted = 0
        for start in range(0, len(values), self.batch_size):
            if max_batches is not None and emitted >= max_batches:
                break
            sequence = values[start : start + self.batch_size]
            yield {"input_ids": sequence, "labels": sequence.clone()}
            emitted += 1


def create_adapter(config: dict[str, Any] | None = None, **kwargs: Any) -> TinyTrack1Adapter:
    del config
    tiny_config = TinyConfig(**kwargs.pop("model", {})) if "model" in kwargs else TinyConfig()
    return TinyTrack1Adapter(config=tiny_config, **kwargs)


def train_reference(
    output_checkpoint: str | Path,
    *,
    adapter: TinyTrack1Adapter | None = None,
    updates: int = 120,
    learning_rate: float = 0.003,
    device: str = "cpu",
) -> dict[str, float]:
    adapter = adapter or TinyTrack1Adapter()
    device_obj = torch.device(device)
    model = adapter.build_model().to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    batches = list(adapter.evaluation_batches("train"))
    model.train()
    final_loss = 0.0
    for update in range(updates):
        batch = adapter.move_batch(batches[update % len(batches)], device_obj)
        loss_sum, count = adapter.batch_loss(model, batch)
        loss = loss_sum / count
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.item())
    destination = Path(output_checkpoint)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "updates": updates}, destination)
    model.eval()
    with torch.inference_mode():
        dev = next(iter(adapter.evaluation_batches("development", max_batches=1)))
        loss_sum, count = adapter.batch_loss(model, adapter.move_batch(dev, device_obj))
    return {"final_train_loss": final_loss, "development_loss": float(loss_sum.item() / count)}
