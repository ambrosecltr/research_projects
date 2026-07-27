"""Conventional and hypersphere-normalized decoder-only transformers."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from poetry50m.model.config import ModelConfig


def _unit(x: Tensor, epsilon: float) -> Tensor:
    return F.normalize(x, p=2.0, dim=-1, eps=epsilon)


class RMSNorm(nn.Module):
    """RMS normalization with a learnable channel scale."""

    def __init__(self, width: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.epsilon = epsilon

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        normalized = x_float * torch.rsqrt(
            x_float.square().mean(dim=-1, keepdim=True) + self.epsilon
        )
        return normalized.to(dtype=input_dtype) * self.weight


class UnitEmbedding(nn.Module):
    """Embedding matrix used as a product of unit row vectors."""

    def __init__(self, vocab_size: int, width: int, epsilon: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, width))
        self.epsilon = epsilon
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(width))
        with torch.no_grad():
            self.weight.copy_(_unit(self.weight, self.epsilon))

    def normalized_weight(self) -> Tensor:
        return _unit(self.weight, self.epsilon)

    def forward(self, input_ids: Tensor) -> Tensor:
        return F.embedding(input_ids, self.normalized_weight())


class UnitLinear(nn.Module):
    """A linear map constrained to a product of unit-vector spheres.

    This makes each learned matrix a product of hyperspheres: both represented
    and stored vectors are unit-normalized after every optimizer retraction.
    NVIDIA's nGPT
    uses row vectors for Q/K/V and MLP input projections, and column vectors
    for attention and MLP output projections.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        epsilon: float,
        bias: bool = False,
        normalization_axis: int = 1,
    ) -> None:
        super().__init__()
        if normalization_axis not in {0, 1}:
            raise ValueError("normalization_axis must be 0 (columns) or 1 (rows)")
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.epsilon = epsilon
        self.normalization_axis = normalization_axis
        nn.init.normal_(self.weight, std=1.0 / math.sqrt(in_features))
        self.retract_()

    def normalized_weight(self) -> Tensor:
        return F.normalize(self.weight, p=2.0, dim=self.normalization_axis, eps=self.epsilon)

    @torch.no_grad()
    def retract_(self) -> None:
        """Project the stored parameter back to its product-of-spheres manifold."""
        self.weight.copy_(self.normalized_weight())

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.normalized_weight(), self.bias)


class RotaryEmbedding(nn.Module):
    """RoPE with configurable rotated fraction and no fixed context cache."""

    def __init__(self, dimension: int, base: float) -> None:
        super().__init__()
        inverse_frequency = 1.0 / (base ** (torch.arange(0, dimension, 2).float() / dimension))
        self.inverse_frequency: Tensor
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)
        self.dimension = dimension

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        # x: [batch, heads, sequence, head_dim], positions: [sequence]
        angles = torch.outer(positions.to(self.inverse_frequency.dtype), self.inverse_frequency)
        cos = angles.cos().to(dtype=x.dtype)[None, None, :, :]
        sin = angles.sin().to(dtype=x.dtype)[None, None, :, :]
        rotated, remainder = x[..., : self.dimension], x[..., self.dimension :]
        even, odd = rotated[..., 0::2], rotated[..., 1::2]
        paired = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)
        return torch.cat((paired, remainder), dim=-1)


@dataclass(slots=True)
class AttentionKVCache:
    """Preallocated post-RoPE keys and values for one attention layer."""

    keys: Tensor
    values: Tensor
    length: int = 0

    def __post_init__(self) -> None:
        if self.keys.ndim != 4 or self.values.ndim != 4:
            raise ValueError("attention cache tensors must have [batch, heads, capacity, width]")
        if self.keys.shape != self.values.shape:
            raise ValueError("attention cache key and value shapes must match")
        if self.keys.device != self.values.device or self.keys.dtype != self.values.dtype:
            raise ValueError("attention cache keys and values must share device and dtype")
        if isinstance(self.length, bool) or not isinstance(self.length, int):
            raise TypeError("attention cache length must be an integer")
        if not 0 <= self.length <= self.capacity:
            raise ValueError("attention cache length exceeds its capacity")

    @property
    def batch_size(self) -> int:
        return self.keys.shape[0]

    @property
    def capacity(self) -> int:
        return self.keys.shape[2]

    @classmethod
    def allocate(cls, keys: Tensor, values: Tensor, capacity: int) -> AttentionKVCache:
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("projected cache tensors must have matching four-dimensional shapes")
        if keys.shape[2] > capacity:
            raise ValueError("prefill sequence exceeds attention cache capacity")
        shape = (keys.shape[0], keys.shape[1], capacity, keys.shape[3])
        cache = cls(keys.new_empty(shape), values.new_empty(shape))
        cache.append(keys, values)
        return cache

    def append(self, keys: Tensor, values: Tensor) -> tuple[Tensor, Tensor]:
        if keys.ndim != 4 or values.shape != keys.shape:
            raise ValueError("appended cache tensors must have matching four-dimensional shapes")
        if (
            keys.shape[:2] != self.keys.shape[:2]
            or keys.shape[3] != self.keys.shape[3]
            or keys.device != self.keys.device
            or keys.dtype != self.keys.dtype
            or values.device != self.values.device
            or values.dtype != self.values.dtype
        ):
            raise ValueError("appended keys and values do not match the attention cache")
        end = self.length + keys.shape[2]
        if end > self.capacity:
            raise ValueError("attention cache capacity exceeded")
        self.keys[:, :, self.length : end].copy_(keys)
        self.values[:, :, self.length : end].copy_(values)
        self.length = end
        return self.keys[:, :, :end], self.values[:, :, :end]


@dataclass(slots=True)
class TransformerKVCache:
    """Request-local cache for every transformer layer."""

    layers: tuple[AttentionKVCache, ...]

    def __post_init__(self) -> None:
        if not self.layers:
            raise ValueError("transformer cache must contain at least one layer")
        self._validated_shared_dimension("length", tuple(layer.length for layer in self.layers))
        self._validated_shared_dimension("capacity", tuple(layer.capacity for layer in self.layers))
        self._validated_shared_dimension(
            "batch_size", tuple(layer.batch_size for layer in self.layers)
        )

    @staticmethod
    def _validated_shared_dimension(name: str, dimensions: tuple[int, ...]) -> int:
        values = set(dimensions)
        if len(values) != 1:
            raise ValueError(f"transformer cache layers disagree on {name}")
        return values.pop()

    @property
    def sequence_length(self) -> int:
        return self._validated_shared_dimension(
            "length", tuple(layer.length for layer in self.layers)
        )

    @property
    def capacity(self) -> int:
        return self._validated_shared_dimension(
            "capacity", tuple(layer.capacity for layer in self.layers)
        )

    @property
    def batch_size(self) -> int:
        return self._validated_shared_dimension(
            "batch_size", tuple(layer.batch_size for layer in self.layers)
        )


class CausalSelfAttention(nn.Module):
    def __init__(self, config: ModelConfig, normalized: bool) -> None:
        super().__init__()
        self.query: UnitLinear | None
        self.key: UnitLinear | None
        self.value: UnitLinear | None
        self.qkv: nn.Linear | None
        self.output: UnitLinear | nn.Linear
        self.qk_scale: nn.Parameter | None
        self.qk_scale_initial_value: float
        if normalized:
            self.query = UnitLinear(
                config.d_model,
                config.d_model,
                config.norm_epsilon,
                config.linear_bias,
                normalization_axis=1,
            )
            self.key = UnitLinear(
                config.d_model,
                config.d_model,
                config.norm_epsilon,
                config.linear_bias,
                normalization_axis=1,
            )
            self.value = UnitLinear(
                config.d_model,
                config.d_model,
                config.norm_epsilon,
                config.linear_bias,
                normalization_axis=1,
            )
            self.output = UnitLinear(
                config.d_model,
                config.d_model,
                config.norm_epsilon,
                config.linear_bias,
                normalization_axis=0,
            )
            self.qkv = None
            self.qk_scale = nn.Parameter(
                config.base_scale * torch.ones(config.d_model, dtype=torch.float32)
            )
            self.qk_scale_initial_value = 1.0
        else:
            self.qkv = nn.Linear(config.d_model, 3 * config.d_model, bias=config.linear_bias)
            self.output = nn.Linear(config.d_model, config.d_model, bias=config.linear_bias)
            self.query = None
            self.key = None
            self.value = None
            self.qk_scale = None
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.base_scale = config.base_scale
        self.normalized = normalized
        self.dropout = config.dropout
        self.epsilon = config.norm_epsilon
        self.rope = RotaryEmbedding(config.rope_dim, config.rope_base)

    def _project_qkv(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        batch_size, sequence_length, _ = x.shape
        if self.normalized:
            if (
                self.query is None
                or self.key is None
                or self.value is None
                or self.qk_scale is None
            ):
                raise RuntimeError("normalized attention parameters were not initialized")
            query = self.query(x).view(batch_size, sequence_length, self.n_heads, self.head_dim)
            key = self.key(x).view(batch_size, sequence_length, self.n_heads, self.head_dim)
            value = self.value(x).view(batch_size, sequence_length, self.n_heads, self.head_dim)
        else:
            if self.qkv is None:
                raise RuntimeError("standard attention projection was not initialized")
            qkv = self.qkv(x).view(batch_size, sequence_length, 3, self.n_heads, self.head_dim)
            query, key, value = qkv.unbind(dim=2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        value = value.transpose(1, 2)
        return query, key, value

    def _position_qk(self, query: Tensor, key: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        query = self.rope(query, positions)
        key = self.rope(key, positions)
        if self.normalized:
            if self.qk_scale is None:
                raise RuntimeError("normalized attention scale was not initialized")
            scale = self.qk_scale.mul(self.qk_scale_initial_value / self.base_scale).view(
                1, self.n_heads, 1, self.head_dim
            )
            query = scale * _unit(query, self.epsilon)
            key = scale * _unit(key, self.epsilon)
            # SDPA divides by sqrt(d); nGPT's attention uses sqrt(d), so Q is
            # multiplied by d to produce the same effective logit scaling.
            query = query * self.head_dim
        return query, key

    def _output_projection(self, attended: Tensor, width: int) -> Tensor:
        batch_size, _, sequence_length, _ = attended.shape
        output = attended.transpose(1, 2).contiguous().view(batch_size, sequence_length, width)
        if isinstance(self.output, UnitLinear):
            return cast(Tensor, self.output(output))
        return F.linear(output, self.output.weight, self.output.bias)

    def forward(self, x: Tensor) -> Tensor:
        _, sequence_length, width = x.shape
        query, key, value = self._project_qkv(x)
        positions = torch.arange(sequence_length, device=x.device)
        query, key = self._position_qk(query, key, positions)
        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        return self._output_projection(attended, width)

    def forward_cached(
        self,
        x: Tensor,
        positions: Tensor,
        cache: AttentionKVCache | None,
        *,
        capacity: int,
    ) -> tuple[Tensor, AttentionKVCache]:
        if self.training:
            raise RuntimeError("attention caching is available only in evaluation mode")
        _, sequence_length, width = x.shape
        if positions.shape != (sequence_length,):
            raise ValueError("cache positions must match the input sequence length")
        query, key, value = self._project_qkv(x)
        query, key = self._position_qk(query, key, positions)
        if cache is None:
            cache = AttentionKVCache.allocate(key, value, capacity)
            full_key = cache.keys[:, :, : cache.length]
            full_value = cache.values[:, :, : cache.length]
            is_causal = sequence_length > 1
        else:
            if sequence_length != 1:
                raise ValueError("cached continuation accepts exactly one token")
            full_key, full_value = cache.append(key, value)
            is_causal = False
        attended = F.scaled_dot_product_attention(
            query,
            full_key,
            full_value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        return self._output_projection(attended, width), cache


class SwiGLU(nn.Module):
    def __init__(self, config: ModelConfig, normalized: bool) -> None:
        super().__init__()
        self.in_projection: UnitLinear | nn.Linear
        self.out_projection: UnitLinear | nn.Linear
        self.uv_scale: nn.Parameter | None
        self.uv_scale_initial_value: float
        self.uv_scale_initial_scaling: float
        self.width: int
        if normalized:
            self.in_projection = UnitLinear(
                config.d_model,
                2 * config.ffn_dim,
                config.norm_epsilon,
                config.linear_bias,
                normalization_axis=1,
            )
            self.out_projection = UnitLinear(
                config.ffn_dim,
                config.d_model,
                config.norm_epsilon,
                config.linear_bias,
                normalization_axis=0,
            )
            self.uv_scale = nn.Parameter(torch.ones(2 * config.ffn_dim, dtype=torch.float32))
            self.uv_scale_initial_value = 1.0
            self.uv_scale_initial_scaling = 1.0
            self.width = config.d_model
        else:
            self.in_projection = nn.Linear(
                config.d_model, 2 * config.ffn_dim, bias=config.linear_bias
            )
            self.out_projection = nn.Linear(config.ffn_dim, config.d_model, bias=config.linear_bias)
            self.uv_scale = None

    def forward(self, x: Tensor) -> Tensor:
        if isinstance(self.in_projection, UnitLinear):
            projected = self.in_projection(x)
        else:
            projected = F.linear(x, self.in_projection.weight, self.in_projection.bias)
        if self.uv_scale is not None:
            projected = projected * (
                self.uv_scale
                * (self.uv_scale_initial_value / self.uv_scale_initial_scaling)
                * math.sqrt(self.width)
            )
        value, gate = projected.chunk(2, dim=-1)
        output = F.silu(gate) * value
        if isinstance(self.out_projection, UnitLinear):
            return cast(Tensor, self.out_projection(output))
        return F.linear(output, self.out_projection.weight, self.out_projection.bias)


class GPTBlock(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.norm_epsilon)
        self.attention = CausalSelfAttention(config, normalized=False)
        self.mlp_norm = RMSNorm(config.d_model, config.norm_epsilon)
        self.mlp = SwiGLU(config, normalized=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: Tensor) -> Tensor:
        attention_update = cast(Tensor, self.attention(self.attention_norm(x)))
        x = x + self.dropout(attention_update)
        mlp_update = cast(Tensor, self.mlp(self.mlp_norm(x)))
        return cast(Tensor, x + self.dropout(mlp_update))

    def forward_cached(
        self,
        x: Tensor,
        positions: Tensor,
        cache: AttentionKVCache | None,
        *,
        capacity: int,
    ) -> tuple[Tensor, AttentionKVCache]:
        attention_update, updated_cache = self.attention.forward_cached(
            self.attention_norm(x), positions, cache, capacity=capacity
        )
        x = x + self.dropout(attention_update)
        mlp_update = cast(Tensor, self.mlp(self.mlp_norm(x)))
        return cast(Tensor, x + self.dropout(mlp_update)), updated_cache


class nGPTBlock(nn.Module):
    """Normalized residual block inspired by nGPT's hypersphere geometry.

    Hidden states and nGPT's axis-specific matrix vectors remain normalized.
    Learned residual interpolation rates keep the residual path stable while
    allowing each channel to choose its update rate.
    """

    _INITIAL_RESIDUAL_RATE: Final[float] = 0.05

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(config, normalized=True)
        self.mlp = SwiGLU(config, normalized=True)
        self.attention_rate = nn.Parameter(
            config.base_scale * torch.ones(config.d_model, dtype=torch.float32)
        )
        self.mlp_rate = nn.Parameter(
            config.base_scale * torch.ones(config.d_model, dtype=torch.float32)
        )
        self.residual_rate_initial_scaling = config.base_scale
        self.dropout = nn.Dropout(config.dropout)
        self.epsilon = config.norm_epsilon

    def _mix(self, previous: Tensor, update: Tensor, rate_logits: Tensor) -> Tensor:
        rate = torch.abs(
            rate_logits * (self._INITIAL_RESIDUAL_RATE / self.residual_rate_initial_scaling)
        ).view(1, 1, -1)
        previous = _unit(previous, self.epsilon)
        update = _unit(self.dropout(update), self.epsilon)
        return _unit(previous + rate * (update - previous), self.epsilon)

    def forward(self, x: Tensor) -> Tensor:
        x = self._mix(x, self.attention(_unit(x, self.epsilon)), self.attention_rate)
        return self._mix(x, self.mlp(_unit(x, self.epsilon)), self.mlp_rate)

    def forward_cached(
        self,
        x: Tensor,
        positions: Tensor,
        cache: AttentionKVCache | None,
        *,
        capacity: int,
    ) -> tuple[Tensor, AttentionKVCache]:
        attention_update, updated_cache = self.attention.forward_cached(
            _unit(x, self.epsilon), positions, cache, capacity=capacity
        )
        x = self._mix(x, attention_update, self.attention_rate)
        return self._mix(x, self.mlp(_unit(x, self.epsilon)), self.mlp_rate), updated_cache


@dataclass(slots=True)
class ModelOutput:
    logits: Tensor
    loss: Tensor | None = None
    per_example_loss: Tensor | None = None
    token_count: int = 0


@dataclass(frozen=True, slots=True)
class CachedModelOutput:
    logits: Tensor
    cache: TransformerKVCache


class DecoderOnlyTransformer(nn.Module):
    """A tied-embedding causal LM with either GPT or nGPT-style blocks."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        if config.architecture == "ngpt":
            self.token_embedding: nn.Module = UnitEmbedding(
                config.vocab_size, config.d_model, config.norm_epsilon
            )
            block_factory: type[nn.Module] = nGPTBlock
        else:
            embedding = nn.Embedding(config.vocab_size, config.d_model)
            nn.init.normal_(embedding.weight, std=0.02)
            self.token_embedding = embedding
            block_factory = GPTBlock
        self.blocks = nn.ModuleList([block_factory(config) for _ in range(config.n_layers)])
        if config.tie_embeddings:
            self.output_projection: nn.Module | None = None
        elif config.architecture == "ngpt":
            # The official nGPT implementation deliberately uses a separate
            # normalized output embedding matrix; Track 1's nGPT config follows it.
            self.output_projection = UnitLinear(
                config.d_model,
                config.vocab_size,
                config.norm_epsilon,
                bias=False,
                normalization_axis=1,
            )
        else:
            self.output_projection = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.final_norm: nn.Module = (
            nn.Identity()
            if config.architecture == "ngpt"
            else RMSNorm(config.d_model, config.norm_epsilon)
        )
        self.dropout = nn.Dropout(config.dropout)
        self.logit_scale: nn.Parameter | None
        self.logit_scale_initial_value: float
        self.logit_scale_initial_scaling: float
        if config.architecture == "ngpt":
            self.logit_scale = nn.Parameter(
                config.base_scale * torch.ones(config.vocab_size, dtype=torch.float32)
            )
            self.logit_scale_initial_value = 1.0
            self.logit_scale_initial_scaling = config.base_scale
        else:
            self.logit_scale = None
        self._initialize_standard_weights()

    def _initialize_standard_weights(self) -> None:
        if self.config.architecture != "gpt":
            return
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def output_weight(self) -> Tensor:
        if self.output_projection is not None:
            if isinstance(self.output_projection, UnitLinear):
                return self.output_projection.normalized_weight()
            if isinstance(self.output_projection, nn.Linear):
                return self.output_projection.weight
            raise TypeError("output_projection is not a supported projection module")
        if isinstance(self.token_embedding, UnitEmbedding):
            return self.token_embedding.normalized_weight()
        if isinstance(self.token_embedding, nn.Embedding):
            return self.token_embedding.weight
        raise TypeError("token_embedding is not a supported embedding module")

    @torch.no_grad()
    def retract_normalized_parameters_(self) -> None:
        """Project every nGPT matrix back onto its declared product manifold."""
        if self.config.architecture != "ngpt":
            return
        if isinstance(self.token_embedding, UnitEmbedding):
            self.token_embedding.weight.copy_(
                _unit(self.token_embedding.weight, self.config.norm_epsilon)
            )
        for module in self.modules():
            if isinstance(module, UnitLinear):
                module.retract_()

    def _validate_input_ids(self, input_ids: Tensor) -> None:
        if input_ids.ndim != 2:
            raise ValueError(
                f"input_ids must have shape [batch, sequence], got {tuple(input_ids.shape)}"
            )
        if input_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("input_ids must use torch.int32 or torch.int64")
        if input_ids.shape[1] < 1:
            raise ValueError("input_ids must contain at least one token")

    def _logits(self, hidden: Tensor) -> Tensor:
        logits = F.linear(hidden, self.output_weight())
        if self.logit_scale is not None:
            logits = logits * (
                self.logit_scale
                * (self.logit_scale_initial_value / self.logit_scale_initial_scaling)
            )
        return logits

    def _final_hidden(self, input_ids: Tensor, active_layers: int) -> Tensor:
        hidden = self.dropout(self.token_embedding(input_ids))
        for block in self.blocks[:active_layers]:
            hidden = block(hidden)
        return cast(Tensor, self.final_norm(hidden))

    def forward(
        self,
        input_ids: Tensor,
        targets: Tensor | None = None,
        loss_mask: Tensor | None = None,
        active_layers: int | None = None,
    ) -> ModelOutput:
        self._validate_input_ids(input_ids)
        if input_ids.shape[1] > self.config.max_seq_len:
            sequence_length = input_ids.shape[1]
            limit = self.config.max_seq_len
            raise ValueError(f"sequence length {sequence_length} exceeds max_seq_len {limit}")
        if active_layers is None:
            active_layers = self.config.n_layers
        if not 1 <= active_layers <= self.config.n_layers:
            raise ValueError(
                f"active_layers must be in [1, {self.config.n_layers}], got {active_layers}"
            )
        hidden = self._final_hidden(input_ids, active_layers)
        logits = self._logits(hidden)
        if targets is None:
            if loss_mask is not None:
                raise ValueError("loss_mask requires targets")
            return ModelOutput(logits=logits)
        return self._loss_output(logits, targets, loss_mask)

    def forward_cached(
        self,
        input_ids: Tensor,
        cache: TransformerKVCache | None = None,
    ) -> CachedModelOutput:
        """Prefill or extend a request-local inference cache."""
        if self.training:
            raise RuntimeError("cached forward is available only in evaluation mode")
        if torch.is_grad_enabled():
            raise RuntimeError("cached forward requires inference_mode or no_grad")
        self._validate_input_ids(input_ids)
        if cache is None:
            start_position = 0
            layer_caches: tuple[AttentionKVCache | None, ...] = (None,) * len(self.blocks)
        else:
            if len(cache.layers) != len(self.blocks):
                raise ValueError("cache layer count does not match the model")
            if cache.capacity != self.config.max_seq_len:
                raise ValueError("cache capacity does not match model max_seq_len")
            if cache.batch_size != input_ids.shape[0]:
                raise ValueError("cache batch size does not match input_ids")
            if input_ids.shape[1] != 1:
                raise ValueError("cached continuation accepts exactly one token")
            start_position = cache.sequence_length
            layer_caches = cache.layers
        end_position = start_position + input_ids.shape[1]
        if end_position > self.config.max_seq_len:
            raise ValueError("cached sequence exceeds max_seq_len")
        positions = torch.arange(start_position, end_position, device=input_ids.device)
        hidden = cast(Tensor, self.token_embedding(input_ids))
        updated_caches: list[AttentionKVCache] = []
        for block, layer_cache in zip(self.blocks, layer_caches, strict=True):
            if isinstance(block, GPTBlock):
                hidden, updated_cache = block.forward_cached(
                    hidden,
                    positions,
                    layer_cache,
                    capacity=self.config.max_seq_len,
                )
            elif isinstance(block, nGPTBlock):
                hidden, updated_cache = block.forward_cached(
                    hidden,
                    positions,
                    layer_cache,
                    capacity=self.config.max_seq_len,
                )
            else:
                raise TypeError("cached forward encountered an unsupported block type")
            updated_caches.append(updated_cache)
        hidden = cast(Tensor, self.final_norm(hidden))
        updated = cache if cache is not None else TransformerKVCache(layers=tuple(updated_caches))
        return CachedModelOutput(logits=self._logits(hidden), cache=updated)

    def anchor_features(self, input_ids: Tensor, positions: Tensor) -> tuple[Tensor, Tensor]:
        """Return selected logits and final residuals through the exact eval path."""
        if self.training:
            raise RuntimeError("anchor_features is available only in evaluation mode")
        if torch.is_grad_enabled():
            raise RuntimeError("anchor_features requires inference_mode or no_grad")
        self._validate_input_ids(input_ids)
        if input_ids.shape[1] > self.config.max_seq_len:
            raise ValueError("anchor feature sequence exceeds max_seq_len")
        if positions.ndim != 1 or positions.numel() < 1:
            raise ValueError("anchor positions must be a non-empty rank-1 tensor")
        if positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("anchor positions must use torch.int32 or torch.int64")
        if torch.unique(positions).numel() != positions.numel():
            raise ValueError("anchor positions must be unique")
        if int(positions.min().item()) < 0 or int(positions.max().item()) >= input_ids.shape[1]:
            raise ValueError("anchor positions must lie within the input sequence")
        hidden = self._final_hidden(input_ids, self.config.n_layers)
        selected_hidden = hidden.index_select(1, positions.to(device=input_ids.device))
        return self._logits(selected_hidden), selected_hidden

    def _loss_output(
        self, logits: Tensor, targets: Tensor, loss_mask: Tensor | None
    ) -> ModelOutput:
        if targets.shape != logits.shape[:2]:
            target_shape = tuple(targets.shape)
            input_shape = tuple(logits.shape[:2])
            raise ValueError(f"targets shape {target_shape} must match input shape {input_shape}")
        if targets.dtype not in (torch.int32, torch.int64):
            raise TypeError("targets must use torch.int32 or torch.int64")
        token_loss = F.cross_entropy(
            logits.float().flatten(0, 1),
            targets.flatten(),
            ignore_index=self.config.ignore_index,
            reduction="none",
        ).view_as(targets)
        valid = targets.ne(self.config.ignore_index)
        if loss_mask is not None:
            if loss_mask.shape != targets.shape:
                raise ValueError("loss_mask must have the same shape as targets")
            if not loss_mask.dtype.is_floating_point and loss_mask.dtype != torch.bool:
                raise TypeError("loss_mask must be floating point or bool")
            if loss_mask.dtype.is_floating_point and (
                not torch.isfinite(loss_mask).all() or torch.any(loss_mask < 0)
            ):
                raise ValueError("loss_mask must contain only finite, non-negative weights")
            valid = valid & loss_mask.to(dtype=torch.bool)
            weights = loss_mask.to(dtype=token_loss.dtype) * valid.to(dtype=token_loss.dtype)
        else:
            weights = valid.to(dtype=token_loss.dtype)
        per_example_denominator = weights.sum(dim=1)
        if torch.any(per_example_denominator <= 0):
            raise ValueError("each example must contain at least one unmasked target token")
        per_example_loss = (token_loss * weights).sum(dim=1) / per_example_denominator
        total_weight = weights.sum()
        if total_weight <= 0:
            raise ValueError("batch contains no valid target tokens")
        loss = (token_loss * weights).sum() / total_weight
        return ModelOutput(
            logits=logits,
            loss=loss,
            per_example_loss=per_example_loss,
            token_count=int(valid.sum().item()),
        )


def count_parameters(module: nn.Module, trainable_only: bool = False) -> int:
    """Return the exact de-duplicated parameter count (tied parameters count once)."""
    parameters = module.parameters()
    seen: set[int] = set()
    count = 0
    for parameter in parameters:
        if trainable_only and not parameter.requires_grad:
            continue
        if id(parameter) not in seen:
            seen.add(id(parameter))
            count += parameter.numel()
    return count
