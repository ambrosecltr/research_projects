from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.func import functional_call

from ..architecture import ArchitectureGraph
from ..mgp.schema import Component, ModelGenomeProgram, TensorProgram

ROLES = (
    "embedding",
    "attention_norm",
    "qkv",
    "attention_output",
    "mlp_norm",
    "mlp_up",
    "mlp_down",
    "final_norm",
    "lm_head",
    "bias",
    "other",
)
ROLE_TO_ID = {name: index for index, name in enumerate(ROLES)}
PRIMITIVES = ("BASE_COPY", "LOW_RANK", "DIRECT_VECTOR")


@dataclass(frozen=True)
class CompilerConfig:
    global_feature_dim: int = 256
    tensor_feature_dim: int = 24
    coordinate_feature_dim: int = 8
    d_model: int = 256
    n_heads: int = 8
    transformer_layers: int = 4
    message_layers: int = 2
    max_rank: int = 32
    dropout: float = 0.0
    target_fraction: float = 0.10
    max_vector_values: int = 4096
    manifest_reserve_bytes: int = 65536
    shared_vocabulary_factors: bool = True


@dataclass
class TensorEvidence:
    name: str
    role: str
    shape: tuple[int, ...]
    tied_to: str | None
    features: torch.Tensor
    row_features: torch.Tensor | None = None
    col_features: torch.Tensor | None = None


@dataclass
class CompilerExample:
    architecture: ArchitectureGraph
    global_features: torch.Tensor
    tensors: list[TensorEvidence]
    base_state_id: str

    def __post_init__(self) -> None:
        if len(self.tensors) != len(self.architecture.tensors):
            raise ValueError("compiler evidence must have one record per architecture tensor")


@dataclass
class CompilerPrediction:
    contexts: torch.Tensor
    primitive_logits: torch.Tensor
    rank_logits: torch.Tensor


def _vocabulary_pair(example: CompilerExample) -> tuple[int, int] | None:
    indices = [
        index
        for index, evidence in enumerate(example.tensors)
        if evidence.role in {"embedding", "lm_head"} and evidence.tied_to is None
    ]
    if len(indices) != 2:
        return None
    left, right = indices
    left_shape = example.tensors[left].shape
    right_shape = example.tensors[right].shape
    if len(left_shape) != 2 or len(right_shape) != 2 or left_shape[0] != right_shape[0]:
        return None
    return left, right


class MessageBlock(torch.nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.update = torch.nn.Sequential(
            torch.nn.Linear(d_model * 2, d_model * 2),
            torch.nn.GELU(),
            torch.nn.Linear(d_model * 2, d_model),
        )
        self.norm = torch.nn.LayerNorm(d_model)

    def forward(self, tokens: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        degree = adjacency.sum(dim=-1, keepdim=True).clamp_min(1.0)
        neighbours = adjacency @ tokens / degree
        return self.norm(tokens + self.update(torch.cat([tokens, neighbours], dim=-1)))


class CoordinateHead(torch.nn.Module):
    def __init__(self, d_model: int, coordinate_feature_dim: int) -> None:
        super().__init__()
        self.coordinate_feature_dim = coordinate_feature_dim
        self.network = torch.nn.Sequential(
            torch.nn.Linear(d_model + coordinate_feature_dim + 5, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, d_model),
            torch.nn.GELU(),
            torch.nn.Linear(d_model, 1),
        )

    def _coordinate_features(self, size: int, device: torch.device) -> torch.Tensor:
        position = torch.linspace(0.0, 1.0, size, device=device).unsqueeze(1)
        frequencies = torch.arange(1, self.coordinate_feature_dim // 2 + 1, device=device).float()
        angles = position * frequencies.unsqueeze(0) * math.pi
        features = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
        return features[:, : self.coordinate_feature_dim]

    def forward(
        self,
        context: torch.Tensor,
        *,
        size: int,
        rank: int,
        side: float,
        local_features: torch.Tensor | None,
    ) -> torch.Tensor:
        if rank <= 0:
            return torch.empty((size, 0), device=context.device)
        coordinate = self._coordinate_features(size, context.device)
        if local_features is None:
            local = torch.zeros((size, 3), device=context.device)
        else:
            local = local_features.to(context.device).float()
            if local.shape[0] != size:
                raise ValueError("coordinate feature length does not match tensor dimension")
            if local.shape[1] < 3:
                local = F.pad(local, (0, 3 - local.shape[1]))
            local = local[:, :3]
        outputs: list[torch.Tensor] = []
        for component in range(rank):
            component_feature = torch.full(
                (size, 1),
                0.0 if rank == 1 else component / (rank - 1),
                device=context.device,
            )
            side_feature = torch.full((size, 1), side, device=context.device)
            expanded = context.unsqueeze(0).expand(size, -1)
            inputs = torch.cat(
                [expanded, coordinate, local, component_feature, side_feature], dim=-1
            )
            outputs.append(self.network(inputs).squeeze(-1))
        return torch.stack(outputs, dim=1)


class GenomeCompiler(torch.nn.Module):
    """One model that emits a compact MGP skeleton and bounded coefficient packets."""

    def __init__(self, config: CompilerConfig = CompilerConfig()) -> None:
        super().__init__()
        self.config = config
        self.global_projection = torch.nn.Linear(config.global_feature_dim, config.d_model)
        self.tensor_projection = torch.nn.Linear(config.tensor_feature_dim, config.d_model)
        self.role_embedding = torch.nn.Embedding(len(ROLES), config.d_model)
        self.message_blocks = torch.nn.ModuleList(
            MessageBlock(config.d_model) for _ in range(config.message_layers)
        )
        layer = torch.nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.n_heads,
            dim_feedforward=config.d_model * 4,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = torch.nn.TransformerEncoder(layer, num_layers=config.transformer_layers)
        self.primitive_head = torch.nn.Linear(config.d_model, len(PRIMITIVES))
        self.rank_head = torch.nn.Linear(config.d_model, config.max_rank + 1)
        self.left_head = CoordinateHead(config.d_model, config.coordinate_feature_dim)
        self.right_head = CoordinateHead(config.d_model, config.coordinate_feature_dim)
        self.row_scale_head = CoordinateHead(config.d_model, config.coordinate_feature_dim)
        self.column_scale_head = CoordinateHead(config.d_model, config.coordinate_feature_dim)
        self.vector_head = CoordinateHead(config.d_model, config.coordinate_feature_dim)

    def forward(self, example: CompilerExample) -> CompilerPrediction:
        device = next(self.parameters()).device
        global_feature = example.global_features.to(device).float()
        if global_feature.numel() != self.config.global_feature_dim:
            raise ValueError("global feature dimension mismatch")
        tensor_features = (
            torch.stack([item.features for item in example.tensors]).to(device).float()
        )
        if tensor_features.shape[1] != self.config.tensor_feature_dim:
            raise ValueError("tensor feature dimension mismatch")
        role_ids = torch.tensor(
            [ROLE_TO_ID.get(item.role, ROLE_TO_ID["other"]) for item in example.tensors],
            device=device,
        )
        tokens = self.tensor_projection(tensor_features) + self.role_embedding(role_ids)
        adjacency = torch.zeros((len(example.tensors), len(example.tensors)), device=device)
        for left, right in example.architecture.edges:
            adjacency[left, right] = 1.0
        adjacency.fill_diagonal_(1.0)
        for block in self.message_blocks:
            tokens = block(tokens, adjacency)
        global_token = self.global_projection(global_feature).reshape(1, 1, -1)
        encoded = self.encoder(torch.cat([global_token, tokens.unsqueeze(0)], dim=1))
        contexts = encoded[0, 1:]
        return CompilerPrediction(
            contexts=contexts,
            primitive_logits=self.primitive_head(contexts),
            rank_logits=self.rank_head(contexts),
        )

    def factors(
        self,
        context: torch.Tensor,
        evidence: TensorEvidence,
        rank: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(evidence.shape) != 2:
            raise ValueError("low-rank factors require a matrix")
        rows, cols = evidence.shape
        left = self.left_head(
            context,
            size=rows,
            rank=rank,
            side=0.0,
            local_features=evidence.row_features,
        )
        right = self.right_head(
            context,
            size=cols,
            rank=rank,
            side=1.0,
            local_features=evidence.col_features,
        )
        return left, right

    def shared_vocabulary_factor(
        self,
        contexts: torch.Tensor,
        evidence: Sequence[TensorEvidence],
        rank: int,
    ) -> torch.Tensor:
        if len(evidence) != 2 or any(len(item.shape) != 2 for item in evidence):
            raise ValueError("shared vocabulary factors require two matrices")
        rows = {item.shape[0] for item in evidence}
        if len(rows) != 1:
            raise ValueError("shared vocabulary matrices require the same row count")
        row_features = None
        if all(item.row_features is not None for item in evidence):
            row_features = torch.stack(
                [item.row_features for item in evidence if item.row_features is not None]
            ).mean(dim=0)
        return self.left_head(
            contexts.mean(dim=0),
            size=next(iter(rows)),
            rank=rank,
            side=0.0,
            local_features=row_features,
        )

    def right_factor(
        self,
        context: torch.Tensor,
        evidence: TensorEvidence,
        rank: int,
    ) -> torch.Tensor:
        if len(evidence.shape) != 2:
            raise ValueError("low-rank right factors require a matrix")
        return self.right_head(
            context,
            size=evidence.shape[1],
            rank=rank,
            side=1.0,
            local_features=evidence.col_features,
        )

    def vector(self, context: torch.Tensor, evidence: TensorEvidence) -> torch.Tensor:
        if len(evidence.shape) != 1:
            raise ValueError("vector head requires a vector")
        return self.vector_head(
            context,
            size=evidence.shape[0],
            rank=1,
            side=0.5,
            local_features=evidence.row_features,
        ).squeeze(1)

    def scales(
        self,
        context: torch.Tensor,
        evidence: TensorEvidence,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(evidence.shape) != 2:
            raise ValueError("Hadamard scales require a matrix")
        rows, columns = evidence.shape
        row = self.row_scale_head(
            context,
            size=rows,
            rank=1,
            side=0.25,
            local_features=evidence.row_features,
        ).squeeze(1)
        column = self.column_scale_head(
            context,
            size=columns,
            rank=1,
            side=0.75,
            local_features=evidence.col_features,
        ).squeeze(1)
        return row, column

    def expected_bytes(
        self, example: CompilerExample, prediction: CompilerPrediction
    ) -> torch.Tensor:
        primitive_prob = prediction.primitive_logits.softmax(dim=-1)
        rank_prob = prediction.rank_logits.softmax(dim=-1)
        ranks = torch.arange(self.config.max_rank + 1, device=rank_prob.device).float()
        expected_rank = (rank_prob * ranks).sum(dim=-1)
        costs = []
        vocabulary_pair = (
            _vocabulary_pair(example) if self.config.shared_vocabulary_factors else None
        )
        vocabulary_indices = set(vocabulary_pair or ())
        for index, evidence in enumerate(example.tensors):
            if index in vocabulary_indices:
                continue
            if len(evidence.shape) == 2:
                dimensions = evidence.shape[0] + evidence.shape[1]
                low_rank = 2.0 * dimensions * (expected_rank[index] + 1.0)
                costs.append(primitive_prob[index, 1] * low_rank)
            elif len(evidence.shape) == 1 and evidence.shape[0] <= self.config.max_vector_values:
                costs.append(primitive_prob[index, 2] * float(2 * evidence.shape[0]))
            else:
                costs.append(torch.zeros((), device=rank_prob.device))
        if vocabulary_pair is not None:
            left, right = vocabulary_pair
            left_evidence = example.tensors[left]
            right_evidence = example.tensors[right]
            primitive = primitive_prob[list(vocabulary_pair), 1].mean()
            rank = expected_rank[list(vocabulary_pair)].mean()
            factor_dimensions = (
                left_evidence.shape[0] + left_evidence.shape[1] + right_evidence.shape[1]
            )
            scale_dimensions = (
                left_evidence.shape[0]
                + left_evidence.shape[1]
                + right_evidence.shape[0]
                + right_evidence.shape[1]
            )
            costs.append(primitive * 2.0 * (factor_dimensions * rank + scale_dimensions))
        return torch.stack(costs).sum()

    def generate_program(
        self,
        example: CompilerExample,
        *,
        direct_fp16_delta_bytes: int,
    ) -> tuple[ModelGenomeProgram, dict[str, torch.Tensor]]:
        self.eval()
        with torch.no_grad():
            prediction = self(example)
            desired_primitive = prediction.primitive_logits.argmax(dim=-1).tolist()
            desired_rank = prediction.rank_logits.argmax(dim=-1).tolist()
            # Enforce the hard target-specific budget by reducing ranks with the weakest marginal
            # confidence. This bounds output packets without a giant autoregressive sequence.
            budget = max(
                0,
                int(direct_fp16_delta_bytes * self.config.target_fraction)
                - self.config.manifest_reserve_bytes,
            )
            ranks = [0] * len(example.tensors)
            vector_enabled = [False] * len(example.tensors)
            import heapq

            heap: list[tuple[float, int, int, int, int]] = []
            # (-score, tensor index, component number, maximum rank, component cost)
            used = 0
            rank_probabilities: dict[int, torch.Tensor] = {}
            vocabulary_pair = (
                _vocabulary_pair(example) if self.config.shared_vocabulary_factors else None
            )
            vocabulary_indices = set(vocabulary_pair or ())
            for index, evidence in enumerate(example.tensors):
                if evidence.tied_to is not None:
                    continue
                if index in vocabulary_indices:
                    continue
                if len(evidence.shape) == 1 and desired_primitive[index] == 2:
                    cost = 2 * evidence.shape[0]
                    if evidence.shape[0] <= self.config.max_vector_values and used + cost <= budget:
                        vector_enabled[index] = True
                        used += cost
                elif len(evidence.shape) == 2 and desired_primitive[index] == 1:
                    maximum = min(desired_rank[index], self.config.max_rank, min(evidence.shape))
                    if maximum <= 0:
                        continue
                    probabilities = prediction.rank_logits[index].softmax(dim=-1)
                    rank_probabilities[index] = probabilities
                    cost = 4 * (evidence.shape[0] + evidence.shape[1])
                    marginal = float(probabilities[1 : maximum + 1].sum())
                    heapq.heappush(heap, (-marginal / max(1, cost), index, 1, maximum, cost))
            rank_component_cost: dict[int, int] = {}
            rank_members: dict[int, tuple[int, ...]] = {}
            if vocabulary_pair is not None and all(
                desired_primitive[index] == 1 for index in vocabulary_pair
            ):
                owner = vocabulary_pair[0]
                members = tuple(vocabulary_pair)
                maximum = min(
                    *(desired_rank[index] for index in members),
                    self.config.max_rank,
                    *(min(example.tensors[index].shape) for index in members),
                )
                if maximum > 0:
                    probabilities = torch.stack(
                        [prediction.rank_logits[index].softmax(dim=-1) for index in members]
                    ).mean(dim=0)
                    rank_probabilities[owner] = probabilities
                    rank_members[owner] = members
                    first = example.tensors[members[0]]
                    second = example.tensors[members[1]]
                    component_cost = 2 * (first.shape[0] + first.shape[1] + second.shape[1])
                    scale_cost = 2 * (
                        first.shape[0] + first.shape[1] + second.shape[0] + second.shape[1]
                    )
                    rank_component_cost[owner] = component_cost
                    first_cost = component_cost + scale_cost
                    marginal = float(probabilities[1 : maximum + 1].sum())
                    heapq.heappush(
                        heap,
                        (-marginal / max(1, first_cost), owner, 1, maximum, first_cost),
                    )
            while heap:
                _, index, component, maximum, cost = heapq.heappop(heap)
                if used + cost > budget:
                    continue
                if component != ranks[index] + 1:
                    continue
                for member in rank_members.get(index, (index,)):
                    ranks[member] = component
                used += cost
                next_component = component + 1
                if next_component <= maximum:
                    probabilities = rank_probabilities[index]
                    marginal = float(probabilities[next_component : maximum + 1].sum())
                    cost = rank_component_cost.get(
                        index,
                        2 * (example.tensors[index].shape[0] + example.tensors[index].shape[1]),
                    )
                    heapq.heappush(
                        heap,
                        (-marginal / max(1, cost), index, next_component, maximum, cost),
                    )
            payloads: dict[str, torch.Tensor] = {}
            tensor_programs: list[TensorProgram] = []
            shared_vocabulary_left = None
            if vocabulary_pair is not None and ranks[vocabulary_pair[0]] > 0:
                shared_vocabulary_left = self.shared_vocabulary_factor(
                    prediction.contexts[list(vocabulary_pair)],
                    [example.tensors[index] for index in vocabulary_pair],
                    ranks[vocabulary_pair[0]],
                )
            for index, evidence in enumerate(example.tensors):
                if evidence.tied_to is not None:
                    tensor_programs.append(
                        TensorProgram(
                            name=evidence.name,
                            shape=evidence.shape,
                            tied_to=evidence.tied_to,
                            components=(
                                Component("COPY_FROM_TIED", arguments={"owner": evidence.tied_to}),
                            ),
                        )
                    )
                    continue
                components: list[Component] = [Component("BASE_COPY")]
                if ranks[index] > 0:
                    row_scale, column_scale = self.scales(
                        prediction.contexts[index],
                        evidence,
                    )
                    row_scale_key = f"tensor.{index}.hadamard_scale.row"
                    column_scale_key = f"tensor.{index}.hadamard_scale.column"
                    payloads[row_scale_key] = row_scale.cpu().to(torch.float16)
                    payloads[column_scale_key] = column_scale.cpu().to(torch.float16)
                    components.append(
                        Component(
                            "HADAMARD_SCALE",
                            payload={"row": row_scale_key, "column": column_scale_key},
                        )
                    )
                    if index in vocabulary_indices and shared_vocabulary_left is not None:
                        left = shared_vocabulary_left
                        right = self.right_factor(
                            prediction.contexts[index], evidence, ranks[index]
                        )
                        left_key = "shared.vocabulary.low_rank.left"
                    else:
                        left, right = self.factors(
                            prediction.contexts[index], evidence, ranks[index]
                        )
                        left_key = f"tensor.{index}.low_rank.left"
                    right_key = f"tensor.{index}.low_rank.right"
                    payloads[left_key] = left.cpu().to(torch.float16)
                    payloads[right_key] = right.cpu().to(torch.float16)
                    components.append(
                        Component(
                            "LOW_RANK",
                            payload={"left": left_key, "right": right_key},
                            arguments={"rank": ranks[index]},
                        )
                    )
                elif vector_enabled[index]:
                    values = self.vector(prediction.contexts[index], evidence)
                    value_key = f"tensor.{index}.vector.values"
                    payloads[value_key] = values.cpu().to(torch.float16)
                    components.append(
                        Component(
                            "DIRECT_VECTOR",
                            payload={"values": value_key},
                        )
                    )
                tensor_programs.append(
                    TensorProgram(
                        name=evidence.name, shape=evidence.shape, components=tuple(components)
                    )
                )
            return (
                ModelGenomeProgram(
                    architecture_id=example.architecture.graph_id,
                    base_state_id=example.base_state_id,
                    tensors=tuple(tensor_programs),
                ),
                payloads,
            )


def decode_teacher_forced(
    compiler: GenomeCompiler,
    example: CompilerExample,
    prediction: CompilerPrediction,
    *,
    target_primitives: torch.Tensor,
    target_ranks: torch.Tensor,
    w0_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    device = prediction.contexts.device
    primitives = target_primitives.to(device)
    ranks = target_ranks.to(device)
    decoded: dict[str, torch.Tensor] = {}
    vocabulary_pair = (
        _vocabulary_pair(example) if compiler.config.shared_vocabulary_factors else None
    )
    shared_vocabulary_left = None
    if (
        vocabulary_pair is not None
        and all(int(primitives[index]) == 1 for index in vocabulary_pair)
        and int(ranks[vocabulary_pair[0]]) > 0
        and int(ranks[vocabulary_pair[0]]) == int(ranks[vocabulary_pair[1]])
    ):
        shared_vocabulary_left = compiler.shared_vocabulary_factor(
            prediction.contexts[list(vocabulary_pair)],
            [example.tensors[index] for index in vocabulary_pair],
            int(ranks[vocabulary_pair[0]]),
        )
    for index, evidence in enumerate(example.tensors):
        primitive = int(primitives[index])
        rank = int(ranks[index])
        if evidence.tied_to is not None:
            decoded[evidence.name] = decoded[evidence.tied_to]
        elif primitive == 1 and rank > 0 and len(evidence.shape) == 2:
            if vocabulary_pair is not None and index in vocabulary_pair:
                left = shared_vocabulary_left
                if left is None:
                    left, right = compiler.factors(prediction.contexts[index], evidence, rank)
                else:
                    right = compiler.right_factor(prediction.contexts[index], evidence, rank)
            else:
                left, right = compiler.factors(prediction.contexts[index], evidence, rank)
            row_scale, column_scale = compiler.scales(prediction.contexts[index], evidence)
            base = w0_state[evidence.name].to(device).float()
            decoded[evidence.name] = left @ right.transpose(0, 1) + base * (
                row_scale.unsqueeze(1) + column_scale.unsqueeze(0)
            )
        elif primitive == 2 and len(evidence.shape) == 1:
            decoded[evidence.name] = compiler.vector(prediction.contexts[index], evidence)
        else:
            decoded[evidence.name] = torch.zeros(evidence.shape, device=device)
    return decoded


def compiler_loss(
    compiler: GenomeCompiler,
    example: CompilerExample,
    *,
    target_primitives: torch.Tensor,
    target_ranks: torch.Tensor,
    target_deltas: Mapping[str, torch.Tensor],
    w0_state: Mapping[str, torch.Tensor],
    rate_weight: float = 1e-6,
    functional_model: torch.nn.Module | None = None,
    functional_batches: Sequence[Mapping[str, torch.Tensor]] = (),
    functional_weight: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    prediction = compiler(example)
    device = prediction.contexts.device
    primitive_targets = target_primitives.to(device)
    rank_targets = target_ranks.to(device)
    primitive_loss = F.cross_entropy(prediction.primitive_logits, primitive_targets)
    rank_mask = primitive_targets == 1
    rank_loss = (
        F.cross_entropy(prediction.rank_logits[rank_mask], rank_targets[rank_mask])
        if bool(rank_mask.any())
        else torch.zeros((), device=device)
    )
    rank_correct = prediction.rank_logits.argmax(dim=-1) == rank_targets
    rank_accuracy = (
        rank_correct[rank_mask].float().mean()
        if bool(rank_mask.any())
        else rank_correct.float().mean()
    )
    predicted = decode_teacher_forced(
        compiler,
        example,
        prediction,
        target_primitives=target_primitives,
        target_ranks=target_ranks,
        w0_state=w0_state,
    )
    reconstruction = torch.zeros((), device=device)
    for evidence in example.tensors:
        target = target_deltas[evidence.name].to(device).float()
        decoded = predicted[evidence.name]
        reconstruction = reconstruction + (decoded - target).square().sum() / (
            target.square().sum() + 1e-8
        )
    reconstruction = reconstruction / max(1, len(example.tensors))
    expected_bytes = compiler.expected_bytes(example, prediction)
    functional = torch.zeros((), device=device)
    if functional_weight > 0:
        if functional_model is None or not functional_batches:
            raise ValueError("functional compiler loss requires model, W0 state and batches")
        functional_model = functional_model.to(device).eval()
        predicted_state = {
            name: w0_state[name].to(device).float() + predicted[name] for name in predicted
        }
        losses = []
        for batch in functional_batches:
            inputs = {name: value.to(device) for name, value in batch.items()}
            outputs = functional_call(functional_model, predicted_state, (), inputs)
            if outputs.loss is None or not torch.isfinite(outputs.loss):
                raise ValueError("compiler functional path produced a non-finite task loss")
            losses.append(outputs.loss)
        functional = torch.stack(losses).mean()
    loss = (
        primitive_loss
        + rank_loss
        + reconstruction
        + rate_weight * expected_bytes
        + functional_weight * functional
    )
    return loss, {
        "loss": float(loss.detach()),
        "primitive_loss": float(primitive_loss.detach()),
        "rank_loss": float(rank_loss.detach()),
        "primitive_accuracy": float(
            (prediction.primitive_logits.argmax(dim=-1) == primitive_targets).float().mean()
        ),
        "rank_accuracy": float(rank_accuracy),
        "reconstruction_loss": float(reconstruction.detach()),
        "functional_loss": float(functional.detach()),
        "expected_bytes": float(expected_bytes.detach()),
    }
