from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn.functional as F
from torch.func import functional_call

from ..architecture import ArchitectureGraph
from ..hashing import stable_u64
from ..state import state_id
from .runtime import execute_program
from .schema import Component, ModelGenomeProgram, TensorProgram
from .serialize import serialized_program_bytes

AllocationStrategy = Literal["energy", "rank_balanced"]


@dataclass(frozen=True)
class FitConfig:
    budget_fraction: float = 0.10
    max_rank: int = 32
    minimum_matrix_rank: int = 0
    allocation_strategy: AllocationStrategy = "energy"
    matrix_scaling: bool = False
    scaling_iterations: int = 8
    vector_quantization: bool = True
    account_for_serialization: bool = False
    svd_method: str = "randomized"
    oversample: int = 8
    power_iterations: int = 4
    seed: int = 20260729
    device: str = "cpu"


def _canonicalize_svd_signs(
    left: torch.Tensor, right: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    left = left.clone()
    right = right.clone()
    for column in range(left.shape[1]):
        vector = left[:, column]
        pivot = int(vector.abs().argmax())
        if vector[pivot] < 0:
            left[:, column].neg_()
            right[:, column].neg_()
    return left, right


def _quantize_vector(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = delta.abs().max()
    scale = torch.tensor(1.0, dtype=torch.float32) if maximum == 0 else maximum.float() / 127.0
    values = torch.round(delta.float() / scale).clamp(-127, 127).to(torch.int8)
    return values, scale.reshape(1)


def _fit_hadamard_scale(
    base: torch.Tensor,
    delta: torch.Tensor,
    *,
    iterations: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    execution_device = torch.device(
        device if device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    base_value = base.to(device=execution_device, dtype=torch.float32)
    delta_value = delta.to(device=execution_device, dtype=torch.float32)
    row = torch.zeros(base_value.shape[0], device=execution_device)
    column = torch.zeros(base_value.shape[1], device=execution_device)
    for _ in range(iterations):
        row = (base_value * (delta_value - base_value * column.unsqueeze(0))).sum(dim=1) / (
            base_value.square().sum(dim=1) + 1e-12
        )
        column = (base_value * (delta_value - base_value * row.unsqueeze(1))).sum(dim=0) / (
            base_value.square().sum(dim=0) + 1e-12
        )
    scaling = base_value * (row.unsqueeze(1) + column.unsqueeze(0))
    return row.cpu(), column.cpu(), (delta_value - scaling).cpu()


def _truncated_svd(
    delta: torch.Tensor, name: str, config: FitConfig
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    device = torch.device(
        config.device if config.device != "cuda" or torch.cuda.is_available() else "cpu"
    )
    values = delta.to(device)
    limit = min(config.max_rank, min(values.shape))
    if limit <= 0:
        raise ValueError("matrix must have positive rank")
    if config.svd_method == "exact":
        u, s, vh = torch.linalg.svd(values, full_matrices=False)
        return u[:, :limit].cpu(), s[:limit].cpu(), vh[:limit].cpu()
    if config.svd_method != "randomized":
        raise ValueError(f"unsupported SVD method {config.svd_method!r}")
    q = min(min(values.shape), limit + max(2, config.oversample))
    devices = [device.index or 0] if device.type == "cuda" else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(config.seed + int(stable_u64(name) % 1_000_000))
        u, s, v = torch.svd_lowrank(values, q=q, niter=config.power_iterations)
    return u[:, :limit].cpu(), s[:limit].cpu(), v[:, :limit].transpose(0, 1).cpu()


def fit_low_rank_program(
    base_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
    graph: ArchitectureGraph,
    *,
    config: FitConfig = FitConfig(),
) -> tuple[ModelGenomeProgram, dict[str, torch.Tensor]]:
    """Fit a transparent compact candidate with global singular-component allocation.

    This produces a candidate, not accepted compiler supervision. It must be serialized, audited,
    executed in the real model and pass the functional gate.
    """
    if set(base_state) != set(target_state):
        raise ValueError("base and target state dictionaries differ")
    if not 0 < config.budget_fraction < 1:
        raise ValueError("budget_fraction must be between zero and one")
    if config.minimum_matrix_rank < 0:
        raise ValueError("minimum_matrix_rank must be non-negative")
    if config.minimum_matrix_rank > config.max_rank:
        raise ValueError("minimum_matrix_rank cannot exceed max_rank")
    if config.allocation_strategy not in {"energy", "rank_balanced"}:
        raise ValueError(f"unsupported allocation_strategy {config.allocation_strategy!r}")
    if config.scaling_iterations <= 0:
        raise ValueError("scaling_iterations must be positive")
    direct_bytes = sum(tensor.numel() * 2 for tensor in target_state.values())
    budget = int(direct_bytes * config.budget_fraction)
    tensor_by_name = {node.name: node for node in graph.tensors}
    payloads: dict[str, torch.Tensor] = {}
    vector_components: dict[str, Component] = {}
    vector_cost = 0
    scaling_components: dict[str, Component] = {}
    scaling_cost = 0
    candidates: list[tuple[float, int, str, int, torch.Tensor, torch.Tensor, torch.Tensor]] = []
    # score, incremental bytes, name, component index, U, S, V
    for name in sorted(base_state):
        node = tensor_by_name[name]
        delta = target_state[name].detach().cpu().float() - base_state[name].detach().cpu().float()
        if node.tied_to is not None:
            continue
        if delta.ndim == 1 and config.vector_quantization:
            values, scale = _quantize_vector(delta)
            cost = values.numel() + 4
            if values.numel() <= 4096:
                prefix = f"tensor.{node.index}.vector"
                payloads[f"{prefix}.values"] = values
                payloads[f"{prefix}.scale"] = scale
                vector_components[name] = Component(
                    primitive="QUANTIZED_VECTOR",
                    payload={"values": f"{prefix}.values", "scale": f"{prefix}.scale"},
                )
                vector_cost += cost
            continue
        if delta.ndim != 2:
            continue
        if (
            config.matrix_scaling
            and delta.shape[0] + delta.shape[1] < delta.shape[0] * delta.shape[1]
        ):
            row, column, delta = _fit_hadamard_scale(
                base_state[name],
                delta,
                iterations=config.scaling_iterations,
                device=config.device,
            )
            prefix = f"tensor.{node.index}.hadamard_scale"
            payloads[f"{prefix}.row"] = row.to(torch.float16)
            payloads[f"{prefix}.column"] = column.to(torch.float16)
            scaling_components[name] = Component(
                primitive="HADAMARD_SCALE",
                payload={
                    "row": f"{prefix}.row",
                    "column": f"{prefix}.column",
                },
            )
            scaling_cost += 2 * (delta.shape[0] + delta.shape[1])
        u, s, vh = _truncated_svd(delta, name, config)
        rank_limit = min(config.max_rank, s.numel())
        for index in range(rank_limit):
            bytes_for_component = 2 * (delta.shape[0] + delta.shape[1])
            energy = float(s[index].square())
            score = energy / max(1, bytes_for_component)
            candidates.append((score, bytes_for_component, name, index, u, s, vh))
    svd_cache: dict[str, tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    for item in candidates:
        name = item[2]
        svd_cache.setdefault(name, (item[4], item[5], item[6]))
    minimum_ranks = {
        name: min(config.minimum_matrix_rank, values[1].numel())
        for name, values in svd_cache.items()
    }
    minimum_cost = sum(
        cost for _, cost, name, index, *_ in candidates if index < minimum_ranks[name]
    )

    def assemble(
        allocation_budget: int,
    ) -> tuple[
        ModelGenomeProgram,
        dict[str, torch.Tensor],
        int,
    ]:
        fixed_cost = vector_cost + scaling_cost + minimum_cost
        if allocation_budget < fixed_cost:
            raise ValueError("serialized budget cannot provide the requested minimum matrix ranks")
        remaining = allocation_budget - fixed_cost
        selected = dict(minimum_ranks)
        selected_cost = minimum_cost
        ordered_candidates = (
            sorted(candidates, key=lambda item: item[0], reverse=True)
            if config.allocation_strategy == "energy"
            else sorted(candidates, key=lambda item: (item[3], -item[0], item[2]))
        )
        for _, cost, name, index, *_ in ordered_candidates:
            if index < minimum_ranks[name]:
                continue
            if cost > remaining:
                continue
            if index != selected.get(name, 0):
                # Components must be prefixes of the ordered SVD.
                continue
            selected[name] = index + 1
            remaining -= cost
            selected_cost += cost
        selected_payloads = dict(payloads)
        programs: list[TensorProgram] = []
        for node in graph.tensors:
            if node.tied_to is not None:
                programs.append(
                    TensorProgram(
                        name=node.name,
                        shape=node.shape,
                        tied_to=node.tied_to,
                        components=(
                            Component(
                                "COPY_FROM_TIED",
                                arguments={"owner": node.tied_to},
                            ),
                        ),
                    )
                )
                continue
            components: list[Component] = [Component("BASE_COPY")]
            if node.name in scaling_components:
                components.append(scaling_components[node.name])
            rank = selected.get(node.name, 0)
            if rank:
                u, s, vh = svd_cache[node.name]
                root = s[:rank].sqrt()
                left = u[:, :rank] * root.unsqueeze(0)
                right = vh[:rank, :].transpose(0, 1) * root.unsqueeze(0)
                left, right = _canonicalize_svd_signs(left, right)
                prefix = f"tensor.{node.index}.low_rank"
                selected_payloads[f"{prefix}.left"] = left.to(torch.float16)
                selected_payloads[f"{prefix}.right"] = right.to(torch.float16)
                components.append(
                    Component(
                        primitive="LOW_RANK",
                        payload={
                            "left": f"{prefix}.left",
                            "right": f"{prefix}.right",
                        },
                        arguments={"rank": rank},
                    )
                )
            elif node.name in vector_components:
                components.append(vector_components[node.name])
            programs.append(
                TensorProgram(
                    name=node.name,
                    shape=node.shape,
                    components=tuple(components),
                )
            )
        program = ModelGenomeProgram(
            architecture_id=graph.graph_id,
            base_state_id=state_id(base_state),
            tensors=tuple(programs),
        )
        used = vector_cost + scaling_cost + selected_cost
        return program, selected_payloads, used

    allocation_budget = budget
    while True:
        program, selected_payloads, used = assemble(allocation_budget)
        if not config.account_for_serialization:
            return program, selected_payloads
        serialized_bytes = serialized_program_bytes(program, selected_payloads)
        if serialized_bytes <= budget:
            return program, selected_payloads
        if used <= vector_cost + scaling_cost + minimum_cost:
            raise ValueError(
                "serialized program metadata, fixed components and minimum ranks exceed the budget"
            )
        overshoot = serialized_bytes - budget
        allocation_budget = min(
            allocation_budget - overshoot,
            used - 1,
        )


class TrainableProgram(torch.nn.Module):
    """Differentiable compact program parameters; the child model weights are not trainable."""

    def __init__(self, program: ModelGenomeProgram, payloads: Mapping[str, torch.Tensor]) -> None:
        super().__init__()
        self.program = program
        self.parameters_by_key = torch.nn.ParameterDict()
        self.constants: dict[str, torch.Tensor] = {}
        self.storage_dtypes: dict[str, torch.dtype] = {}
        for key, value in payloads.items():
            self.storage_dtypes[key] = value.dtype
            if value.is_floating_point():
                self.parameters_by_key[key.replace(".", "__")] = torch.nn.Parameter(value.float())
            else:
                self.constants[key] = value.detach().clone()

    def payloads(self) -> dict[str, torch.Tensor]:
        result = dict(self.constants)
        for key, value in self.parameters_by_key.items():
            result[key.replace("__", ".")] = value
        return result

    def export_payloads(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().to(dtype=self.storage_dtypes[key], device="cpu")
            for key, value in self.payloads().items()
        }

    def materialize(self, base_state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return execute_program(
            base_state, self.program, self.payloads(), output_dtype=torch.float32
        )


def _token_mean_kl(student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
    per_token = F.kl_div(
        F.log_softmax(student_logits, dim=-1),
        F.softmax(teacher_logits, dim=-1),
        reduction="none",
    ).sum(dim=-1)
    return per_token.mean()


def refine_program_functionally(
    model: torch.nn.Module,
    base_state: Mapping[str, torch.Tensor],
    program: ModelGenomeProgram,
    payloads: Mapping[str, torch.Tensor],
    batches: Iterable[Mapping[str, torch.Tensor]],
    *,
    steps: int = 100,
    learning_rate: float = 1e-3,
    teacher_model: torch.nn.Module | None = None,
    kl_weight: float = 0.0,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    trainable = TrainableProgram(program, payloads).to(device)
    model = model.to(device).eval()
    if teacher_model is not None:
        teacher_model = teacher_model.to(device).eval()
    optimizer = torch.optim.AdamW(trainable.parameters(), lr=learning_rate)
    cached = list(batches)
    if not cached:
        raise ValueError("functional refinement requires at least one batch")
    for step in range(steps):
        batch = {key: value.to(device) for key, value in cached[step % len(cached)].items()}
        optimizer.zero_grad(set_to_none=True)
        state = {
            name: tensor.to(device) for name, tensor in trainable.materialize(base_state).items()
        }
        outputs = functional_call(model, state, (), batch)
        loss = outputs.loss
        if loss is None:
            raise ValueError("model did not return a task loss")
        if teacher_model is not None and kl_weight > 0:
            with torch.no_grad():
                teacher = teacher_model(**batch).logits
            student = outputs.logits
            loss = loss + kl_weight * _token_mean_kl(student, teacher)
        if not torch.isfinite(loss):
            raise ValueError("functional refinement produced non-finite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable.parameters(), 1.0)
        optimizer.step()
    return trainable.export_payloads()
