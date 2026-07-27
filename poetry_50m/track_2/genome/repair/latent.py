from __future__ import annotations

from dataclasses import dataclass
from itertools import cycle
from typing import Any, Mapping, Sequence

import torch
from torch.func import functional_call

from ..adapters.base import Track1Adapter
from ..mgp.interpreter import decode_program
from ..types import GenomeProgram, TensorSpec


@dataclass(frozen=True)
class LatentRefinementConfig:
    steps: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    split: str = "probe"
    max_batches: int | None = None
    device: str = "cpu"


@dataclass
class LatentRefinementResult:
    program: GenomeProgram
    metrics: list[dict[str, float]]
    optimized_keys: list[str]


def _code_keys(program: GenomeProgram) -> list[str]:
    keys = set(program.manifest.get("shared_payload_keys", []))
    for record in program.records:
        for component in record.components:
            if component.opcode == "NEURAL_BLOCK_FIELD":
                keys.update(component.payload_keys)
    return sorted(keys)


def refine_neural_genome_codes(
    adapter: Track1Adapter,
    program: GenomeProgram,
    interpreter: Any,
    base_state: Mapping[str, torch.Tensor],
    inventory: Sequence[TensorSpec],
    *,
    tied_groups: Sequence[Sequence[str]] = (),
    config: LatentRefinementConfig | None = None,
) -> LatentRefinementResult:
    """Optimize only genome codes on an allowed probe split.

    The shared interpreter is frozen. This is G0/G1 repair, not evidence of zero-training synthesis.
    """
    config = config or LatentRefinementConfig()
    device = torch.device(config.device)
    candidate = program.clone_without_payload_aliases()
    keys = _code_keys(candidate)
    if not keys:
        raise ValueError("program contains no neural genome codes")
    parameters = []
    for key, value in list(candidate.payload_tensors.items()):
        if key in keys:
            parameter = torch.nn.Parameter(value.detach().to(device).clone())
            candidate.payload_tensors[key] = parameter
            parameters.append(parameter)
        else:
            candidate.payload_tensors[key] = value.detach().to(device)
    candidate.patch_tensors = {key: value.detach().to(device) for key, value in candidate.patch_tensors.items()}
    base = {key: value.detach().to(device) for key, value in base_state.items()}

    for parameter in interpreter.decoder.parameters():
        parameter.requires_grad_(False)
    model = adapter.build_model().to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        parameters, lr=config.learning_rate, weight_decay=config.weight_decay
    )
    batches = list(adapter.evaluation_batches(config.split, max_batches=config.max_batches))
    if not batches:
        raise ValueError("probe split produced no batches")
    iterator = cycle(batches)
    metrics = []
    for step in range(1, config.steps + 1):
        batch = adapter.move_batch(next(iterator), device)
        state = decode_program(
            candidate,
            base,
            inventory,
            tied_groups=tied_groups,
            interpreter=interpreter,
            verify_checksums=False,
        )
        args, kwargs = adapter.model_call(batch)
        output = functional_call(
            model,
            state,
            args,
            kwargs,
            tie_weights=False,
            strict=True,
        )
        logits = adapter.extract_logits(output)
        loss_sum, count = adapter.loss_from_logits(logits, batch)
        loss = loss_sum / max(count, 1)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.grad_clip_norm)
        optimizer.step()
        if step == 1 or step % max(1, config.steps // 20) == 0 or step == config.steps:
            metrics.append({"step": float(step), "probe_loss": float(loss.detach().item())})

    candidate.payload_tensors = {
        key: value.detach().cpu().contiguous() for key, value in candidate.payload_tensors.items()
    }
    candidate.patch_tensors = {
        key: value.detach().cpu().contiguous() for key, value in candidate.patch_tensors.items()
    }
    candidate.manifest = {
        **candidate.manifest,
        "latent_refinement": {
            "steps": config.steps,
            "learning_rate": config.learning_rate,
            "split": config.split,
            "optimized_keys": keys,
        },
    }
    return LatentRefinementResult(program=candidate, metrics=metrics, optimized_keys=keys)
