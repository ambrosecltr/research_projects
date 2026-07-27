from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Iterable, Mapping

import torch
from torch import nn


class Track1Adapter(ABC):
    """The only boundary between GENOME and the concrete Track 1 project.

    The real project should implement this class in a small adapter module. GENOME must not
    import Track 1 internals from arbitrary locations after this boundary exists.
    """

    adapter_id: str = "track1-adapter"

    @abstractmethod
    def build_model(self) -> nn.Module:
        """Construct the exact Track 1 architecture in its initial state."""

    def initial_state(self) -> Mapping[str, torch.Tensor]:
        """Return a reproducible W0 state.

        Override when construction alone does not replay initialization exactly.
        """
        return self.build_model().state_dict()

    def validate_endpoint_checkpoint(self, path: str | Path) -> dict[str, Any]:
        """Validate that ``path`` is eligible to become the frozen WT endpoint.

        Generic adapters cannot infer completion, so they return a minimal record. Concrete
        adapters should fail closed when a checkpoint is partial, malformed, or from the wrong run.
        """
        resolved = Path(path).expanduser().resolve(strict=True)
        return {"path": str(resolved), "complete": True}

    def validate_base_checkpoint(
        self, path: str | Path, *, endpoint_checkpoint: str | Path | None = None
    ) -> dict[str, Any]:
        """Validate that ``path`` is the exact W0 corresponding to the endpoint lineage.

        Generic adapters can only validate existence. Concrete adapters should check step zero,
        architecture/training identities, and run lineage whenever those records are available.
        """
        resolved = Path(path).expanduser().resolve(strict=True)
        return {"path": str(resolved), "valid_base": True}

    def load_checkpoint(self, model: nn.Module, path: str | Path) -> None:
        """Load a source Track 1 checkpoint into ``model``.

        Source checkpoints may use torch serialization. MGP artifacts never do.
        """
        checkpoint = torch.load(Path(path), map_location="cpu", weights_only=True)
        if isinstance(checkpoint, Mapping):
            for key in ("model", "state_dict", "model_state_dict"):
                if key in checkpoint and isinstance(checkpoint[key], Mapping):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, Mapping):
            raise TypeError(f"unsupported checkpoint object: {type(checkpoint)!r}")
        missing, unexpected = model.load_state_dict(dict(checkpoint), strict=False)
        if missing or unexpected:
            raise ValueError(f"checkpoint mismatch; missing={missing}, unexpected={unexpected}")

    def architecture_manifest(self, model: nn.Module) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "model_class": f"{model.__class__.__module__}:{model.__class__.__qualname__}",
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        }

    def tokenizer_manifest(self) -> dict[str, Any]:
        return {
            "adapter_id": self.adapter_id,
            "status": "adapter did not provide tokenizer metadata",
        }

    def corpus_manifest(self) -> dict[str, Any]:
        return {"adapter_id": self.adapter_id, "status": "adapter did not provide corpus metadata"}

    def training_recipe(self) -> dict[str, Any]:
        return {"adapter_id": self.adapter_id, "status": "adapter did not provide training recipe"}

    def split_manifest(self) -> dict[str, Any]:
        return {"adapter_id": self.adapter_id, "splits": {}}

    def evaluation_batches(self, split: str, max_batches: int | None = None) -> Iterable[Any]:
        raise NotImplementedError("adapter does not expose functional evaluation batches")

    def move_batch(self, batch: Any, device: torch.device) -> Any:
        if isinstance(batch, torch.Tensor):
            return batch.to(device)
        if isinstance(batch, Mapping):
            return {key: self.move_batch(value, device) for key, value in batch.items()}
        if isinstance(batch, tuple):
            return tuple(self.move_batch(value, device) for value in batch)
        if isinstance(batch, list):
            return [self.move_batch(value, device) for value in batch]
        return batch

    def model_call(self, batch: Any) -> tuple[tuple[Any, ...], dict[str, Any]]:
        """Return positional/keyword inputs for the model, excluding labels and record IDs."""
        if isinstance(batch, Mapping):
            return (), {k: v for k, v in batch.items() if k not in {"labels", "record_ids"}}
        if isinstance(batch, tuple):
            return tuple(batch), {}
        return (batch,), {}

    def extract_logits(self, output: Any) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if hasattr(output, "logits"):
            return output.logits
        if isinstance(output, Mapping) and "logits" in output:
            return output["logits"]
        raise TypeError("adapter could not extract logits from model output")

    def forward_logits(self, model: nn.Module, batch: Any) -> torch.Tensor:
        args, kwargs = self.model_call(batch)
        return self.extract_logits(model(*args, **kwargs))

    def loss_from_logits(self, logits: torch.Tensor, batch: Any) -> tuple[torch.Tensor, int]:
        """Return summed objective and denominator from already-computed logits."""
        if not isinstance(batch, Mapping) or "labels" not in batch:
            raise NotImplementedError("override loss_from_logits or provide a mapping with labels")
        labels = batch["labels"]
        if logits.ndim != 3 or labels.ndim != 2:
            raise ValueError("default language-model loss expects logits [B,T,V] and labels [B,T]")
        if logits.shape[:2] == labels.shape:
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = labels[:, 1:].contiguous()
        elif logits.shape[1] + 1 == labels.shape[1]:
            shift_logits = logits
            shift_labels = labels[:, 1:].contiguous()
        else:
            raise ValueError("logit/label sequence shapes are incompatible")
        valid = shift_labels.ne(-100)
        count = int(valid.sum().item())
        loss = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
            reduction="sum",
        )
        return loss, count

    def batch_loss(self, model: nn.Module, batch: Any) -> tuple[torch.Tensor, int]:
        """Return summed loss and denominator token/item count."""
        return self.loss_from_logits(self.forward_logits(model, batch), batch)

    def select_anchor_logits(
        self, logits: torch.Tensor, batch: Any, *, max_positions: int = 8
    ) -> torch.Tensor:
        """Select a small deterministic function-space signature from one batch.

        Full `[B,T,V]` logits are too large to retain for a 50M model. The default prefers
        supervised positions from `loss_mask` or non-ignored `labels`, then samples evenly across
        that ordered set. Concrete adapters may override this for a precommitted anchor contract.
        """
        if logits.ndim != 3:
            raise ValueError("anchor selection expects logits with shape [batch, sequence, vocab]")
        if (
            isinstance(max_positions, bool)
            or not isinstance(max_positions, int)
            or max_positions < 1
        ):
            raise ValueError("max_positions must be a positive integer")
        valid: torch.Tensor | None = None
        if isinstance(batch, Mapping):
            loss_mask = batch.get("loss_mask")
            labels = batch.get("labels")
            if isinstance(loss_mask, torch.Tensor) and loss_mask.shape == logits.shape[:2]:
                valid = loss_mask.to(device=logits.device, dtype=torch.bool)
            elif isinstance(labels, torch.Tensor) and labels.shape == logits.shape[:2]:
                # The generic LM contract compares logits at t with labels at t+1.
                valid = torch.zeros(logits.shape[:2], dtype=torch.bool, device=logits.device)
                valid[:, :-1] = labels[:, 1:].to(device=logits.device).ne(-100)
        if valid is None:
            valid = torch.ones(logits.shape[:2], dtype=torch.bool, device=logits.device)
        coordinates = valid.nonzero(as_tuple=False)
        if coordinates.numel() == 0:
            raise ValueError("anchor selection found no valid positions")
        count = min(max_positions, coordinates.shape[0])
        if count == coordinates.shape[0]:
            selected = coordinates
        else:
            offsets = torch.linspace(
                0, coordinates.shape[0] - 1, steps=count, device=coordinates.device
            ).round().to(torch.long)
            selected = coordinates.index_select(0, offsets)
        return logits[selected[:, 0], selected[:, 1]]

    def evaluate_batch(
        self,
        model: nn.Module,
        batch: Any,
        *,
        capture_anchors: bool = False,
        anchor_positions: int = 8,
    ) -> tuple[torch.Tensor, int, torch.Tensor | None]:
        """Evaluate once, optionally returning only a compact anchor-logit signature."""
        if not capture_anchors:
            loss, count = self.batch_loss(model, batch)
            return loss, count, None
        logits = self.forward_logits(model, batch)
        loss, count = self.loss_from_logits(logits, batch)
        anchors = self.select_anchor_logits(logits, batch, max_positions=anchor_positions)
        return loss, count, anchors

    def generation_prompts(self) -> list[Any]:
        return []

    def generate(self, model: nn.Module, prompt: Any, *, seed: int) -> Any:
        raise NotImplementedError("adapter does not expose generation")

    def export_evaluation_checkpoint(
        self,
        state: Mapping[str, torch.Tensor],
        *,
        template_checkpoint: str | Path,
        output: str | Path,
        candidate_id: str,
        provenance: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Export a candidate in the source project's read-only evaluation format."""
        del state, template_checkpoint, output, candidate_id, provenance
        raise NotImplementedError("adapter does not expose evaluation-checkpoint export")
