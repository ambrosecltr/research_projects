from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import torch

from .adapters.base import Track1Adapter
from .bit_accounting import account_mgp
from .hashing import sha256_file
from .metrics import logits_kl, parameter_metrics, perplexity_from_mean_loss, topk_agreement
from .mgp.interpreter import decode_program
from .mgp.serializer import load_program
from .specimen import FrozenSpecimen, load_specimen
from .tensor_inventory import assert_tied_equal
from .types import EvaluationReport


@torch.inference_mode()
def evaluate_model_state(
    adapter: Track1Adapter,
    state: Mapping[str, torch.Tensor],
    *,
    split: str = "development",
    max_batches: int | None = None,
    device: str | torch.device = "cpu",
    capture_logits: bool = False,
    anchor_positions_per_batch: int = 8,
) -> dict[str, Any]:
    device_obj = torch.device(device)
    model = adapter.build_model().to(device_obj)
    missing, unexpected = model.load_state_dict(dict(state), strict=False)
    if missing or unexpected:
        raise ValueError(f"state/model mismatch; missing={missing}, unexpected={unexpected}")
    model.eval()
    total_loss = 0.0
    total_count = 0
    logits_parts: list[torch.Tensor] = []
    batch_count = 0
    start = time.perf_counter()
    for batch_index, raw_batch in enumerate(adapter.evaluation_batches(split, max_batches=max_batches)):
        if max_batches is not None and batch_index >= max_batches:
            break
        batch = adapter.move_batch(raw_batch, device_obj)
        loss, count, anchors = adapter.evaluate_batch(
            model,
            batch,
            capture_anchors=capture_logits,
            anchor_positions=anchor_positions_per_batch,
        )
        total_loss += float(loss.item())
        total_count += int(count)
        batch_count += 1
        if anchors is not None:
            logits_parts.append(anchors.detach().to(torch.float32).cpu())
    elapsed = time.perf_counter() - start
    if total_count == 0:
        raise ValueError(f"adapter produced no evaluable items for split {split!r}")
    mean_loss = total_loss / total_count
    result: dict[str, Any] = {
        "split": split,
        "batches": batch_count,
        "items": total_count,
        "loss_sum": total_loss,
        "mean_loss": mean_loss,
        "perplexity": perplexity_from_mean_loss(mean_loss),
        "seconds": elapsed,
    }
    if capture_logits:
        result["logits"] = logits_parts
    return result


def compare_functional_metrics(candidate: dict[str, Any], reference: dict[str, Any]) -> dict[str, Any]:
    result = {
        "candidate_mean_loss": candidate["mean_loss"],
        "reference_mean_loss": reference["mean_loss"],
        "loss_gap": candidate["mean_loss"] - reference["mean_loss"],
        "candidate_perplexity": candidate["perplexity"],
        "reference_perplexity": reference["perplexity"],
    }
    candidate_logits = candidate.get("logits", [])
    reference_logits = reference.get("logits", [])
    if candidate_logits and reference_logits:
        if len(candidate_logits) != len(reference_logits):
            raise ValueError("captured logit batch counts differ")
        kl_values = []
        top1_values = []
        top5_values = []
        for c, r in zip(candidate_logits, reference_logits, strict=True):
            kl_values.append(logits_kl(c, r))
            top1_values.append(topk_agreement(c, r, 1))
            top5_values.append(topk_agreement(c, r, min(5, c.shape[-1])))
        result.update(
            {
                "anchor_logit_kl": sum(kl_values) / len(kl_values),
                "top1_agreement": sum(top1_values) / len(top1_values),
                "top5_agreement": sum(top5_values) / len(top5_values),
            }
        )
    return result


class GenomeGate:
    def __init__(
        self,
        adapter: Track1Adapter,
        specimen: FrozenSpecimen | str | Path,
        *,
        split: str = "development",
        max_batches: int | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.adapter = adapter
        self.specimen = specimen if isinstance(specimen, FrozenSpecimen) else load_specimen(specimen)
        self.split = split
        self.max_batches = max_batches
        self.device = device
        self._base_state: dict[str, torch.Tensor] | None = None
        self._target_state: dict[str, torch.Tensor] | None = None
        self._reference_metrics: dict[str, Any] | None = None

    def _base(self) -> dict[str, torch.Tensor]:
        if self._base_state is None:
            self._base_state = self.specimen.load_base()
        return self._base_state

    def _target(self) -> dict[str, torch.Tensor]:
        if self._target_state is None:
            self._target_state = self.specimen.load_target()
        return self._target_state

    def _reference(self) -> dict[str, Any]:
        if self._reference_metrics is None:
            self._reference_metrics = evaluate_model_state(
                self.adapter,
                self._target(),
                split=self.split,
                max_batches=self.max_batches,
                device=self.device,
                capture_logits=True,
            )
        return self._reference_metrics

    def evaluate_state(
        self,
        candidate_id: str,
        state: Mapping[str, torch.Tensor],
        *,
        mgp_path: str | Path | None = None,
        decode_seconds: float = 0.0,
    ) -> EvaluationReport:
        target = self._target()
        assert_tied_equal(state, self.specimen.tied_groups)
        parameters = parameter_metrics(state, target, self.specimen.inventory)
        reference = self._reference()
        candidate = evaluate_model_state(
            self.adapter,
            state,
            split=self.split,
            max_batches=self.max_batches,
            device=self.device,
            capture_logits=True,
        )
        functional = compare_functional_metrics(candidate, reference)
        functional["candidate_eval_seconds"] = candidate["seconds"]
        functional["reference_eval_seconds"] = reference["seconds"]
        artifact_bytes = 0
        mgp_hash = None
        if mgp_path is not None:
            mgp_path_obj = Path(mgp_path).expanduser().resolve(strict=True)
            if mgp_path_obj.is_dir():
                accounting = account_mgp(mgp_path_obj)
                artifact_bytes = int(accounting["mgp_bytes"])
                # Hash the canonical manifest as the stable candidate identity.
                mgp_hash = sha256_file(mgp_path_obj / "manifest.json")
            else:
                artifact_bytes = mgp_path_obj.stat().st_size
                mgp_hash = sha256_file(mgp_path_obj)
        decision = "PASS" if functional["loss_gap"] <= 0.01 else "REVIEW"
        return EvaluationReport(
            candidate_id=candidate_id,
            mgp_sha256=mgp_hash,
            validity={"valid": True, "tied_weights": True},
            bytes={"mgp_artifact": artifact_bytes},
            compute={"decode_seconds": decode_seconds, "evaluation_seconds": candidate["seconds"]},
            parameter_metrics=parameters,
            functional_metrics=functional,
            generation_metrics={},
            decision=decision,
            failure_codes=(),
        )

    def evaluate_mgp(self, mgp_path: str | Path, *, interpreter: Any | None = None) -> EvaluationReport:
        program = load_program(mgp_path)
        base = self._base()
        start = time.perf_counter()
        state = decode_program(
            program,
            base,
            self.specimen.inventory,
            tied_groups=self.specimen.tied_groups,
            interpreter=interpreter,
            contract={
                "architecture_manifest_sha256": self.specimen.manifest["contract_hashes"]["architecture"],
                "tensor_inventory_sha256": self.specimen.manifest["contract_hashes"]["tensor_inventory"],
                "base_state_sha256": self.specimen.manifest["state_hashes"]["W0"],
            },
        )
        elapsed = time.perf_counter() - start
        return self.evaluate_state(
            program.manifest["candidate_id"], state, mgp_path=mgp_path, decode_seconds=elapsed
        )
