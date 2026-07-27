from __future__ import annotations

import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from .bit_accounting import account_mgp
from .codecs import LowRankSparseCodec, QuantizedDeltaCodec, SVDCodec, SVDWorkspace
from .evaluator import GenomeGate
from .hashing import sha256_file
from .io import (
    replace_directory_atomic,
    temporary_directory,
    write_json,
)
from .mgp.serializer import save_program
from .specimen import FrozenSpecimen


@dataclass(frozen=True)
class RateDistortionPoint:
    family: str
    label: str
    rank: int | None = None
    bits: int | None = None
    sparse_fraction: float = 0.0
    factor_dtype: str = "float32"


def default_rate_distortion_points() -> list[RateDistortionPoint]:
    return [
        RateDistortionPoint("quantized", "int8", bits=8),
        RateDistortionPoint("quantized", "int4", bits=4),
        *[RateDistortionPoint("svd", f"svd_r{rank}", rank=rank) for rank in (0, 1, 2, 4, 8, 16, 32)],
        *[
            RateDistortionPoint("svd_sparse", f"svd_r{rank}_sp1e3", rank=rank, sparse_fraction=0.001)
            for rank in (2, 4, 8, 16)
        ],
    ]


def run_rate_distortion(
    specimen: FrozenSpecimen,
    gate: GenomeGate,
    *,
    output_dir: str | Path,
    points: Sequence[RateDistortionPoint] | None = None,
    manifest_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    destination = Path(output_dir).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"rate-distortion output already exists: {destination}")
    output = temporary_directory(destination.parent, f".{destination.name}.building.")
    try:
        required_metadata = {
            "architecture_manifest_sha256": specimen.manifest["contract_hashes"][
                "architecture"
            ],
            "tensor_inventory_sha256": specimen.manifest["contract_hashes"][
                "tensor_inventory"
            ],
            "base_state_sha256": specimen.manifest["state_hashes"]["W0"],
            "research_level": "G0",
        }
        effective_metadata = dict(manifest_metadata or {})
        for key, expected in required_metadata.items():
            actual = effective_metadata.get(key, expected)
            if actual != expected:
                raise ValueError(f"rate-distortion metadata mismatch for {key}")
            effective_metadata[key] = expected

        base = specimen.load_base()
        target = specimen.load_target()
        selected_points = list(points or default_rate_distortion_points())
        if not selected_points:
            raise ValueError("rate-distortion sweep requires at least one point")
        labels = [point.label for point in selected_points]
        if any(not label for label in labels) or len(labels) != len(set(labels)):
            raise ValueError("rate-distortion point labels must be non-empty and unique")
        svd_labels = [
            point.label
            for point in selected_points
            if point.family in {"svd", "svd_sparse"}
        ]
        workspace = (
            SVDWorkspace.build(
                base,
                target,
                specimen.inventory,
                tied_groups=specimen.tied_groups,
            )
            if svd_labels
            else None
        )
        context = {
            "format_version": 1,
            "specimen_id": specimen.specimen_id,
            "specimen_manifest_sha256": sha256_file(specimen.root / "manifest.json"),
            "state_hashes": dict(specimen.manifest["state_hashes"]),
            "points": [point.__dict__ for point in selected_points],
            "shared_svd_workspace": {
                "built": workspace is not None,
                "matrix_count": 0 if workspace is None else workspace.matrix_count,
                "factorization_seconds": (
                    0.0 if workspace is None else workspace.factorization_seconds
                ),
                "reused_by_candidates": svd_labels,
                "accounting_policy": "charge_once_across_the_frontier",
            },
        }
        write_json(output / "rate_distortion_context.json", context)

        results: list[dict[str, Any]] = []
        for point in selected_points:
            if point.family == "quantized":
                codec = QuantizedDeltaCodec(point.bits or 8, candidate_id=point.label)
            elif point.family == "svd":
                codec = SVDCodec(
                    rank=point.rank,
                    factor_dtype=getattr(torch, point.factor_dtype),
                    vector_bits=8,
                    candidate_id=point.label,
                    workspace=workspace,
                )
            elif point.family == "svd_sparse":
                codec = LowRankSparseCodec(
                    rank=point.rank or 0,
                    sparse_fraction=point.sparse_fraction,
                    factor_dtype=getattr(torch, point.factor_dtype),
                    candidate_id=point.label,
                    workspace=workspace,
                )
            else:
                raise ValueError(f"unknown rate-distortion family: {point.family}")
            fit_start = time.perf_counter()
            program = codec.fit(
                base,
                target,
                specimen.inventory,
                tied_groups=specimen.tied_groups,
                manifest_metadata=effective_metadata,
            )
            fit_seconds = time.perf_counter() - fit_start
            mgp_path = output / f"{point.label}.mgp"
            save_program(program, mgp_path)
            report = gate.evaluate_mgp(mgp_path).to_dict()
            report["compute"]["fit_seconds_excluding_shared_svd"] = fit_seconds
            if point.family in {"svd", "svd_sparse"} and workspace is not None:
                report["compute"]["shared_svd_factorization_seconds"] = (
                    workspace.factorization_seconds
                )
                report["compute"][
                    "shared_svd_cost_policy"
                ] = "charge_once_across_the_frontier"
            report["rate_distortion_point"] = point.__dict__
            report["bit_accounting"] = account_mgp(
                mgp_path, base_path=specimen.base_path
            )
            write_json(mgp_path / "evaluation.json", report)
            results.append(report)
        write_json(output / "rate_distortion_results.json", results)
        replace_directory_atomic(output, destination)
        return results
    except BaseException:
        shutil.rmtree(output, ignore_errors=True)
        raise
