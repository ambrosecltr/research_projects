from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

from ..tensor_inventory import tied_owner_map
from ..types import TensorGenomeRecord, TensorSpec


def make_manifest(
    *,
    candidate_id: str,
    codec: str,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    source = dict(metadata or {})
    return {
        "format": "MGP",
        "version": "0.1.0",
        "project": "GENOME",
        "research_level": source.pop("research_level", "G0"),
        "candidate_id": candidate_id,
        "created_unix": time.time(),
        "codec": codec,
        **source,
    }


def make_records(
    specs: Sequence[TensorSpec],
    tied_groups: Sequence[Sequence[str]],
) -> tuple[list[TensorGenomeRecord], dict[str, str]]:
    owner_by_alias = tied_owner_map(tied_groups)
    records = []
    for spec in specs:
        records.append(
            TensorGenomeRecord(
                tensor_name=spec.name,
                canonical_index=spec.canonical_index,
                role=spec.role,
                layer_index=spec.layer_index,
                shape=spec.shape,
                output_dtype=spec.dtype,
                base_source="W0",
                components=[],
                tied_owner=owner_by_alias.get(spec.name),
            )
        )
    return records, owner_by_alias
