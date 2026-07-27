from __future__ import annotations

import torch

from genome.codecs import SVDCodec
from genome.metrics import parameter_metrics
from genome.mgp.interpreter import decode_program


def test_full_rank_svd_approaches_target(tiny_artifacts):
    specimen = tiny_artifacts["specimen"]
    base = specimen.load_base()
    target = specimen.load_target()
    max_rank = max(min(spec.shape) for spec in specimen.inventory if len(spec.shape) == 2)
    program = SVDCodec(
        rank=max_rank,
        factor_dtype=torch.float32,
        vector_bits=32,
        candidate_id="full_rank_svd",
    ).fit(base, target, specimen.inventory, tied_groups=specimen.tied_groups)
    decoded = decode_program(
        program,
        base,
        specimen.inventory,
        tied_groups=specimen.tied_groups,
        verify_checksums=False,
    )
    metrics = parameter_metrics(decoded, target, specimen.inventory)
    assert metrics["relative_l2"] < 1e-5


def test_svd_workspace_reuses_factors(tiny_artifacts, monkeypatch):
    from genome.codecs import LowRankSparseCodec, SVDWorkspace

    specimen = tiny_artifacts["specimen"]
    base = specimen.load_base()
    target = specimen.load_target()
    workspace = SVDWorkspace.build(
        base,
        target,
        specimen.inventory,
        tied_groups=specimen.tied_groups,
    )
    from genome.tensor_inventory import tied_owner_map

    aliases = tied_owner_map(specimen.tied_groups)
    expected_matrices = sum(
        1
        for spec in specimen.inventory
        if len(spec.shape) == 2 and spec.name not in aliases
    )
    assert workspace.matrix_count == expected_matrices

    def forbidden_svd(*args, **kwargs):
        raise AssertionError("workspace-backed codecs recomputed an SVD")

    monkeypatch.setattr(torch.linalg, "svd", forbidden_svd)
    svd_program = SVDCodec(
        rank=2,
        workspace=workspace,
        candidate_id="workspace_svd",
    ).fit(base, target, specimen.inventory, tied_groups=specimen.tied_groups)
    sparse_program = LowRankSparseCodec(
        rank=2,
        sparse_fraction=0.01,
        workspace=workspace,
        candidate_id="workspace_sparse",
    ).fit(base, target, specimen.inventory, tied_groups=specimen.tied_groups)
    assert svd_program.records
    assert sparse_program.records
