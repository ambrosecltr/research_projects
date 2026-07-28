"""Source identities separate coordinate semantics from offline analysis code."""

from __future__ import annotations

from pathlib import Path

import pytest

from poetry50m.config import RunPolicy, coordinate_source_hash, tree_hash


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_tree_hash_ignores_metadata_and_caches_but_tracks_python(tmp_path: Path) -> None:
    _write(tmp_path / "module.py", "VALUE = 1\n")
    baseline = tree_hash(tmp_path)
    _write(tmp_path / ".DS_Store", "finder")
    _write(tmp_path / ".hidden.py", "VALUE = 99\n")
    _write(tmp_path / "__pycache__" / "poison.py", "VALUE = 99\n")
    assert tree_hash(tmp_path) == baseline
    _write(tmp_path / "module.py", "VALUE = 2\n")
    assert tree_hash(tmp_path) != baseline


def _coordinate_tree(root: Path) -> None:
    for relative in (
        "model/transformer.py",
        "training/config.py",
        "training/engine.py",
        "training/stream.py",
        "data/batch_stream.py",
        "workflows/training.py",
    ):
        _write(root / relative, f"# {relative}\n")


def test_analysis_sources_do_not_change_coordinate_lineage(tmp_path: Path) -> None:
    _coordinate_tree(tmp_path)
    baseline = coordinate_source_hash(tmp_path)
    _write(tmp_path / "trajectory" / "endpoint_geometry.py", "METHOD = 'new'\n")
    _write(tmp_path / "workflows" / "trajectory.py", "METHOD = 'new'\n")
    _write(tmp_path / "workflows" / "reporting.py", "METHOD = 'new'\n")
    assert coordinate_source_hash(tmp_path) == baseline


@pytest.mark.parametrize(
    "relative",
    (
        "model/transformer.py",
        "training/config.py",
        "training/engine.py",
        "training/stream.py",
        "data/batch_stream.py",
        "workflows/training.py",
    ),
)
def test_coordinate_sources_change_training_lineage(tmp_path: Path, relative: str) -> None:
    _coordinate_tree(tmp_path)
    baseline = coordinate_source_hash(tmp_path)
    _write(tmp_path / relative, "CHANGED = True\n")
    assert coordinate_source_hash(tmp_path) != baseline


def test_run_policy_rejects_boolean_format_version(tmp_path: Path) -> None:
    path = tmp_path / "policy.json"
    _write(
        path,
        """
{
  "format_version": true,
  "trajectory_config_sha256": "0000000000000000000000000000000000000000000000000000000000000000",
  "verification": {
    "fixed_heldout_batches": 1,
    "anchor_positions_per_batch": 1,
    "fixed_probe_batches": 1,
    "probe_steps": 1,
    "optimizer_policy": "retain"
  }
}
""".strip(),
    )
    with pytest.raises(TypeError, match="format_version"):
        RunPolicy.load(path)
