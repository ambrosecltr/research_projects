from __future__ import annotations

from pathlib import Path

import yaml
from typer.testing import CliRunner

from examples.tiny_track1 import TinyTrack1Adapter, train_reference
from genome.cli import app


def test_freeze_resolves_specimen_paths_from_config_file(tmp_path: Path) -> None:
    project = tmp_path / "track_2"
    config_dir = project / "configs"
    source = project / "source" / "R0.pt"
    config_dir.mkdir(parents=True)
    train_reference(source, adapter=TinyTrack1Adapter(), updates=1)
    config = config_dir / "track2.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "adapter": {
                    "project_root": str(Path(__file__).resolve().parents[1]),
                    "factory": "examples.tiny_track1:create_adapter",
                    "kwargs": {},
                },
                "specimen": {
                    "id": "tiny-path-test",
                    "output": "../artifacts/specimen",
                    "final_checkpoint": "../source/R0.pt",
                },
            }
        ),
        encoding="utf-8",
    )
    result = CliRunner().invoke(app, ["freeze", "--config", str(config)])
    assert result.exit_code == 0, result.output
    assert (project / "artifacts/specimen/manifest.json").is_file()
