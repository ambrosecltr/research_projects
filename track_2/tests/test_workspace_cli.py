from __future__ import annotations

from typer.testing import CliRunner

from genome.cli import app
from genome.workspace import initialize_workspace


def test_workspace_is_clean_forward_only(tmp_path) -> None:
    manifest = initialize_workspace(tmp_path / "genome")
    assert manifest["rules"]["fresh_isolated_volume"] is True
    assert (tmp_path / "genome" / "source" / "hf").is_dir()
    assert (tmp_path / "genome" / "compiler" / "checkpoints").is_dir()


def test_cli_help_contains_only_forward_commands() -> None:
    result = CliRunner().invoke(app, ["--help"])
    assert result.exit_code == 0
    for forbidden in ("autodecoder", "train-decoder", "refine-latent"):
        assert forbidden not in result.stdout.lower()
    assert "train-compiler" in result.stdout
    assert "fit-compact-target" in result.stdout
    assert "canonicalize-life" in result.stdout
