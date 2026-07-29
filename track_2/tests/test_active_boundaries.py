from __future__ import annotations

import json
import subprocess
import sys

from typer.testing import CliRunner

from genome.cli import app


def test_active_cli_excludes_v4_production_commands() -> None:
    runner = CliRunner()
    top = runner.invoke(app, ["--help"])
    pythia = runner.invoke(app, ["polypythia", "--help"])
    demo = runner.invoke(app, ["demo", "--help"])
    assert top.exit_code == 0, top.output
    assert pythia.exit_code == 0, pythia.output
    assert demo.exit_code == 0, demo.output
    visible = f"{top.output}\n{pythia.output}\n{demo.output}"
    for forbidden in (
        "fit-neural",
        "refine-latent",
        "train-decoder",
        "fit-development-code",
        "train-compiler",
        "predict-hidden",
    ):
        assert forbidden not in visible
    assert "--neural" not in demo.output
    assert "fit-compact-target" in top.output
    assert "validate-life" in top.output
    assert "audit-source" in top.output


def test_active_package_import_does_not_load_v4_modules() -> None:
    script = """
import json
import sys
import genome
import genome.cli
print(json.dumps(sorted(name for name in sys.modules if name.startswith("genome.neural"))))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(result.stdout) == []
