from __future__ import annotations

from pathlib import Path

from genome.config import load_config, resolve_config_path


def test_configured_paths_resolve_from_yaml_directory(tmp_path: Path) -> None:
    config_dir = tmp_path / "project" / "configs"
    config_dir.mkdir(parents=True)
    path = config_dir / "run.yaml"
    path.write_text("value: ok\n", encoding="utf-8")
    config = load_config(path)
    assert resolve_config_path(config, "../artifacts/R0") == (
        tmp_path / "project" / "artifacts" / "R0"
    ).resolve()
    absolute = (tmp_path / "elsewhere").resolve()
    assert resolve_config_path(config, absolute) == absolute


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    path = tmp_path / "duplicate.yaml"
    path.write_text("value: one\nvalue: two\n", encoding="utf-8")
    try:
        load_config(path)
    except ValueError as error:
        assert "duplicate YAML key" in str(error)
    else:
        raise AssertionError("duplicate YAML key was accepted")


def test_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    from genome.io import read_json

    path = tmp_path / "duplicate.json"
    path.write_text('{"value": 1, "value": 2}\n', encoding="utf-8")
    try:
        read_json(path)
    except ValueError as error:
        assert "duplicate JSON key" in str(error)
    else:
        raise AssertionError("duplicate JSON key was accepted")
