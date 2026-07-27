from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from examples.tiny_track1 import TinyTrack1Adapter, train_reference
from genome.specimen import freeze_specimen


@pytest.fixture(scope="session")
def tiny_artifacts(tmp_path_factory: pytest.TempPathFactory):
    root = tmp_path_factory.mktemp("tiny_genome")
    adapter = TinyTrack1Adapter()
    checkpoint = root / "R0.pt"
    train_reference(checkpoint, adapter=adapter, updates=16)
    specimen = freeze_specimen(
        adapter,
        output_dir=root / "specimen",
        specimen_id="tiny_R0_test",
        final_checkpoint=checkpoint,
        source_metadata={"test": True},
    )
    return {"root": root, "adapter": adapter, "specimen": specimen}
