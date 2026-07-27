"""Deterministic conditional poem generation and sealed-snapshot loading."""

from poetry50m.inference.generation import (
    GenerationConfig,
    GenerationResult,
    build_conditioning_tokens,
    generate,
    load_snapshot_into_model,
)
from poetry50m.inference.manifest import (
    GenerationRecord,
    load_generation_records,
    run_generation_manifest,
    save_generation_records,
)

__all__ = [
    "GenerationConfig",
    "GenerationRecord",
    "GenerationResult",
    "build_conditioning_tokens",
    "generate",
    "load_generation_records",
    "load_snapshot_into_model",
    "run_generation_manifest",
    "save_generation_records",
]
