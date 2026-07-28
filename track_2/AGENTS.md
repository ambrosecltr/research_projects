# GENOME codebase agent instructions

Read these before editing code:

1. `docs/track_2_genome/AGENTS.md`
2. `docs/track_2_genome/00_README.md`
3. `IMPLEMENTATION_STATUS.md`
4. `TRACK1_INTEGRATION.md`
5. `CODEMAP.md`

The research pack remains authoritative. This repository is the executable implementation of its first phases.

## Immediate mission

Use the concrete `genome.adapters.poetry50m.Poetry50MAdapter`, preflight the active Track 1 run, freeze the fully trained endpoint as R0 only after completion, reproduce its evaluation, and run the transparent G0 rate–distortion audit. Do not replace the working MGP/codec foundation with a larger neural model before those measurements.

## Rules specific to this codebase

- Keep all Track 1 imports behind `genome.adapters.Track1Adapter`.
- Never hard-code R0 values, target tensors, validation records, or checkpoint bytes into Python files or tests.
- Never let `decode_program()` read WT. Its only model-state input is W0.
- Preserve canonical tensor indices after R0 is frozen.
- MGP artifacts are immutable. Create a new candidate ID/path instead of overwriting one.
- Rate–distortion output directories are immutable and atomically published. Never resume into or overwrite a final frontier directory.
- Keep `safetensors`/JSON as canonical artifact formats. Source Track 1 checkpoints may use their existing loader.
- Reject WT unless its trainer checkpoint, final snapshot, run manifest, and train receipt all agree.
- Files from `export-track1-checkpoint` are evaluation-only. Never pass one to Track 1 `train --resume`.
- Run `pytest` after every change to specimen, MGP, codec, tied-weight, accounting, or decoder code.
- A functional result requires execution through the Track 1 adapter. Parameter error alone is diagnostic.
- Report MGP bytes, interpreter bytes, W0/base bytes, fitting time, decoding time, verification time, and repair time separately.
- Build one SVD workspace per frozen specimen/frontier. Charge its factorization cost once, and preserve `rate_distortion_context.json` with every report.
- Do not include `evaluation.json`, plots, or reports in the target-specific MGP byte count.
- Hidden-run endpoints and hidden verifier records must never enter compiler training inputs.

## First real-repository task

Copy `configs/poetry50m_track1.example.yaml` to a local untracked configuration and run:

```bash
genome track1-preflight --config configs/poetry50m_track1.yaml --require-ready
```

While R0 is active this command exits non-zero; stop there. Once it succeeds, run:

```bash
genome freeze --config configs/poetry50m_track1.yaml
genome verify --specimen artifacts/specimens/track1_R0 --config configs/poetry50m_track1.yaml
```

Do not run codecs until both commands pass and the W0-to-WT improvement is reproduced. Never disable `require_complete_endpoint` for the real R0.
