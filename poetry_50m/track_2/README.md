# GENOME Track 2 — executable research codebase

**GENOME** stands for **Generative Endpoint Neural Operator for Model Emission**.

This project turns the Track 2 research pack into a runnable implementation. Its first target is the fully trained Track 1 poetry model, called **R0**. The implementation deliberately separates:

1. **G0 representation:** Can `Delta-T = WT - W0` be encoded compactly while preserving R0's function?
2. **G1 compilation:** Can a model infer that genome from allowed early-training and dataset evidence?
3. **G2 transfer:** Can it do so for a hidden seed or data order whose endpoint it never saw?

## Implemented

The codebase contains:

- a concrete adapter for this repository's `poetry_50m/track_1` model, seeding, checkpoint, tokenizer, packed-data, and loss contracts;
- R0 completion preflight and a fail-closed prohibition on freezing partial endpoints;
- exact W0/WT config-hash, step-zero, and run-manifest lineage validation;
- final-snapshot and train-receipt hash reconciliation before WT is accepted;
- immutable W0/WT specimen freezing with repeated W0 hash verification;
- canonical tensor inventory, exact Track 1 tensor roles, layer indices, and tied-weight handling;
- Model Genome Program (MGP) v0.1 serialization and deterministic decoding;
- dense, int8, packed-int4, fixed/budgeted SVD, and low-rank-plus-sparse codecs;
- one-time SVD workspaces reused across the complete rank frontier, with shared-cost accounting;
- actual file-byte and shared-interpreter accounting;
- parameter, role, loss, perplexity, logit-KL, and top-k functional evaluation;
- Delta-T and singular-spectrum analysis;
- deterministic CountSketch gradient fingerprints from W0;
- trajectory features and code-space extrapolation;
- a role-conditioned neural block auto-decoder;
- a fixed-size probabilistic genome compiler baseline and model-life dataset;
- latent-code and full-weight repair baselines;
- report generation, a tiny end-to-end demonstration, and automated tests;
- an evaluation-only checkpoint bridge into Track 1's existing poetry generation suite;
- strict artifact loaders that reject unsupported versions, path traversal, symlinks, undeclared files, and hash mismatches.

## Install and validate now

R0 does not need to be finished before installing Track 2:

```bash
cd poetry_50m/track_2
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,analysis]'
pytest

genome demo --output artifacts/demo
```

The demo trains a deterministic tiny causal LM, freezes W0/WT, serializes multiple MGPs, destroys and reloads their in-memory representations, decodes them, and evaluates the resulting phenotypes.

## Connect the real Track 1 run

```bash
cp configs/poetry50m_track1.example.yaml configs/poetry50m_track1.yaml

genome track1-preflight \
  --config configs/poetry50m_track1.yaml \
  --output artifacts/preflight.json
```

While R0 is still training, the preflight report should show `ready_to_freeze: false` and the latest checkpoint's progress. Once the full endpoint reaches the configured `max_steps`:

```bash
genome freeze --config configs/poetry50m_track1.yaml

genome verify \
  --specimen artifacts/specimens/track1_R0 \
  --config configs/poetry50m_track1.yaml

genome analyze \
  --specimen artifacts/specimens/track1_R0 \
  --output artifacts/analysis/R0

genome rate-distortion \
  --specimen artifacts/specimens/track1_R0 \
  --config configs/poetry50m_track1.yaml \
  --output artifacts/rate_distortion/R0 \
  --ranks 0,1,2,4,8,16,32,64
```

See [`TRACK1_INTEGRATION.md`](TRACK1_INTEGRATION.md) for the exact artifact paths, checks, and post-R0 sequence.

## Repository layout

```text
genome/
  adapters/       Track 1 boundary and exact poetry50m adapter
  codecs/         transparent G0 codecs and reusable SVD workspaces
  mgp/            program schema, serializer, validator, interpreter
  neural/         block auto-decoder and endpoint compiler
  repair/         latent and full-weight repair baselines
  evaluator.py    Genome Gate functional evaluation
  fingerprint.py  model-native gradient evidence
  specimen.py     immutable R0 freezer
configs/          demo, generic, and exact poetry50m configurations
docs/             full theory, mathematics, experiment plan, and task cards
scripts/          one-purpose command wrappers
tests/            deterministic unit and vertical-slice tests
```

## Scientific guardrails

The authoritative rules are in `docs/track_2_genome/AGENTS.md`. In particular:

- never let a G1/G2 compiler read WT or endpoint-derived fitted codes;
- never fit against hidden verification data;
- never report weight MSE alone as success;
- count the MGP, shared decoder, candidate sampling, probing, and repair compute;
- do not call a fitted G0 genome a general optimizer;
- do not average raw coordinates across independent seeds without proven alignment;
- do not freeze an in-progress R0 checkpoint as the reference endpoint.

Track 1 asks whether comparable poetry quality can be reached in fewer training steps and less elapsed GPU time, without assuming in advance whether the answer is an optimizer, transport rule, controller, or hybrid. Track 2 retains that evidence standard rather than defining success as checkpoint compression alone.

## Status

The software foundation and exact Track 1 integration are ready. The real R0 rate–distortion, neural genome, and compiler results cannot be claimed until the full endpoint artifact exists and the commands above have been run. See [`IMPLEMENTATION_STATUS.md`](IMPLEMENTATION_STATUS.md) and [`VALIDATION.md`](VALIDATION.md).

## Reuse Track 1's poetry generation suite

After encoding or fitting a candidate MGP, export it into a full, explicitly
evaluation-only Track 1 checkpoint:

```bash
genome export-track1-checkpoint \
  --specimen artifacts/specimens/track1_R0 \
  --mgp artifacts/rate_distortion/R0/int4.mgp \
  --config configs/poetry50m_track1.yaml \
  --output artifacts/track1_eval_checkpoints/int4.pt
```

Track 1 can then run its unchanged prompt suite, generation manifest, metrics,
and blinded comparison tooling against that file. The exported checkpoint
contains `evaluation_only: true` and `resume_forbidden: true`; never use it as a
training resume source.
