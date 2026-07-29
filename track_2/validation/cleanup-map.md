# GENOME recovery cleanup map

## Active command path

`genome/cli.py` no longer imports the failed neural V1-V4 stack. It no longer exposes
`fit-neural`, `refine-latent`, interpreter options, or `demo --neural`.

The normal `polypythia` command now contains only source pinning, source materialization,
evaluation-text preparation, revealed-candidate evaluation, and LM Evaluation Harness evaluation.
It does not import the failed decoder or compiler.

New active commands perform real work:

- `validate-life`
- `audit-source`
- `fit-compact-target`
- `audit-program-tokens`

## Archived code

The former PolyPythia V4 CLI is now:

  genome/legacy/polypythia_v4/cli.py

Its learned-decoder evaluation adapters are now:

  genome/legacy/polypythia_v4/evaluate.py

The only command entry is the clearly named archival script:

  scripts/legacy_polypythia_v4.py

The V1-V4 implementation files remain under `genome/neural/`. Moving the full dependency graph
would create large reproduction-only changes. Each file now has a failed-experiment archive
notice. `genome/neural/__init__.py` exports nothing. Active modules have no static or runtime
import edge to this package.

Two learned-decoder evaluation functions remain in `genome/polypythia/evaluate.py` because they
share the sealed/revealed endpoint checks with current evaluation code. They no longer import a
decoder. The archival adapter injects the old decoder loader. No active command calls them.

`genome/mgp/interpreter.py` still reads the historical `NEURAL_BLOCK_FIELD` opcode so old MGPs can
be reproduced. Active decode commands do not load a learned interpreter. The compiler-target
policy rejects this opcode when its code width is per-weight, when it uses residual mode, or when
its payload exceeds the compact-code limit.

## Removed code and files

- Removed `genome/repair/latent.py`.
- Removed `scripts/fit_autodecoder.py`.
- Removed `scripts/refine_genome.py`.
- Retired `SHA256SUMS.txt`. It was stale and was not an enforced artifact. Immutable experiment
  manifests and result files keep their own hashes.

## Legacy tests

These tests moved under `tests/legacy/` and use the `legacy` marker:

- `test_polypythia_round1.py`
- `test_neural_autodecoder.py`
- `test_block_rate_distortion.py`
- `test_compiler_and_fingerprint.py`
- `test_compiler_training.py`

Track 1 compatibility tests remain. They use the `track1_evaluation` marker. Track 1 is legacy G0
or future evaluation-only. It is not an active compiler-training source.

## Rewritten active foundations

- `genome/compact_targets.py`: serializes every candidate and repeats the byte-policy audit with
  actual MGP file bytes.
- `genome/mgp/policy.py`: rejects residual mode, a code value per weight, dense matrix targets,
  large sparse patches, and over-budget serialized artifacts.
- `genome/program_compiler.py`: removes the constant teacher-length rate term and removes the
  unconstrained generator.
- `genome/program_grammar.py` and `genome/program_tokens.py`: remain the only production
  generation and inverse path.
- `genome/program_scalability.py`: reports flat-token capacity for Pythia 14M and 31M.
- `genome/source_audit.py`: requires true W0, WT, dataset content, exact data order, tokenizer,
  recipe, and provenance before a source is complete or approved. It labels storage values as
  estimates and quarantines revealed seed9.
- `genome/life_schema.py`: keeps hidden WT out of the compiler view and reports a clear validation
  error if a hidden endpoint is exposed.

## Documentation

`README.md`, `RECOVERY.md`, `IMPLEMENTATION_STATUS.md`, `CODEMAP.md`, and the active recovery
documents now define one learned model: the GENOME Compiler. The Runtime is deterministic.

The old Round One runbook and configuration are marked as archived. The immutable
`POLYPYTHIA_ROUND1_RESULTS.md` remains unchanged as negative evidence. Older design documents have
an archive warning and do not authorize the learned-interpreter path.
