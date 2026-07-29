# GENOME recovery validation report

## Decision

overall: **FAIL**

The cleanup and local correctness validation pass. Production Pythia compiler readiness fails.
The current flat coefficient-token format exceeds the 4,096-token limit by a large amount. The
repository also has no approved public source matrix and no real Pythia compact target that passed
the functional Genome Gate.

validated source commit:

  77bdfa9cd9163f4f657bcbcba3980d148a4300aa

validation branch:

  agent/genome-recovery-validation

validation branch commit:

  3df5ebe6aa8f5f7de371fec92f87cb65e12e2f9d

The RunPod implementation agent must use this exact branch and validation branch commit. It must not
start from `main` or from the uncleaned recovery source commit.

## Local validation result

All required local commands pass in Python 3.11.15 on macOS arm64:

| Command | Result | Wall time |
|---|---|---:|
| `pip install -e '.[dev,evaluation]'` | PASS | 2.01 s |
| `pip check` | PASS | 0.23 s |
| `compileall -q genome tests` | PASS | 0.05 s |
| `pytest -q` | PASS: 86 passed, 1 skipped, 5 warnings | 5.11 s |
| `python -m genome --help` | PASS | 0.98 s |
| deterministic demo | PASS | 1.63 s |
| focused readiness tests | PASS: 35 passed, 5 warnings | 2.29 s |
| `git diff --check` | PASS | 0.02 s |

The only skip is
`tests/legacy/test_polypythia_round1.py::test_neural_runtime_returns_cuda_decoder_output_to_cpu_base`.
It is an explicit CUDA-only test in the archived V4 suite. This machine has no CUDA device.

The five warnings are the same informational PyTorch Transformer nested-tensor warning. No
required CPU test is skipped.

Raw output is under `validation/`. `validation/environment.txt` contains the Python version,
platform, Torch/CUDA state, and full `pip freeze`.

## Active CLI

The active top-level commands are:

- `validate-life`
- `audit-source`
- `track1-preflight` — legacy G0 or future evaluation-only
- `freeze` — evaluation specimen only
- `verify`
- `analyze`
- `encode` — transparent G0 baseline
- `fit-compact-target`
- `audit-program-tokens`
- `decode`
- `evaluate`
- `export-track1-checkpoint` — legacy G0 or future evaluation-only
- `architecture-graph`
- `fingerprint`
- `rate-distortion`
- `report`
- `demo`
- `polypythia`

The active `polypythia` subcommands are:

- `plan-sources`
- `materialize-sources`
- `prepare-evaluation-texts`
- `evaluate-revealed`
- `evaluate-lm-harness`

The failed V4 training commands are available only through
`scripts/legacy_polypythia_v4.py`. Its help text identifies it as an archived failed experiment.

## Files moved, removed, and rewritten

Moved:

- the old `genome/polypythia/cli.py` implementation to
  `genome/legacy/polypythia_v4/cli.py`;
- learned-decoder evaluation injection to `genome/legacy/polypythia_v4/evaluate.py`;
- five V1-V4 test modules to `tests/legacy/`.

Removed:

- `genome/repair/latent.py`;
- `scripts/fit_autodecoder.py`;
- `scripts/refine_genome.py`;
- stale, unenforced `SHA256SUMS.txt`.

Rewritten:

- active top-level and PolyPythia CLIs;
- compact target serialization and byte policy;
- compiler loss and generation boundary;
- source audit and revealed-life quarantine;
- program scalability estimator and regression tests;
- active recovery documents and archive notices.

The complete file map is in `validation/cleanup-map.md`.
`POLYPYTHIA_ROUND1_RESULTS.md` is unchanged.

## Legacy paths that remain

The V1-V4 modules remain under `genome/neural/` for historical reproduction. Moving the full
dependency graph would cause a large reproduction-only rewrite. Each module has an archive notice.
`genome/neural/__init__.py` exports nothing.

The deterministic Runtime can still read the historical `NEURAL_BLOCK_FIELD` opcode. This is an
old-artifact compatibility path. Active commands do not load a learned interpreter. The
compiler-target policy rejects residual mode, per-weight code width, and oversized neural-code
payloads.

Two shared-decoder evaluation functions remain in `genome/polypythia/evaluate.py` because they
share sealed/revealed endpoint controls with useful evaluation code. They import no neural module
and require an injected loader. Only the archival adapter supplies that loader.

## Forbidden-path audit

The AST import graph contains:

- 59 active modules;
- 492 import edges;
- zero active edge to `genome.neural`;
- zero active edge to `genome.legacy`.

Importing `genome.cli` loads zero neural or legacy module.

The active CLI grep has zero match for:

  fit-neural
  refine-latent
  train-decoder
  fit-development-code
  train-compiler
  predict-hidden
  --neural

The active import grep has zero neural or legacy import match. The per-weight grep finds only
rejection code and documentation in the active path. It finds no active emitter.

Exact output is in `validation/import-boundary.log` and
`validation/forbidden-path-grep.log`.

## Deterministic target identity

PASS in one environment.

`created_unix` is fixed at `0.0` in canonical compiler targets. Repeated fitting from the same W0,
WT, inventory, configuration, and random-free code produces the same manifest hash, payload file
hash, and complete artifact-directory hash.

Cross-hardware SVD identity is not certified. Sign canonicalization removes only the simple sign
ambiguity. Repeated or nearly repeated singular-value subspaces can still rotate or reorder across
hardware and linear-algebra libraries. Production training must use Runtime/function loss or a
tested canonical subspace rule.

## Serialized byte result

PASS for the local policy boundary.

An in-memory target audit is marked `estimate` and cannot become training supervision. After
serialization, the target is audited again with actual MGP file sizes. The measured total includes
the canonical manifest, Safetensors container overhead, payload tensors, patch tensors, indices,
scales, and embedded shared payloads.

A regression test proves that container and manifest overhead can turn a logical-budget pass into
a serialized-budget failure. A second test checks actual MGP bytes against every executable MGP
file. `compiler_target_audit.json` is a report sidecar and is not part of the executable MGP.

Passing this byte policy does not mean the target is functionally valid.

## Pythia program-length result

The estimator uses 16 fp16 coefficient values per flat numeric token, 76 tensor records, and a
4,096-token limit.

| Model | Exact parameters | 5% median-budget case | 10% upper-budget case | Limit |
|---|---:|---:|---:|---:|
| Pythia 14M | 14,067,712 | 44,268 | 88,230 | 4,096 |
| Pythia 31M | 30,494,720 | 95,602 | 190,898 | 4,096 |

The 5% case is an analytic median-budget case. It is not an empirical median from fitted targets.
The exact parameter counts were checked by locally instantiating the untied official GPT-NeoX
configurations. The source configurations are the official
[Pythia 14M config](https://huggingface.co/EleutherAI/pythia-14m/blob/main/config.json) and
[Pythia 31M config](https://huggingface.co/EleutherAI/pythia-31m/blob/main/config.json).

This is a hard readiness failure. The implementation must use an autoregressive program skeleton
and bounded per-tensor coefficient packets from shared primitive-specific heads. It must not
truncate coefficients or raise the flat context to tens of thousands of tokens.

## Other deep checks

- V4 residual mode, a code value per weight, and dense matrix targets are rejected.
- The constant teacher-length rate loss is removed. A future rate objective must depend on
  predictions and must be calibrated against serialized bytes.
- The unconstrained `VariableProgramCompiler.generate` method is removed. Production generation
  uses the grammar state machine and the deterministic inverse.
- Impossible rank, negative vector scale, premature EOS, trailing tokens, and missing tensor
  records are rejected before an MGP is returned.
- SVD fitting produces only a target candidate. Documentation and CLI output do not call it valid
  training supervision before Runtime, model, functional, and byte gates pass.
- A complete source now requires true W0, WT, dataset content, exact data order, tokenizer,
  complete recipe, and provenance.
- Storage values are labelled estimates until actual HF/LFS receipts exist.
- Track 1 cannot enter an active public-life compiler split.
- Revealed Pythia 14M seed9 must remain in the quarantine split.

## Required work before production training

1. Replace flat coefficient tokens with a hierarchical skeleton and bounded coefficient packets.
2. Commit and validate the complete public model-life source matrix and frozen whole-life split.
3. Fit development targets and pass deterministic Runtime, real-model, functional, and actual-byte
   gates.
4. Resolve SVD label non-uniqueness through Runtime/function loss or a tested canonical subspace
   rule.
5. Add a calibrated prediction-dependent rate proxy before claiming rate-regularized training.

The detailed implementation contract is in `validation/blockers.md`.

## Compute and endpoint statement

This validation created no RunPod resource, downloaded no Pythia endpoint, fitted no real model
program, trained no production compiler, and revealed no fresh hidden endpoint.
