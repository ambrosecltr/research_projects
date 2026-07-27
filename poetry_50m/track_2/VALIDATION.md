# GENOME v0.2.0 validation record

Validation date: **27 July 2026**

This record describes checks performed on the packaged codebase. It is evidence that the implementation executes as designed on controlled fixtures and that the exact `poetry50m` integration boundary is implemented. It is **not** evidence that direct endpoint compilation has already succeeded on the real full R0 model.

## Environment used

- Python 3.13 in the artifact build environment
- CPU execution for the automated suite and reference demonstration
- PyTorch, NumPy, safetensors, PyYAML, and Typer available
- The real Track 1 runtime dependency `tokenizers` was not installed in the artifact container; it is declared by the package and is already required by Track 1.

The package declares Python 3.11 or newer and avoids Python-3.13-only syntax.

## Automated test suite

Command:

```bash
PYTHONPATH=. pytest -q
```

Result:

```text
............................                                             [100%]
28 passed in 1.83s
```

Covered behaviours include:

- immutable W0/WT specimen creation and verification;
- specimen and MGP rejection of path traversal, symlinks, undeclared files, and malformed integrity manifests;
- repeated W0 construction with identical state hashes;
- endpoint- and base-lineage validation evidence stored in the specimen;
- final snapshot, run manifest, and train receipt hash reconciliation;
- source-import origin verification against the configured Track 1 checkout;
- config-relative path resolution for freeze/preflight artifacts;
- compact anchor-logit capture rather than retaining full `[B,T,V]` outputs;
- tied-weight discovery and restoration;
- dense MGP exact round-trip at the declared target dtype;
- fail-closed corruption detection;
- true packed-int4 encode/decode;
- int8 and int4 candidate validity;
- full-rank SVD convergence;
- one-time SVD factorization reuse across multiple rank/sparse candidates;
- atomic rate–distortion publication and cleanup after a candidate failure;
- neural block interpreter fitting on a controlled tensor;
- genome compiler output dimensions;
- compiler artifact training/save/load;
- compiler/interpreter path traversal and manifest-integrity rejection;
- deterministic CountSketch fingerprints;
- architecture graph construction;
- source-level split isolation;
- canonical byte accounting that excludes unrelated report files;
- exact Track 1 semantic roles for fused QKV, attention output, fused SwiGLU input, MLP output, scales, and residual rates;
- poetry50m W0 seed replay;
- trainer-checkpoint parsing;
- rejection of a partial endpoint and acceptance of a completed endpoint;
- W0/WT model-config, training-config, step-zero, and run-manifest checks;
- equality between the compact-anchor evaluator loss and Track 1 model loss;
- duplicate YAML/JSON key rejection;
- evaluation-only Track 1 checkpoint export and round-trip verification.

## Python compilation and CLI discovery

Commands:

```bash
PYTHONPATH=. python -m compileall -q genome examples scripts tests
PYTHONPATH=. python -m genome --help
```

Both completed successfully. The CLI exposes:

```text
track1-preflight, freeze, verify, analyze, encode, fit-neural,
decode, evaluate, export-track1-checkpoint, architecture-graph, fingerprint,
rate-distortion, refine-latent, report, demo
```

The optional `ruff` and `mypy` executables were not installed in the artifact runtime, so their commands were not represented as completed checks. The concrete tokenizer/packed-artifact path can only be exercised once this package is placed beside the user's Track 1 checkout, whose active training environment contains those dependencies and artifacts.

## Complete executable reference run

Command:

```bash
PYTHONPATH=. python -m genome demo \
  --output /tmp/genome_demo_validation_v0_2 \
  --updates 20 \
  --neural
```

The demonstration:

1. builds and trains a deterministic tiny causal language model;
2. freezes W0 and trained WT;
3. records repeated W0 hash agreement and endpoint eligibility;
4. exports architecture, inventory, ties, corpus, tokenizer, split, recipe, environment, and integrity contracts;
5. emits dense, int8, packed-int4, SVD, SVD-plus-sparse, and neural MGP candidates;
6. reloads every artifact from disk;
7. decodes every candidate from W0 plus only its declared payload;
8. executes the reconstructed child model;
9. evaluates parameter and functional distortion;
10. counts actual serialized bytes;
11. writes machine-readable and Markdown results.

Observed summary:

| Candidate | MGP artifact bytes | Loss gap from WT | Relative parameter L2 | Development decision |
|---|---:|---:|---:|---|
| dense | 99,779 | 0.000000 | 0.000000 | PASS |
| int8 | 27,872 | -0.000214 | 0.001090 | PASS |
| packed int4 | 22,479 | -0.007984 | 0.019841 | PASS |
| SVD rank 4 | 32,000 | 0.075453 | 0.114110 | REVIEW |
| SVD rank 4 + sparse patch | 37,505 | 0.065010 | 0.108241 | REVIEW |
| neural block genome v0 | 18,773 | 0.723149 | 0.204587 | REVIEW |

Negative int8/int4 loss gaps are fixture-level outcomes, not claims that quantization improves R0. The neural candidate is an early executable baseline and did not pass the short-budget fixture gate.

## Exact Track 1 integration implemented

The concrete adapter was checked against the repository interfaces inspected at commit:

```text
44092a5d05015de4d526e063809e8d4804c500f6
```

The integration uses:

- `poetry50m.model.DecoderOnlyTransformer`;
- `poetry50m.model.ModelConfig.from_mapping`;
- `poetry50m.training.engine.seed_everything` before model construction;
- trainer checkpoint model state under `model`;
- `poetry50m.weights.v1` snapshot state under `state_dict`;
- `poetry50m.data.artifacts.read_packed_sequences`;
- checkpoint and snapshot model/training hashes plus `run.manifest.json` identity;
- final checkpoint/snapshot hashes and run-manifest hash from `train.receipt.json`;
- per-pack `input_ids[:-1]`, targets, and `loss_mask` evaluation;
- the model's returned mean loss and `token_count` for exact weighted aggregation.

The adapter also checks the declared GPT/nGPT parameter counts, verifies that the imported package came from the configured checkout, reconciles completion artifacts, records model/config/source-file hashes, and can export a decoded phenotype back into Track 1's evaluation-only checkpoint format.

The transparent frontier builds one exact float32 SVD workspace over all unique
two-dimensional Delta-T tensors, reuses it across every requested rank and
low-rank-plus-sparse candidate, and records the shared factorization cost once.
The frontier is written to a temporary sibling directory and atomically
published only after every candidate and report succeeds.

## What remains unvalidated because R0 is still completing

- freeze and hash the real completed endpoint;
- compare exact W0 replay with the real initial snapshot;
- reproduce the real validation/test loss from canonical WT;
- inspect the actual tensor inventory and tied groups;
- run the real rate–distortion frontier;
- fit and evaluate the first neural genome on R0;
- perform compile-and-polish repair measurements;
- train G1 on independent model lives;
- run sealed G2 transfer against an unseen data order or seed;
- measure matched-quality wall-clock acceleration.

No result in this validation record should be presented as a G1, G2, general-optimizer, or RSI result.
