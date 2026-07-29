# GENOME recovery: compact programs, not disguised checkpoints

## Decision

The PolyPythia Round One V4 branch is preserved as a valid failed prediction experiment, but it is no longer the active GENOME architecture.

V4 used a learned block predictor and then stored:

```text
true Delta-T block - learned prediction
```

as a 256-value fp16 residual for every 16x16 block. A block contains 256 positions, so the residual stored one value per weight. Its target-specific MGP was larger than direct fp16 Delta-T before counting the shared decoder. The known-endpoint reconstruction therefore succeeded by construction, while the hidden compiler prediction was dramatically worse than W0.

That result established useful negative evidence:

- the proposed low-dimensional global/layer/tensor codes were insufficient;
- raw block MSE did not preserve model function;
- hash-derived data-order vectors were not meaningful semantic evidence;
- eight lives from one 14M architecture/corpus/recipe were not enough to predict a model-sized endpoint;
- exact raw cross-seed coordinate prediction is an unnecessarily hostile target;
- hidden sealing and evaluation mechanics worked.

## What remains trustworthy

Keep and reuse:

- deterministic MGP serialization and Runtime;
- Track 1 exact known-endpoint round trips;
- source revisions, LFS hashes, download receipts and licence records;
- GPT-NeoX native/canonical conversion and tests;
- complete-life split enforcement and hidden reveal controls;
- Wikitext and LM Evaluation Harness evaluation;
- SVD, quantization and spectral diagnostics;
- the full Round One report as failed-experiment evidence.

The old V4 modules may remain in the repository for reproducibility, but no active command, runbook or agent instruction may present them as the production path.

## Correct system

```text
                         semantic corpus evidence
                                   |
true random W0 -- architecture graph -- complete staged recipe
                                   |
                                   v
                     Variable GENOME Compiler
                                   |
                  compact executable MGP tokens
                                   |
                                   v
                       deterministic MGP Runtime
                                   |
                                   v
                         runnable final model
```

There is one learned model in the active path: the compiler. The Runtime is deterministic.

A separate learned decoder may be reconsidered only if a measured experiment shows that:

```text
shared decoder bytes + per-target code bytes
```

beats transparent program primitives at matched functional quality. It cannot be introduced merely to make reconstruction easier.

## New foundation in this recovery branch

### Complete model lives

`genome.life_schema` defines `GENOME_MODEL_LIFE` version `0.3.0` with:

- immutable source artifacts;
- true initialization;
- tokenizer;
- datasets and semantic fingerprints;
- ordered pretraining/SFT/DPO/RL/etc. stages;
- checkpoints and trajectory;
- final endpoint or sealed hidden endpoint;
- endpoint-free compiler evidence;
- whole-life split commitments.

The compiler view intentionally removes WT, fitted programs, trajectory, evaluations, hashes and paths.

### Semantic evidence

`genome.semantic_fingerprint` creates model inputs from actual content and W0 response:

- token unigram CountSketch;
- token bigram CountSketch;
- byte frequencies;
- sequence-length distribution;
- supervised-token proportion;
- per-role W0 gradient CountSketch and moments;
- activation moments and quantiles.

Cryptographic hashes remain in provenance manifests and are explicitly excluded from the model input tensor order.

### Compact program policy

`genome.mgp.policy` rejects:

- dense large-tensor deltas;
- excessive direct quantized values;
- exact residual declarations;
- residual block mode;
- one neural code value per weight;
- oversized sparse patches;
- target MGPs that exceed the declared byte fraction;
- target MGPs not smaller than direct fp16 Delta-T.

The primary default is 10% target-specific bytes. Up to 25% may be explored but is labelled exploratory, not success.

### Deterministic program primitives

The Runtime now supports:

- low-rank factors;
- Kronecker sums;
- sparse orthonormal DCT modes;
- shared bases;
- block codebooks;
- bounded sparse and low-rank patches;
- tied-tensor copies.

These are executable formulas. None requires a learned interpreter.

### Canonical compact labels

`genome.compact_targets` fits deterministic low-rank target programs with:

- global byte allocation by singular energy per byte;
- canonical SVD sign conventions;
- no dense matrix residual;
- policy auditing before a program can become compiler supervision.

A fitted target is still not accepted until the Genome Gate executes it and verifies functional quality.

### Variable compiler

`genome.program_compiler.VariableProgramCompiler`:

- accepts variable numbers of training stages and tensors;
- represents logical tensor roles and graph connectivity;
- uses graph message passing and bidirectional condition encoding;
- autoregressively emits an MGP token stream and coefficient chunks;
- has no fixed output head for one architecture's blocks;
- has output length proportional to program description length rather than child parameter count.

`genome.program_tokens` provides a deterministic tokenization/inverse for canonical compact targets. The first vertical slice supports canonical low-rank programs. Additional primitives must be added with a deterministic inverse and tests; there is no fallback dense token.

## Recovery gates

### R0 — foundation integrity

Pass when:

- old tests still pass;
- new model-life, fingerprint, policy, primitive, tokenizer and compiler tests pass;
- V4 residual MGPs are rejected as compiler targets;
- a toy compact MGP round-trips through tokens and Runtime;
- a variable compiler forward/backward/generation smoke passes.

### R1 — public source audit

Before large downloads, commit a matrix containing:

- repository and exact revision;
- licence;
- architecture and sizes;
- true W0 status;
- WT status;
- checkpoint count;
- dataset and tokenizer access;
- complete recipe availability;
- download/canonical/derived storage;
- accepted role: complete, partial or endpoint-only;
- train/development/hidden assignment.

### R2 — compact target language

Fit programs for development lives at explicit byte budgets. At every point report:

- actual serialized target bytes;
- shared bytes;
- validation loss and perplexity;
- logit KL and top-k agreement;
- task metrics;
- decode time;
- failure by tensor family.

Do not train the compiler until multiple development lives pass the compact functional gate.

### R3 — tiny compiler smoke

Train on a tiny controlled set and prove:

```text
life evidence -> program tokens -> MGP -> Runtime -> executable model
```

without WT access at inference and without dense residuals.

### R4 — hidden same-family transfer

Freeze a hidden whole life. Seal the predicted MGP and Runtime output before revealing WT. The first result must meaningfully beat W0.

Report progress:

```text
(L(W0) - L(predicted)) / (L(W0) - L(WT))
```

No size/family transfer work begins until this gate succeeds.

### R5 — transfer and Track 1

Proceed in order:

1. hidden seed/data order;
2. hidden size;
3. hidden dataset;
4. hidden recipe;
5. hidden architecture family;
6. Track 1 poetry life.

Track 1 is evaluation-only during public model-life training.

## No-paid-compute checklist

Do not launch production training until all are committed:

- source audit and storage estimate;
- complete model-life schema and manifests;
- whole-life split commitment;
- semantic fingerprint contract;
- native/canonical round-trip tests;
- compact target grammar and byte policy;
- development rate-distortion frontier;
- variable compiler configuration;
- leakage threat model;
- tiny end-to-end smoke result;
- hidden protocol and acceptance band;
- resumable training/checkpoint plan.

The objective is not to improve V4. The objective is to compile a genuinely compact model program.
