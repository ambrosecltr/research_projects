# GENOME implementation status

## Active direction

Track 2 is being recovered as a **compact executable program compiler**.

```text
true W0 + variable architecture graph + semantic dataset/W0 evidence + complete staged recipe
    -> one learned GENOME Compiler
    -> compact MGP
    -> deterministic MGP Runtime
    -> runnable model
```

The PolyPythia V4 decoder/compiler remains a recorded failed experiment. It is not the active architecture.

## What was already trustworthy

- Canonical W0/WT tensor inventories and tied-weight handling.
- Deterministic MGP serialization, integrity validation and decoding.
- Exact known-endpoint Track 1 round trips.
- Dense, quantized, SVD and sparse rate-distortion baselines.
- GPT-NeoX native/canonical conversion and real-model round-trip tests.
- Immutable Hugging Face revisions, LFS hashes and download receipts.
- Whole-life train/development/hidden split mechanics.
- Hidden endpoint prediction seals and reveal controls.
- Wikitext and LM Evaluation Harness evaluation.

These establish representation and evaluation mechanics. They do not establish endpoint prediction.

## PolyPythia Round One: preserved negative evidence

Round One used Pythia 14M seeds0–7 for training, seed8 for development and seed9 as a sealed hidden life.

V1–V3 failed to represent functional endpoints compactly. V4 then stored one fp16 residual for every position in every 16x16 block:

```text
true normalized Delta-T block - learned decoder prediction
```

Its target-specific MGP was approximately 28.45 MB, while direct fp16 Delta-T was approximately 28.14 MB, before counting the approximately 4.61 MB learned decoder. V4 therefore did not demonstrate a compact genome language.

The hidden compiler result was substantially worse than W0:

- W0 Wikitext loss: `11.072410`;
- predicted loss: `78.254531`;
- true WT loss: `5.253024`;
- relative parameter error: `1.144849`;
- top-1 and top-5 agreement: `0`.

The strict hidden protocol worked. The predictive architecture failed. See `POLYPYTHIA_ROUND1_RESULTS.md` for immutable evidence.

## Recovery foundation implemented

### Complete model lives

`genome/life_schema.py` adds `GENOME_MODEL_LIFE` v0.3.0:

- true random initialization;
- tokenizer and vocabulary;
- dataset records and semantic fingerprints;
- ordered pretraining/SFT/DPO/RL/etc. stages;
- checkpoint trajectory;
- available or sealed final endpoint;
- endpoint-free compiler evidence;
- complete/partial/endpoint-only classification;
- whole-lineage split validation and split commitments.

The compiler view excludes WT, fitted programs, trajectory, evaluations, hashes, paths and run identity.

### Semantic evidence

`genome/semantic_fingerprint.py` provides deterministic bounded evidence from actual content and W0 response:

- token unigram and bigram CountSketches;
- byte frequencies;
- sequence-length and supervision statistics;
- per-role W0 gradient sketches and moments;
- W0 activation moments and quantiles.

SHA-256 and repository revisions remain provenance only and are never concatenated into semantic model features.

### Compact target policy

`genome/mgp/policy.py` rejects:

- dense large-tensor Delta-T;
- full/exact residuals;
- residual block mode;
- one neural code value per weight;
- excessive direct values or sparse exceptions;
- target programs outside their declared byte budget;
- target programs not smaller than direct fp16 Delta-T.

The primary target-specific budget defaults to 10% of direct fp16 Delta-T. Up to 25% is an explicitly exploratory band.

### Deterministic structured MGP Runtime

The Runtime now supports:

- `LOW_RANK`;
- `KRONECKER`;
- `SPECTRAL_DCT`;
- `SHARED_BASIS`;
- `CODEBOOK_BLOCKS`;
- bounded low-rank and sparse patches;
- tied copies.

These are executable formulas and require no learned interpreter.

### Canonical compact target fitting

`genome/compact_targets.py` implements:

- canonical SVD sign conventions;
- global singular-component allocation by energy per byte;
- no dense matrix residual;
- optional small int8 vectors under aggregate policy limits;
- compiler-target policy auditing.

A fitted MGP is not accepted until the functional Genome Gate executes it and confirms the predeclared quality band.

### Variable program compiler

`genome/program_compiler.py` implements one learned compiler with:

- variable training-stage tokens;
- variable logical tensor tokens;
- role, shape and architecture features;
- tensor-graph message passing;
- bidirectional conditioning attention;
- autoregressive MGP token and coefficient generation;
- no fixed output head tied to one model's block count.

`genome/program_tokens.py` provides deterministic tokenization and inverse reconstruction for the first canonical low-rank target language. It rejects dense fallback payloads.

The flat coefficient-token representation is a production blocker. Analytic 5% budget estimates
are 44,268 tokens for Pythia 14M and 95,602 for Pythia 31M, compared with the current 4,096-token
limit. Production work needs a hierarchical skeleton plus bounded coefficient packets.

### Regression coverage added

New tests cover:

- complete multi-stage life validation and hidden WT exclusion;
- whole-lineage split leakage;
- semantic corpus, gradient and activation fingerprints;
- Kronecker, DCT, shared-basis and codebook Runtime execution;
- compact target fitting and canonical SVD signs;
- rejection of V4-style per-weight residuals and dense target labels;
- MGP program tokenization/inverse reconstruction;
- variable compiler forward, backward and bounded generation.

## Validation state

Validation runs locally with Python 3.11 because GitHub Actions are unavailable. The active CLI
does not import or expose V1-V4 commands. The full result is in `VALIDATION_REPORT.md`. No
production training or large download has been launched from the recovery branch.

## Next implementation gates

### R0 — green recovery foundation

- Existing tests pass unchanged.
- New recovery tests pass.
- V4 residual targets are rejected.
- A toy compact target round-trips through target tokens and Runtime.
- The variable compiler performs a finite forward/backward/generation smoke.

### R1 — public source audit

Commit the exact source matrix before downloading:

- repository/revision/licence;
- architecture and sizes;
- true W0 and WT status;
- checkpoint count;
- dataset/tokenizer/exact-order/recipe/provenance access;
- complete/partial/endpoint-only role;
- storage estimates;
- frozen whole-life split.

### R2 — functional compact target frontier

For development lives, evaluate actual serialized MGP bytes against:

- validation loss and perplexity;
- task metrics;
- logit KL and top-k agreement;
- tensor-family failure;
- decode time.

Do not train the compiler until multiple development lives pass both byte and functional gates.

The current raw-SVD fitter has no Pythia functional proof. Its sign convention also does not make
repeated singular subspaces unique. Use Runtime or function loss, or add a verified deterministic
subspace rule before coefficient-supervised training.

### R3 — tiny compiler smoke

Prove the complete path without WT access at inference:

```text
life evidence -> program tokens -> MGP -> Runtime -> executable model
```

### R4 — hidden same-family transfer

Seal the MGP and Runtime output before revealing WT. The first hidden candidate must meaningfully beat W0. Report:

```text
(L(W0) - L(predicted)) / (L(W0) - L(WT))
```

Cross-size, cross-dataset and cross-family work remains blocked until same-family hidden transfer succeeds.

## Track 1 boundary

The old 50M Track 1 integration remains useful for deterministic G0 compatibility. The current 8M life must be recorded from true random pretraining W0 through pretraining and SFT to final WT. The pretrain-to-SFT segment may remain an auxiliary stage-local record, but it cannot replace the full life.

Track 1 WT remains outside public-life compiler training and is evaluated last.
