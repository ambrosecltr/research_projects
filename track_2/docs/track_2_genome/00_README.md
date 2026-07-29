# GENOME Track 2 active design

This file describes the recovery design. It replaces the original decoder-first design pack.
Read `../../RECOVERY.md` and `../../AGENTS.md` before you change code.

## Project definition

GENOME tests this path:

```text
true random W0
+ variable architecture graph
+ semantic corpus and W0-response evidence
+ complete ordered training recipe
    -> one learned GENOME Compiler
    -> compact executable MGP
    -> deterministic MGP Runtime
    -> runnable trained model
```

The active path has one learned model: the GENOME Compiler.

The Runtime executes transparent mathematical primitives. It does not require a learned weight
decoder. A learned decoder can return only after a measured experiment shows a better functional
rate-distortion result after all shared and target-specific bytes are counted.

## Scientific boundary

For one complete model life:

\[
W_T = \mathcal A(D,\mathcal G,W_0,r,e).
\]

The compiler receives an endpoint-free view:

\[
p = C_\phi(\Phi(D,W_0),\mathcal G,W_0,r).
\]

The deterministic Runtime produces:

\[
\widehat W = \operatorname{Runtime}(p,W_0,\mathcal G).
\]

The compiler must not receive WT, endpoint hashes, fitted endpoint programs, or later checkpoints
from the hidden life.

## Required evidence

A compiler input can include:

- true W0;
- a variable architecture graph;
- tokenizer structure;
- corpus statistics from real content;
- W0 gradient and activation summaries;
- the full ordered training recipe;
- explicit optimizer and schedule values.

Cryptographic hashes and repository revisions are provenance. They are not semantic model inputs.

## MGP boundary

An active MGP can use deterministic primitives such as:

```text
BASE_COPY
LOW_RANK
SHARED_BASIS
KRONECKER
SPECTRAL_DCT
CODEBOOK_BLOCKS
LOW_RANK_PATCH
SPARSE_PATCH
COPY_FROM_TIED
```

An active compiler target must not contain:

- a full dense Delta-T;
- a full fp16 or int8 residual;
- one code value for each weight;
- target-specific decoder weights;
- an exact-residual escape path.

The normal target-specific limit is 10% of direct fp16 Delta-T. A result from 10% to 25% is
exploratory only. The audit uses real serialized MGP bytes before a candidate becomes training
supervision.

## Required gates

1. Validate complete model-life records and whole-life splits.
2. Audit public sources, licences, revisions, recipe access, and storage estimates.
3. Build deterministic semantic fingerprints.
4. Fit compact target candidates.
5. Serialize each candidate and repeat the byte audit.
6. Decode each candidate through the Runtime.
7. Load the result into the real model.
8. Require improvement over W0 and a predeclared functional gate.
9. Train the compiler only after several development lives pass.
10. Seal a hidden prediction before WT reveal.

Weight MSE is diagnostic. A raw SVD fit is not a successful target until it passes the Runtime,
model-load, functional, and byte gates.

## Track 1 boundary

Track 1 is legacy G0 compatibility and future evaluation only. It is not part of the current
public-life compiler-training split.

## Historical files

`../../POLYPYTHIA_ROUND1_RESULTS.md` is immutable negative evidence. The failed V1-V4 code remains
only for reproduction. Use `scripts/legacy_polypythia_v4.py`; do not use it for a new experiment.

The other documents in this directory are pre-recovery research notes unless their first section
states that they are active. They do not override `RECOVERY.md`.
