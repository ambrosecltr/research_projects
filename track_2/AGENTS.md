# GENOME codebase agent instructions

This file is authoritative for Track 2. The PolyPythia V4 residual experiment is a recorded failed branch, not the active architecture.

Read before editing:

1. `RECOVERY.md`
2. `docs/track_2_genome/00_README.md`
3. `docs/track_2_genome/01_THEORY_AND_MATH.md`
4. `docs/track_2_genome/05_EXPERIMENT_PLAN.md`
5. `README.md`
6. `POLYPYTHIA_ROUND1_RESULTS.md` as negative evidence only

## Active mission

Build GENOME as:

```text
true W0 + architecture graph + semantic dataset/W0 evidence + complete recipe
    -> one learned GENOME Compiler
    -> genuinely compact executable MGP
    -> deterministic MGP Runtime
    -> runnable trained model
```

The compiler must never see the final weights of the life on which it is tested. The output must scale with formulas and coefficients, not with the child model's parameter count.

## Components

1. **MGP Runtime** is deterministic code and expands a program against W0.
2. **GENOME Compiler** is the learned variable-architecture program generator.
3. A separate learned decoder is not part of the active design. It may return only after a measured experiment proves that shared-decoder bytes plus target-specific bytes improve the functional rate-distortion frontier.

## Forbidden shortcuts

A compiler target or prediction must not contain:

- full dense Delta-T;
- a full fp16/int8 residual;
- one stored or generated value per weight;
- block-code width equal to all positions in the block;
- target-specific decoder parameters;
- endpoint hashes, fitted endpoint codes, or later hidden checkpoints as compiler inputs;
- cryptographic hashes used as semantic dataset vectors;
- raw cross-seed endpoint averaging without verified alignment or a function-space objective.

There is no exact-residual escape hatch. A life that cannot pass the compact functional gate is labelled not representable by the current grammar.

## Compact target gate

Before compiler training, every fitted target MGP must:

1. serialize deterministically;
2. be smaller than direct fp16 Delta-T;
3. ordinarily use at most 10% of direct fp16 Delta-T target-specific bytes;
4. use at most 25% only as an explicitly labelled exploratory result;
5. execute through the deterministic Runtime;
6. satisfy the predeclared functional quality band.

Weight MSE alone is never success. Count shared assets separately and amortize them honestly.

## Model-life rules

- The split unit is one complete life, never one checkpoint.
- A complete life begins at true random W0 and contains every ordered training stage through final WT.
- Pretraining, SFT, DPO, RL and other stages are one ordered recipe when they belong to the same organism.
- Complete, partial and endpoint-only sources must be labelled explicitly.
- Hidden WT and later hidden checkpoints remain unavailable until the MGP and Runtime output are sealed.
- The Track 1 poetry life remains evaluation-only for public-life compiler training.

Use `genome.life_schema.ModelLifeManifest` for new records. The old flat model-life record and PolyPythia Round One corpus remain legacy compatibility artifacts.

## Semantic evidence

Repository revisions and SHA-256 values are provenance only. Compiler evidence must be calculated from content and model response, including applicable subsets of:

- token/byte/ngram and sequence statistics;
- deterministic corpus samples;
- W0 losses;
- per-role W0 gradient sketches;
- W0 activation summaries;
- objective mixture and tokenizer statistics;
- explicit numeric optimizer, schedule and stage features.

The cost of evidence generation is part of child-generation compute.

## Program language

The active grammar may use deterministic combinations of:

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

Small directly stored vectors are permitted only under the strict aggregate direct-value budget. Dense matrix deltas are baselines, not compiler labels.

## Compiler architecture

Use variable stage, layer and logical-tensor tokens with graph connectivity and bidirectional attention/message passing. Do not use a fixed embedding table or output head tied to one model's tensor/block count.

The active implementation is `genome.program_compiler.VariableProgramCompiler`. It emits a variable MGP token stream and numeric coefficient chunks. `genome.program_tokens` supplies the deterministic target tokenizer. Extend that tokenizer and its inverse together; never add an opaque fallback payload.

## Experiment order

1. Correct and validate complete model-life records.
2. Freeze whole-life train/development/hidden splits.
3. Audit source licences, revisions, W0/WT status and storage before downloads.
4. Prove every native/canonical adapter round trip.
5. Build reproducible semantic fingerprints.
6. Fit canonical compact target MGPs on known lives.
7. Execute and functionally gate those targets.
8. Train the variable program compiler only after target labels pass.
9. Run a tiny complete smoke pipeline.
10. Test a hidden same-family life and require meaningful improvement over W0.
11. Attempt size/dataset/recipe/family transfer only after same-family hidden success.
12. Evaluate Track 1 last.

## Paid-compute boundary

Do not start a production run or large download until the source matrix, split commitment, fingerprint contract, compact target frontier, compiler configuration, leakage threat model, smoke test and storage/compute estimate are committed and reviewed.

## Required result reporting

Every experiment records:

- complete life and split identities;
- code/config/environment hashes;
- compiler inputs and forbidden-input audit;
- target-specific, shared and base bytes;
- fit, compile, decode, evidence and evaluation compute;
- validation loss/perplexity and task metrics;
- logit/function metrics;
- W0, predicted and true-WT comparisons;
- one-shot and repaired results separately;
- explicit acceptance or failure reason.

A failed experiment is preserved. It is not silently upgraded with a dense residual and renamed a success.
