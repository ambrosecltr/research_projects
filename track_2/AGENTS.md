# Mandatory GENOME operating rules

## Mission

Train one GENOME Compiler to map an endpoint-free model-life description to a compact executable program that produces a trained model.

```text
W0 + architecture + semantic data/W0 evidence + recipe -> compact MGP -> deterministic Runtime
```

## Non-negotiable rules

1. The compiler must never receive hidden WT, endpoint hashes, accepted endpoint programs, later hidden checkpoints, hidden evaluation outputs, or any transform from which WT can be recovered.
2. The active path contains one learned model: `GenomeCompiler`.
3. Do not add a learned decoder, hypernetwork Runtime, diffusion decoder, residual predictor, or second AI model.
4. Do not add dense Delta-T, full residual, exact residual, one code per weight, one output per weight, or a block code as wide as the block.
5. Actual serialized target-specific MGP bytes—not logical tensor estimates—determine the rate gate.
6. A target MGP is not supervision until the decoded real model passes both the byte and functional gates.
7. Weight MSE is diagnostic only. The model must execute and beat W0.
8. Split by complete model life. Checkpoints from one life inherit its split.
9. Pythia 14M seed9 is a training life. Its W0 and WT may be used to fit a compact target and train the compiler.
10. Pythia 31M seed9 WT remains unresolved and unavailable until a prediction and Runtime output are sealed.
11. Cryptographic hashes are provenance, not semantic model inputs.
12. Do not add more model families or sizes before the fresh 31M hidden gate.
13. Report one-shot generation before any repair. Repair must be a separate result.
14. Failed gates are results. Never bypass them by increasing payload size or relaxing thresholds after seeing hidden data.

## Program policy

Primary target-specific budget:

```text
MGP bytes <= 10% of direct fp16 Delta-T bytes
```

Programs between 10% and 20% are exploratory and cannot train the production compiler without an explicit new decision record. Programs above 20% are rejected.

Matrices may use compact formula primitives only. Vectors may use bounded int8 storage. Sparse patches are capped at 0.1% of model values. Shared assets must be frozen from training lives, separately counted, and contain no per-life endpoint lookup.

## Functional target policy

A development target is accepted only when:

```text
finite model
candidate loss < W0 loss
endpoint progress >= 0.80
actual target-specific bytes <= 10% of fp16 Delta-T
```

The exact evaluation corpus and thresholds must be frozen before hidden reveal.

## Compiler policy

The compiler may use:

- architecture graph and tensor roles;
- W0 tensor statistics and coordinate features;
- content-derived corpus sketches;
- W0 loss, gradient sketches and activation summaries;
- tokenizer statistics;
- complete numeric training recipe.

The compiler output is hierarchical:

- primitive and rank decisions per tensor;
- bounded factor/vector packets from shared heads;
- deterministic MGP materialization.

It may not use a fixed output table tied to one checkpoint, architecture, seed, or tensor index count.

## RunPod policy

Use a new network volume and do not copy any pre-existing GENOME workspace. The source directory becomes immutable after receipts are written. Do not download all intermediate checkpoints. Initial source materialization is W0 and WT for training/development lives, and W0 only for the hidden life.

## Required stopping conditions

Stop rather than improvise when:

- true W0 cannot be established;
- source identity cannot be pinned;
- semantic evidence cannot be generated;
- adapter round trip changes model function;
- no compact target beats W0;
- no development target reaches the declared gate;
- compiler outputs exceed the byte policy;
- hidden WT becomes visible before sealing;
- training produces NaN/Inf or invalid programs;
- checkpoint resume is not reliable.
