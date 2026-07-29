# GENOME active experiment plan

This plan replaces the old decoder-first and Track 1-first plan.

## R0 — foundation integrity

Required:

- complete model-life schema tests;
- semantic fingerprint tests;
- structured Runtime primitive tests;
- compact target policy tests;
- deterministic program tokenization and inverse tests;
- variable compiler forward and backward tests;
- grammar-constrained generation tests;
- active CLI and import-boundary tests.

The failed PolyPythia V1-V4 code must not be an active import or command.

## R1 — public source audit

Before a large download, record:

- repository and immutable revision;
- licence;
- architecture and model sizes;
- true W0 status;
- WT status;
- tokenizer and dataset access;
- complete ordered recipe and source provenance;
- checkpoint availability;
- accepted role: complete, partial, endpoint-only, or pending verification;
- training, development, hidden, evaluation-only, or quarantined role.

Parameter-count storage values are estimates. Download receipts must later use actual HF/LFS byte
counts.

Track 1 remains evaluation-only. Revealed Pythia 14M seed9 remains quarantined from a new hidden
claim.

## R2 — compact target frontier

Fit transparent target candidates on known development lives at declared byte budgets.

For every candidate:

1. fit without a dense or exact residual;
2. serialize the full MGP;
3. audit manifest, container, payload, patch, index, scale, and shared bytes;
4. decode through the deterministic Runtime;
5. load the state into the real model;
6. run finite-logit and functional tests;
7. compare with W0 and WT;
8. record the acceptance or failure reason.

Report:

- actual target-specific bytes;
- actual shared bytes;
- validation loss and perplexity;
- logit KL and top-k agreement;
- task metrics;
- decode time;
- failure by tensor family.

A raw SVD approximation is not a successful compiler target.

## R2b — label identity and non-uniqueness

Repeat target fitting in one environment. Require identical program bytes and hashes.

Record the numerical environment. The current SVD sign rule does not resolve repeated singular
subspaces. Before coefficient-supervised compiler training, use Runtime or function loss, or add a
verified deterministic subspace rule.

## R2c — representation scalability

Tokenize realistic Pythia 14M and 31M targets. Compare the result with the compiler limit.

The current analytic flat-token estimates are:

| Model | 5% median-budget case | 10% upper-budget case | Limit |
|---|---:|---:|---:|
| Pythia 14M | 44,268 | 88,230 | 4,096 |
| Pythia 31M | 95,602 | 190,898 | 4,096 |

This is a blocker. Implement a hierarchical skeleton plus bounded coefficient packets. Do not
raise the flat context to an extreme size. Do not truncate coefficients.

## R3 — tiny compiler smoke

After several development targets pass R2:

```text
life evidence
  -> grammar-constrained compiler output
  -> MGP materialization
  -> deterministic Runtime
  -> real executable model
```

The compiler must not receive WT at inference. The generated program must pass the same serialized
byte and functional gates as its teacher targets.

The loss can include token, numeric, Runtime, and functional terms. A rate proxy is permitted only
when it depends on predictions and is calibrated against serialized bytes.

## R4 — hidden same-family transfer

Freeze one complete hidden life. Keep WT, endpoint hashes, fitted programs, and later checkpoints
unavailable.

Sequence:

1. compile one MGP;
2. validate grammar and policy;
3. serialize and seal the MGP;
4. decode and seal the Runtime output;
5. run a finite model smoke;
6. reveal WT;
7. evaluate W0, prediction, and WT with the same protocol.

Require meaningful improvement over W0. Report failures without repair as the primary result.

## R5 — transfer order

Proceed only after R4 passes:

1. hidden seed or data order;
2. hidden model size;
3. hidden dataset;
4. hidden recipe;
5. hidden architecture family;
6. Track 1 evaluation.

## Stop rules

Do not train the production compiler when:

- no development life passes the compact functional gate;
- actual serialized bytes exceed policy;
- target identity is unstable;
- coefficient sequences exceed the compiler representation;
- sources are incomplete or not verified;
- the hidden protocol is not frozen.

Do not launch paid compute from this validation plan.
