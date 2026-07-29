# GENOME Track 2

**GENOME** means **Generative Endpoint Neural Operator for Model Emission**.

Track 2 asks whether a learned compiler can replace most of a model's training trajectory with a genuinely compact executable program:

```text
true random W0
+ variable architecture graph
+ semantic dataset and W0-response evidence
+ complete staged training recipe
    -> GENOME Compiler
    -> compact Model Genome Program (MGP)
    -> deterministic MGP Runtime
    -> runnable trained model
```

The compiler must never see the final weights of the life on which it is tested.

## Recovery status

The PolyPythia Round One hidden protocol was honest, but its V4 representation was not a compact genome. It stored one fp16 residual value for every weight position and was larger than direct fp16 Delta-T before counting its learned decoder. Its hidden prediction was far worse than W0.

Round One is preserved as negative evidence in `POLYPYTHIA_ROUND1_RESULTS.md`. It is not the active architecture.

Read [`RECOVERY.md`](RECOVERY.md) first.

## Active architecture

There are two active components:

1. **GENOME Compiler** — one learned variable-architecture model that emits an MGP token stream and numeric coefficient chunks.
2. **MGP Runtime** — deterministic code that executes the program against W0.

A separate learned decoder is not part of the active path. It may return only after a rate-distortion experiment proves that shared-decoder bytes plus target-specific bytes improve matched functional quality.

## New recovery foundation

### Complete model lives

`genome.life_schema` defines strict multi-stage lives from true W0 through every ordered stage to WT. It supports pretraining, continued pretraining, SFT, DPO, RL, RLVR, distillation and other explicit stages. Hidden lives expose W0 but not WT or fitted endpoint programs.

The split unit is the complete life, never one checkpoint.

### Semantic evidence

`genome.semantic_fingerprint` builds compiler inputs from actual content and model response:

- token unigram and bigram CountSketches;
- byte frequencies;
- sequence-length and supervision statistics;
- per-role W0 gradient sketches and moments;
- W0 activation moments and quantiles.

Repository revisions and SHA-256 values remain provenance only. Cryptographic hashes are never semantic model features.

### Compact compiler-target policy

`genome.mgp.policy` rejects:

- dense large-tensor Delta-T;
- full residual payloads;
- residual block mode;
- one generated or stored code value per weight;
- excessive sparse exceptions;
- target programs outside the byte budget;
- programs not smaller than direct fp16 Delta-T.

The primary target-specific budget is 10% of direct fp16 Delta-T. Up to 25% is an explicitly labelled exploratory band.

### Deterministic structured MGPs

The Runtime supports:

```text
BASE_COPY
LOW_RANK
KRONECKER
SPECTRAL_DCT
SHARED_BASIS
CODEBOOK_BLOCKS
LOW_RANK_PATCH
SPARSE_PATCH
COPY_FROM_TIED
```

No learned interpreter is required for these formulas.

### Canonical compact labels

`genome.compact_targets` fits a transparent first compiler language using globally budgeted
canonical low-rank factors. It has no dense matrix residual. A fit is only a candidate. It becomes
compiler supervision after post-serialization byte audit and the functional Genome Gate.

### Variable program compiler

`genome.program_compiler.VariableProgramCompiler`:

- supports variable numbers of stages and logical tensors;
- uses tensor roles, shapes and graph connectivity;
- applies graph message passing and bidirectional condition encoding;
- emits a variable program sequence rather than a model-sized block output head;
- scales with program description length rather than parameter count.

`genome.program_tokens` provides deterministic target tokenization and inverse reconstruction. The first vertical slice handles canonical low-rank MGPs. New primitives require a deterministic inverse and tests; there is no dense fallback token.

The current flat coefficient-token format is not production-scalable. At the 5% median-budget
case, analytic estimates are 44,268 tokens for Pythia 14M and 95,602 for Pythia 31M. A
hierarchical skeleton plus bounded coefficient packets is required before production training.

## Preserved infrastructure

The following remain useful:

- deterministic MGP serialization and decoding;
- exact known-endpoint Track 1 round trips;
- Hugging Face revision/LFS/download provenance;
- GPT-NeoX native/canonical conversion;
- hidden prediction sealing and reveal controls;
- Wikitext and LM Evaluation Harness evaluation;
- transparent SVD, quantization and spectral diagnostics.

The old V4 decoder/compiler modules have no active package exports or normal CLI commands. They
remain only for reproduction through `scripts/legacy_polypythia_v4.py`.

## Install and test

```bash
cd track_2
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,evaluation]'
python -m pip check
python -m compileall -q genome tests
python -m pytest -q
```

## Experiment order

1. Validate complete multi-stage model-life manifests.
2. Audit public sources and storage before downloading.
3. Freeze whole-life train/development/hidden splits.
4. Prove all source adapters round-trip.
5. Build semantic corpus and W0-response fingerprints.
6. Fit compact canonical MGPs for known lives.
7. Execute them and construct the functional rate-distortion frontier.
8. Train the compiler only after target labels pass.
9. Run a tiny complete compiler-to-Runtime smoke test.
10. Test a hidden same-family life and require meaningful improvement over W0.
11. Attempt size, dataset, recipe and architecture transfer in that order.
12. Evaluate the Track 1 poetry model last.

No production training or large download begins before the source matrix, split commitment,
compact target frontier, leakage model, hierarchical program representation, smoke test, storage
estimate, and hidden acceptance protocol are committed.

## Scientific claims

Keep these separate:

- **Proven:** the deterministic MGP machinery can reproduce known endpoints when supplied a sufficient program.
- **Failed:** PolyPythia V4 did not learn a compact genome language or transfer to hidden seed9.
- **Next gate:** compact target programs preserve useful endpoint function at the declared byte budget.
- **Compiler gate:** one-shot hidden compilation meaningfully beats W0 without WT, repair or dense residuals.
- **Later:** transfer across sizes, datasets, recipes, architecture families and finally Track 1.

Parameter error is diagnostic. A generated model succeeds only through actual execution and functional evaluation.
