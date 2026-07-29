# GENOME Track 2

GENOME trains a model to generate trained models.

The active experiment is deliberately narrow and testable:

```text
true random W0
+ architecture graph
+ semantic evidence from the training corpus and W0
+ complete intended training recipe
    -> one learned GENOME Compiler
    -> compact executable Model Genome Program (MGP)
    -> deterministic Runtime
    -> runnable candidate endpoint
```

The compiler never receives the final weights of the life on which it is being tested.

## Clean project boundary

This directory is the authoritative Track 2 implementation. Only forward-use code, tests and documentation are included.

There is exactly one learned model in the active path: `GenomeCompiler`.

The Runtime is normal deterministic tensor code. It executes compact primitives:

```text
BASE_COPY
LOW_RANK
HADAMARD_SCALE
QUANTIZED_VECTOR
SPARSE_PATCH
COPY_FROM_TIED
```

There is no dense-delta opcode, exact-residual opcode, neural-field opcode, or one-value-per-weight escape hatch.

## First corpus

Only the standard non-deduplicated Pythia/PolyPythia family is used initially.

| Lives | Assignment |
|---|---|
| Pythia 14M seeds 0–7 and 9 | training |
| Pythia 14M seed 8 | development |
| Pythia 31M seeds 0–7 | training |
| Pythia 31M seed 8 | development |
| Pythia 31M seed 9 | fresh hidden evaluation |

The split unit is one complete model life. A checkpoint is not an independent example.
Pythia 14M seed9 is a normal training life. Its W0 and WT are available during
training-data preparation, and its accepted compact target may supervise the compiler.

## Program and functional gates

A fitted target MGP is only a candidate until it passes all of these gates:

1. It uses only approved compact primitives.
2. Its actual serialized target-specific bytes are at most 10% of direct fp16 Delta-T.
3. It loads and executes through the Runtime.
4. The resulting model has finite logits.
5. It beats W0.
6. On development lives, it closes at least 80% of the W0-to-WT validation-loss gap.

The development threshold is declared before the fresh hidden endpoint is revealed.

Endpoint progress is:

\[
P=\frac{L(W_0)-L(\widehat W)}{L(W_0)-L(W_T)}.
\]

No repair is included in the one-shot result.

## Compiler architecture

The compiler is hierarchical rather than a flat coefficient language model:

```text
model/task evidence
    -> graph message passing over logical tensors
    -> bidirectional tensor encoder
    -> primitive and rank decisions
    -> shared coordinate-conditioned coefficient heads
    -> bounded per-tensor coefficient packets
    -> deterministic MGP serialization
```

Coefficients are emitted through bounded structured heads, not one sequence token per handful of floats.

## Repository map

| Path | Purpose |
|---|---|
| `genome/life.py` | Complete model-life and whole-life split contracts. |
| `genome/sources.py` | Pythia source plan, ref resolution, materialization and hidden reveal. |
| `genome/adapters/gpt_neox.py` | Reversible Pythia/GPT-NeoX state adapter. |
| `genome/fingerprint.py` | Corpus and endpoint-free W0 response evidence. |
| `genome/mgp/` | Compact program schema, fitting, policy, serialization and Runtime. |
| `genome/compiler/model.py` | The single learned GENOME Compiler. |
| `genome/compiler/data.py` | Architecture/W0/evidence features and compiler corpus records. |
| `genome/compiler/train.py` | Resumable compiler training and development selection. |
| `genome/evaluation.py` | Functional Genome Gate and endpoint-progress metrics. |
| `genome/hidden.py` | Prediction sealing before hidden WT reveal. |
| `genome/workspace.py` | Fresh RunPod volume layout. |
| `docs/THEORY_AND_MATH.md` | Formal problem and objective. |
| `docs/EXPERIMENT_PLAN.md` | Ordered gates from source audit through hidden evaluation. |
| `docs/RUNPOD_HANDOFF.md` | Exact handoff for the next agent. |
| `docs/SOURCE_MATRIX.md` | Pythia source and storage plan. |

## Local commands

```bash
cd track_2
python -m pip install -e '.[dev,evaluation]'
python -m compileall -q genome tests
python -m pytest -q
python -m genome --help
```

Initialize a new RunPod network volume:

```bash
genome init-workspace --root /workspace/genome_v1
```

Write and resolve the source plan:

```bash
genome write-source-plan --output configs/sources/pythia_v1.generated.json
genome resolve-source-plan \
  --plan configs/sources/pythia_v1.generated.json \
  --output /workspace/genome_v1/control/pythia_v1.pinned.json
```

The hidden 31M seed9 WT is not resolved or downloaded by this command.

## Current status

Implemented and locally exercised:

- complete model-life schema;
- whole-life split enforcement;
- hidden endpoint exclusion;
- source planning and pinning;
- reversible GPT-NeoX adapter;
- corpus and W0-response fingerprints;
- compact MGP schema and deterministic Runtime;
- low-rank target fitting with scalable truncated SVD;
- actual serialized byte audits;
- functional target refinement hook;
- functional Genome Gate;
- hierarchical variable-tensor compiler;
- prediction-dependent byte proxy;
- structural and functional compiler losses;
- compiler checkpointing and resume artifacts;
- hidden prediction seal;
- local unit and end-to-end compiler smoke tests.

Still intentionally left for the RunPod agent:

- pin exact Hugging Face commits in the network environment;
- materialize the approved Pythia W0/WT pairs;
- build real Pile sample and W0-response evidence;
- fit and functionally refine real compact target programs;
- admit only programs that pass the gates;
- construct the compiler corpus manifest;
- run the real GPU smoke and production compiler training;
- seal and evaluate fresh hidden Pythia 31M seed9.

Read `AGENTS.md` and `docs/RUNPOD_HANDOFF.md` before modifying the project.
