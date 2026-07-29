# GENOME Track 2

GENOME trains one compiler that generates compact programs for trained models.

```text
random W0 + architecture + endpoint-free corpus evidence + training recipe
    -> GENOME Compiler
    -> compact Model Genome Program (MGP)
    -> deterministic Runtime
    -> runnable candidate model
```

The compiler never receives the final weights of the life that it must predict.

## Pythia v1 split

| Role | Lives |
|---|---|
| Training and formula development | Pythia 14M seeds 0–6, 8, 9 |
| Training and formula development | Pythia 31M seeds 0–6, 8 |
| Fresh development | Pythia 14M seed7 and Pythia 31M seed7 |
| Hidden | Pythia 31M seed9 |

Seed8 affected formula design. Its old results are training and formula-development
evidence. They are not independent development confirmation.

Pythia 14M seed9 is a normal training life. Its W0 and WT are available for target
preparation. It is never a hidden evaluation.

Pythia 31M seed9 is the only hidden life. Its W0, architecture, evidence, and recipe
are available. Its WT stays unresolved until the compiler output and Runtime result
are sealed.

## Fixed rules

- Actual serialized target-specific bytes must be at most 10% of direct fp16 Delta-T.
- Development endpoint progress must be at least 80%.
- Do not add a residual or direct matrix.
- `DIRECT_VECTOR` is only for one-dimensional fp16 tensors with at most 4,096 values.
- Every audit reports the total bytes used by all `DIRECT_VECTOR` payloads.
- The formula is one immutable object with one `formula_id`.
- Production targets use the single `produce-target` command.
- Historical low-level fit, refine, evaluate, and accept commands are diagnostic tools.

GENOME targets a functionally good endpoint from the same pinned corpus and recipe.
It does not claim to recreate the exact raw WT coordinates or training path.

## Data checks

The existing even records are refinement data. The existing odd records are
formula-tuning data. They are not fresh development data.

Fresh development uses 128 batches from a second immutable verifier. The verifier
uses a different source shard and has its own receipt and SHA-256.

Every new evaluation and acceptance report binds:

- run ID and immutable formula ID;
- program ID, manifest SHA, and payload SHA;
- W0 and WT state IDs;
- evaluation JSONL SHA;
- source-plan ID;
- full code commit.

The compiler-corpus builder checks every binding again.

## Required order

1. Commit and test the protocol.
2. Freeze one global formula.
3. Rerun Pythia 14M seed5 once with that exact formula.
4. Regenerate every training target with that formula.
5. Run each seed7 development life exactly once.
6. Build the compiler corpus from accepted bound targets.
7. Train the compiler.
8. Select its checkpoint by free-running generated model quality.
9. Compile and seal one hidden Pythia 31M seed9 result.
10. Reveal hidden WT and evaluate.

The code blocks seed7 while the formula is not frozen. It also blocks seed7 until
all training reports for the same formula and source plan are present.

Pythia 14M seed5 is a declared rejected training target. Its frozen-formula rerun
must preserve full diagnostics. The expected compiler corpus is therefore 16 accepted
training targets and two accepted development targets, for 18 records.

## Compiler checkpoint selection

Teacher-forced development loss is diagnostic only. A selectable checkpoint must:

1. generate an MGP without teacher forcing;
2. serialize the real MGP;
3. pass the actual byte audit;
4. execute through the Runtime;
5. run real Pythia loss evaluation for at least 128 batches;
6. report endpoint progress.

The selected checkpoint has the best mean generated endpoint progress across the
two seed7 development lives.

## Hidden result tiers

| Endpoint progress | Meaning |
|---:|---|
| `P <= 0` | no signal |
| `0 < P < 0.25` | weak signal only |
| `0.25 <= P < 0.80` | partial result |
| `P >= 0.80` | strong result |

## Local checks

```bash
cd track_2
python -m pip install -e '.[dev,evaluation]'
python -m compileall -q genome tests
python -m pytest -q
python -m genome --help
```

Current state: both RunPod pods are stopped. The 100 GB network volume
`4kwmhcepgj` is intact. No compiler training has started. No seed7 target has run.
No hidden WT has been revealed.

Read `AGENTS.md`, `docs/EXPERIMENT_PLAN.md`, and `docs/RUNPOD_HANDOFF.md` before
continuing.
