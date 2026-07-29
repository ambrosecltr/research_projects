# Production readiness blockers

The cleanup foundation passes local tests. Production Pythia compiler training is not ready.

## 1. Flat program tokens do not scale

The current token format stores coefficient chunks in the autoregressive sequence. With 16 fp16
values per chunk and 76 tensor records:

| Model | Exact parameters | 5% median-budget case | 10% upper-budget case | Limit |
|---|---:|---:|---:|---:|
| Pythia 14M | 14,067,712 | 44,268 tokens | 88,230 tokens | 4,096 |
| Pythia 31M | 30,494,720 | 95,602 tokens | 190,898 tokens | 4,096 |

These are analytic capacity estimates. The 5% case is a median-budget case, not a measured median
from fitted targets. Both cases exceed the compiler limit by a large amount.

Required implementation:

- generate a short autoregressive program skeleton;
- select primitives and discrete arguments in that skeleton;
- emit bounded per-tensor coefficient packets from shared primitive-specific heads;
- serialize those packets outside the flat autoregressive token stream;
- reject overflow. Do not truncate it and do not raise the context to tens of thousands of tokens.

## 2. SVD labels are not unique

The fitter fixes the simple sign ambiguity. It does not uniquely order repeated or nearly repeated
singular subspaces across hardware and linear-algebra libraries.

Required implementation:

- train through the deterministic Runtime and a function/model loss so an equivalent
  factorization is not marked wrong; or
- define and test a stable canonical subspace rule for the approved source family.

The Runtime/function-loss option is the safer contract.

## 3. No approved public source matrix exists

The source-audit schema is strict, but no real source matrix and frozen whole-life split is
committed. The validation did not download endpoints.

Required implementation:

- audit exact repositories, revisions, licences, LFS files, true W0, WT, dataset content, exact
  data order, tokenizer, complete recipe, and provenance;
- use actual HF/LFS byte receipts after download;
- freeze training, development, hidden, evaluation-only, and quarantine assignments;
- keep the revealed Pythia 14M seed9 life in quarantine.

## 4. No real Pythia compact target has passed the functional gate

The SVD fitter produces only a target candidate. Passing the structural and serialized-byte policy
does not make it valid training supervision.

Required implementation for each development target:

- decode the serialized MGP through the deterministic Runtime;
- load the state into the real GPT-NeoX model;
- prove that it beats W0 under a predeclared functional gate;
- pass validation loss, logit, task, decode-time, and actual-byte checks;
- reject it if any required gate fails.

## 5. There is no calibrated differentiable rate objective

The old rate term was a constant teacher-target length. It had no useful gradient and was removed.

If the compiler needs a rate term, implement a prediction-dependent proxy. Calibrate it against
actual serialized bytes. Report the calibration error. Do not label teacher length as predicted
rate.

## Readiness decision

Overall production readiness is **FAIL**. A RunPod implementation agent must solve blockers 1-4
before production training. Blocker 5 is required before any rate-regularized compiler objective is
claimed.
