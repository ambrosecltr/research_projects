# Track 1 experiment protocol

## Decision

Start with **verified low-rank checkpoint transport**. It is a compact endpoint-informed experiment: one reference 8M training trajectory teaches a forecast rule; one replay tests same-run compression; one sealed run tests transfer. The aim is a useful poetry model faster, not a paper-shaped grid.

The protocol treats published components as reusable. It re-tests only the integration boundaries created by this corpus, this model scale, the transport combination, and the success claim being made.

## Fixed object of study

### First lineage

The implemented first lineage is **training from scratch with a corpus-trained
tokenizer** on the three pinned Hugging Face artifacts in
[`configs/data/huggingface_sources.json`](../configs/data/huggingface_sources.json).
Every run, trajectory, and endpoint claim is about that exact corpus plus its
acquisition, build, preparation, and exposure-plan receipts. There is no
Cerebras generation or synthetic merge in this lineage.

| Source | Pinned revision | Role |
| --- | --- | --- |
| `openbmb/Ultra-FineWeb-L3` | pinned in source config | English Multi-Style general-prose NTP (synthetic rewriting) |
| `biglam/gutenberg-poetry-corpus` | pinned in source config | CC0-released line corpus, unconditional book-verse NTP |
| `yoonholee/poetry-greats-public-domain` | pinned in source config | public-domain poems, conditional prompt-to-poem |

Ultra-FineWeb-L3 is synthetic rewriting, not human-reviewed prose; its role is
explicitly bounded to auxiliary NTP. The Gutenberg line corpus does not retain
per-book rights records, so its individual books remain marked unverified.
Poetry Greats is retained under its dataset-card public-domain assertion. Do
not call the whole blend "human reviewed" or uniformly public domain.

### Data contract

Acquire the three pinned revisions into an external acquisition directory, then
run `corpus-build --selection-config configs/data/knowledge_corpus_selection.json`
to create the canonical JSONL inputs. The selection policy fixes the
hash-priority Ultra-FineWeb subset. The receipts bind source revisions,
artifact hashes, selection, transformations, and final output hashes. A source
document has one stable semantic document ID, so all conditioning for one poem
remains in one split.

Every retained source record has an immutable source ID, source locator, rights
status, raw text, cleaned text, and transformation lineage. Ultra-FineWeb rows
are auxiliary prose NTP; Gutenberg rows are contiguous `verse_document` book
records for raw poetry NTP; Poetry Greats rows carry deterministic title or
author-style prompts for conditional poetry.

Create records with actual conditioning:

```text
<PROMPT>
{source title, or source author when the title is absent or unusable}
<THOUGHT>
{short attributed philosophical theme or passage}
<POEM>
{poem or source excerpt}
```

For Poetry Greats, the corpus builder deterministically uses a clean source
title, then falls back to a source-author conditioning prompt when a title is
absent or unusable.
All variants retain their shared source ID so none cross the split boundary.
The fixed suite contains 40 predeclared, deliberately novel generation
requests and a separate 10-request development set. Hold out poem/passage
material separately for NTP validation and testing. Do not generate evaluation
prompts from evaluation outputs after inspecting models. This protocol records
provenance and does not assert redistribution or publication rights.

Every prompt/poem relation produces a prompt-only
`<PROMPT>…<POEM>` training example matching the public generation interface.
When an attributed thought exists,
`<PROMPT>…<THOUGHT>…<POEM>` is an additional variant, never a replacement for
the prompt-only row.

For curriculum, use three predeclared orderings only: shuffled baseline, document-level hard-to-easy scored by the reference model’s first-pass loss, and one selected ordering for the accelerated run. The hard-to-easy option is supported by [Agrawal et al.](https://aclanthology.org/2021.sustainlp-1.15/) in a different LM setting; it is an integration experiment here, not an assumed improvement. Do not substitute token masking studies for this claim.

The approved objective mix is conditional poetry `0.1`, auxiliary prose NTP `0.4`,
and Gutenberg book-verse NTP `0.5`. The scheduler uses actual unpadded data-token
exposure rather than whole-batch counts. Before a full run, `plan-exposure` must
freeze an immutable receipt and derived training configuration for two 20x passes
(at least 333,400,320 tokens for the 8,335,008-parameter model).

### Model contract

Use the fixed 8,335,008-parameter decoder in `configs/model/track1_8m.yaml` for
the first lineage: 6 layers, width 288, 6 heads, FFN width 768, tied 8,192-token
embeddings, and a 1,024-token context. Record tokenizer hash, precision,
optimizer, data order, and seed. A conventional normalized decoder is the
control; nGPT remains only a possible later co-design branch.

The main optimizer should be a strong, simple baseline with all states checkpointed. Newton-Muon is eligible only after confirming its implementation and matrix-shape assumptions; its reported GPT-2 speedup in [Newton-Muon](https://arxiv.org/abs/2604.01472) is evidence for a candidate baseline, not a required dependency.

## Runs and sealed information

| Run | Role | May inform the transport rule? | May decide final claim? |
| --- | --- | ---: | ---: |
| R0 | Reference trajectory; fixed seed/order | Yes | Establishes target endpoint only |
| R1 | Same seed/order replay | Yes, through R0 only | Level 1 |
| R2 | Sealed raw-transport transfer: same initialization, unseen data order | No endpoint/checkpoint information | Level 2 |

R2 is initialized with the same seed/coordinates as R0, assigned an unseen data order, and endpoint-hidden before fitting the transport rule. Inspecting an R2 checkpoint to redesign the rule voids Level 2 and creates a new development run. Cross-seed raw weight transport is rejected because matching tensor shapes do not establish matched hidden-unit/head coordinates; it may be examined only through the function-space diagnostic path. This is not an ablation demand; it prevents endpoint memorization from being misreported.

## R0: reference trajectory capture

Train R0 normally to the predeclared target token budget or selected fixed
quality point. The implemented first branch saves:

- full checkpoints with optimizer, scheduler, scaler, RNG, and data cursor at
  the configured explicit steps;
- weights-only analysis snapshots at the configured cadence;
- training loss, processed/supervised token counts, step time, and elapsed
  wall-clock telemetry;
- configured-cadence per-layer parameter/update norms and cosine similarities;
  and
- data-order/curriculum position plus coordinate-affecting code and
  configuration hashes.

The first branch does not persist per-cadence anchor or per-matrix spectral
feature files. Full R0 snapshots and the immutable prepared validation artifact
permit deterministic post-hoc extraction of the exact selected anchor logits
and final residuals after the target policy is frozen. The low-rank forecast
reports temporal-Gram diagnostics when it is run; it does not claim a stored
randomized-SVD capture stream.

The target policy commits 16 validation packs × 8 supervised positions by
default, separate from the 40 poetry-generation prompts. That deterministic
selection is fixed before R1/R2 training, used for jump verification, and never
used to select model prose by hand.

## Transport rule and acceptance

At a candidate jump point \(t\), form a recent per-matrix delta history \(\Delta W\). Test two no-learned-model rules:

1. **Linear finite difference:** \(\widehat W_{t+K}=W_t+K(W_t-W_{t-h})/h\).
2. **Low-rank coefficient forecast:** obtain a truncated basis from recent deltas of each matrix, fit a low-order curve to its coefficients, extrapolate only the coefficient vector, then reconstruct the delta.

The model is never replaced blindly. Online verification normally compares the prepared candidate with the live current checkpoint \(W_t\), which is a cheap safety/proxy gate. A matched continued-baseline checkpoint \(W_{t+K}\) may be supplied explicitly for a true leap-versus-continued comparison; record which comparator was used. A candidate is accepted only if all gates pass on fixed verification material:

- NTP verification loss is no worse than the selected comparator: the live \(W_t\) proxy for a cheap online gate, or an explicit matched \(W_{t+K}\) continued baseline when supplied;
- anchor logits/representations stay within a predeclared drift envelope;
- every prepared candidate tensor passes finite-value and predeclared
  relative-norm checks;
- the selected anchor logits and final residual representations produce finite
  drift metrics; and
- the cloned optimizer/scheduler/scaler/RNG/stream continuation completes and
  its post-leap held-out loss does not spike beyond the same envelope.

If no candidate passes, resume normal training. This “jump and verify” structure follows the useful aspect of [Leap+Verify](https://arxiv.org/abs/2602.19580); its negative result on direct Adam-moment extrapolation is reason to exclude that method from the initial rule.

For normalized-vector architectures, project/retract each constrained vector separately **before** safety checks, fixed-anchor evaluation, acceptance, and application; do not score raw off-manifold weights and normalize only after acceptance. This respects nGPT’s stated per-vector product-of-spheres geometry. The accepted candidate and the applied candidate must be the same retracted state dict.

## Symmetry handling

R0-to-R1 and the narrowly permitted raw R0-to-R2 case do not require
re-basing: both use the exact same initialization, architecture, and named
tensor coordinates. R2 changes only the data order. Any transported candidate
still has to pass the fixed function and loss gates; same coordinate labels are
not evidence that the transport is useful.

Cross-initialization or cross-structure comparisons do not have that coordinate
contract. Raw transport is prohibited for them in the first branch. A future
such experiment would first need an implemented alignment method whose
candidate preserves anchor outputs; otherwise it must operate on
permutation-robust objects such as spectra, update norms, subspaces, or anchor
functions.

[Git Re-Basin](https://arxiv.org/abs/2209.04836) and [DeepWeightFlow](https://arxiv.org/abs/2601.05052) establish that symmetry-aware alignment/canonicalization is a real tool, not that one generic transform solves transformer symmetries. Flat cross-seed deltas are therefore excluded from the first experiment.

## Fixed evaluation

At matched token/quality checkpoints, produce outputs for the 40 fixed unseen prompts with identical sampling parameters and three fixed random seeds. Blind the model identity before personal judgement. Record:

- prompt relevance;
- poetic quality/image/music judged pairwise against the reference;
- repetition or degeneration;
- verbatim substring and n-gram overlap against training poems;
- held-out NTP loss;
- wins/losses/ties against reference; and
- steps, processed tokens, elapsed time, and cost to the selected point.

The user’s literary judgement is the primary quality signal. The fixed prompt set, blinding, multiple seeds, and overlap check make that judgement comparable rather than mechanized.

## Cost accounting and stop rules

For every run, report four quantities rather than “steps saved” alone:

\[
\text{step reduction}=1-\frac{S_{\mathrm{accelerated}}}{S_{\mathrm{reference}}},\qquad
\text{wall-clock reduction}=1-\frac{T_{\mathrm{accelerated}}}{T_{\mathrm{reference}}}.
\]

Also report:

- reference-run accelerator seconds and cost when directly measured;
- offline analysis/predictor accelerator and CPU seconds and cost;
- replay accelerator seconds and cost;
- verification and checkpoint I/O overhead;
- total discovery cost; and
- amortized cost after \(n\) future uses, \((C_{R0}+C_{analysis}+nC_{accelerated})/n\), with the break-even \(n\) where it beats \(n\) baseline runs.

Compute total discovery, per-replay, amortized, and break-even figures
independently for accelerator seconds, wall-clock seconds, CPU seconds,
synchronized device-active wall seconds, and USD whenever every component for
that unit is known. CUDA event durations are accelerator time. MPS exposes no
equivalent device-only duration in this workflow, so its accelerator time
remains `unknown`; synchronized MPS command-section elapsed is reported only as
device-active wall time when that section actually exercises the accelerator,
because it includes host scheduling. CPU-only trajectory forecasting and
checkpoint I/O are excluded from that unit. An unknown unit, missing price, or
unavailable MPS peak-memory measurement remains `unknown`; it is never treated
as zero and does not prevent known wall-clock or CPU units from being computed.
Record current MPS allocated memory with explicit semantics, checkpoint bytes
read/written, and checkpoint-I/O wall time beside these totals.

The pinned analysis receipt partitions its command wall and process-CPU time
into three non-overlapping roles: analysis, checkpoint I/O, and verification
per replay. A cost assembly references that same receipt and hash for all three;
role-aware extraction selects one component rather than charging the whole
analysis command repeatedly. Each assembly `estimated_cost_usd` is the total
accelerator-plus-CPU cost attributable to its selected role. It remains
`unknown` unless that total is known.

Stop the first branch if either (a) no transport candidate is accepted over the predeclared analysis windows, or (b) accepted jumps do not improve matched-quality wall-clock after one R1 replay. Preserve R0 artefacts; the outcome then directly selects the function-space local-controller backup. Do not spend on a new reference run merely to force a result.

## Claims allowed after the first lineage

- **Level 1:** Claim “same-run trajectory compression” only if R1 reaches the predeclared quality point with lower wall-clock and no fixed-evaluation regression.
- **Level 2:** Claim “transferable accelerator on this architecture” only if sealed R2 passes the same criterion. It must not use R2’s endpoint as a target during rule fitting.
- **Level 3:** Claim “general optimizer/training law” only after one predeclared material transfer — changed curriculum/corpus mix, width/depth, or architecture family — passes. This is intentionally future work, not a requirement for beginning Track 1.

## Backup branch: anchor-function local controller

If low-rank transport fails, keep the same R0 captures and train a small, layer-shared controller to output a conservative transport coefficient or baseline-update multiplier. Inputs are current/short-history loss slope, gradient and update norms, activation-similarity phase signal, and spectral summaries. Its training target is future anchor-function improvement, not an exact raw final weight.

This branch is a **speculative hypothesis**. It is bounded by the same R1/R2 gate and cost accounting. It does not begin with full weight diffusion, a hypernetwork that emits 8M scalars, or a broad optimizer meta-training campaign. G.pt, RELEX, NExt, and DeepWeightFlow make those later branches credible directions; they do not make them the efficient first personal experiment.
