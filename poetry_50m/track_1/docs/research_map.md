# Track 1 research map: endpoint-informed poetry training

## Purpose and evidence standard

Track 1 asks a concrete, deliberately open question: after observing how an 8M language model reaches a useful poetry endpoint, can part of that path be predicted well enough to reach comparable poetry quality in fewer training steps and less elapsed GPU time? The result may be a conventional optimizer, intermittent checkpoint transport, a controller of the training path, or a hybrid. It is not assumed to be any one of those before evidence selects one.

Each claim below is labelled as one of:

- **Demonstrated** — a cited experiment showed this in its stated setting.
- **Plausible extrapolation** — a reasonable use of demonstrated work outside its original setting; it is not itself established.
- **Speculative hypothesis** — a Track 1 proposal requiring an experiment.

The training regime matters. A result in **from-scratch pretraining** is closest to the target. **Fine-tuning** results are useful geometry or mechanism evidence, but not pretraining proof. **RLVR** results are highly adjacent trajectory evidence, but optimize a reward-driven adaptation path rather than ordinary next-token pretraining.

## Track 1 corpus contract

The first lineage trains from scratch on pinned Nano Wiki and distilled BabyLM
knowledge artifacts in
[`configs/data/huggingface_sources.json`](../configs/data/huggingface_sources.json)
plus separately generated, critiqued, locally filtered Cerebras GPT-OSS poetry.
The pinned sources create the auxiliary prose NTP stream and the synthetic
records create conditional poetry targets. Exact normalized-text duplicates are
removed before the whole-document split and tokenizer fit. Nano Wiki is
attributed as CC BY 4.0; distilled BabyLM remains rights-unknown because its
upstream per-document provenance was not retained.

The intended 80/20 mix means four conditional-poetry batches for each
auxiliary-prose batch (`1.0` / `0.25`), not a claim about raw documents or
tokens. Prepared artifacts report the realized packed supervised-token mix.

## The six mechanism families

### 1. Direct future-checkpoint prediction and speculative transport

**Idea.** Train normally for a short window, forecast a checkpoint \(K\) steps ahead, and only keep the leap if an inexpensive, fixed verification set accepts it. The predictor can be a finite difference, a low-order local fit, or a small learned predictor. This is a transport operation, not necessarily a per-step optimizer.

- **Demonstrated — mixed vision transfer.** [Introspection](https://arxiv.org/abs/1704.04959) trained a compact network to forecast scalar weight evolution from a source run and used jumps while training MNIST, CIFAR-10, and ImageNet classifiers. Its authors explicitly note early jumps could harm the outcome. This is neither transformer nor language-model pretraining evidence.
- **Demonstrated — from-scratch LM pretraining, but preliminary.** [Leap+Verify](https://arxiv.org/abs/2602.19580) evaluated analytic forecasts on GPT-2 124M and Qwen 2.5 1.5B on WikiText-103. Adam-moment extrapolation produced very large loss increases; linear and quadratic finite-difference forecasts were sometimes accepted after held-out-loss verification. Its regime detector used a fixed probe set’s activation-space cosine similarity. The paper measures acceptance, not an established end-to-end wall-clock win.
- **Plausible extrapolation.** An 8M model is small enough that periodic checkpoints, a probe set, and several forecast candidates are cheap. The verified-jump pattern is therefore a particularly good personal-project substrate.
- **Speculative hypothesis.** A forecast in a symmetry-aware, low-dimensional coordinate system will be accepted more often than a raw parameter forecast at the same compute budget.

**Track 1 implication.** Direct, verified finite-difference jumps are the first control baseline. Never use extrapolated Adam moments as the first method; Leap+Verify makes that a poor bet rather than an impossibility.

### 2. Low-rank and low-dimensional trajectory models

**Idea.** Rather than forecast every scalar weight independently, find a small moving subspace \(U\) and forecast coefficients \(a_t\), so \(W_t \approx W_0 + U a_t\). The representation may be global, per layer, per matrix, or spectral.

- **Demonstrated — RLVR.** [RELEX](https://arxiv.org/abs/2605.21468) reports that most measured RLVR gains of three Qwen-family models lie in a rank-one parameter-delta direction with near-linear coefficient evolution. Its linear extrapolation method matched or exceeded full RLVR results in its experiments while observing as little as 15% of the training steps. This is a strong adjacent result, not ordinary pretraining evidence.
- **Demonstrated — RLVR with LoRA.** [NExt](https://arxiv.org/abs/2604.11446) reports that the rank-one component in LoRA RLVR is important but non-linear, then learns a predictor for that trajectory and reports about 37.5% less RLVR compute. It does not show that dense from-scratch transformer pretraining is rank one.
- **Demonstrated — from-scratch transformer pretraining observation.** [The Spectral Lifecycle of Transformer Training](https://arxiv.org/abs/2604.22778) tracked full SVDs every 25 steps for 30M–285M transformer pretraining runs and reports depth-dependent, projection-dependent spectral dynamics. Its evidence supports inspecting singular structure at the target scale; it does not demonstrate a training accelerator.
- **Plausible extrapolation.** The low-rank RLVR findings and structured 30M pretraining spectra justify testing low-dimensional coordinates before building a learned optimizer.
- **Speculative hypothesis.** For this corpus and architecture, a per-matrix truncated-SVD delta basis with a low-order coefficient fit will beat a global flattened-weight basis, because attention and MLP matrices need not share a common timescale.

**Track 1 implication.** This family is the highest-information first experiment: it is transparent, cheap, and distinguishes “the path is predictable” from “a large auxiliary model memorized checkpoints.”

### 3. Learned local optimizer or preconditioner

**Idea.** Preserve the current gradient step, but learn or discover an update correction from local information: gradient/momentum, update history, parameter norms, layer input covariance, activation statistics, loss slope, and training phase. The output may be a scalar schedule, diagonal multiplier, matrix preconditioner, or a low-rank transport coefficient.

- **Demonstrated — analytic local preconditioning in pretraining.** [Newton-Muon](https://arxiv.org/abs/2604.01472) derives a matrix update using gradient and layer-input second moment, then reports 6% fewer iterations and about 4% less wall-clock time on its reproduction of an early Modded-NanoGPT GPT-2 pretraining configuration. It is a concrete candidate baseline, not proof for every architecture or corpus.
- **Demonstrated — program discovery.** [Symbolic Discovery of Optimization Algorithms](https://arxiv.org/abs/2302.06675) discovered Lion through program search and simplification. Lion was evaluated across several modalities, including autoregressive and masked language modelling, but was not designed around endpoint trajectories.
- **Demonstrated — learned-optimizer feasibility, with differing budgets.** [Celo](https://openreview.net/forum?id=SLqJbt4emY) reports a versatile learned optimizer meta-trained in 24 GPU hours; [Celo2](https://openreview.net/forum?id=hxDB30LwVe) reports a 4.5-GPU-hour learned rule that scales in its experiments to 1.3B language-model pretraining. These are evidence that learned update rules need not require VeLO’s enormous historical budget, not a promise of a free general optimizer.
- **Plausible extrapolation.** A small layer-shared controller trained from Track 1 reference checkpoints could learn when to apply a safe jump or modify a Newton-Muon/Muon-like direction without learning a full optimizer from scratch.
- **Speculative hypothesis.** Fit a learned local controller first, then distill its behaviour by symbolic search into a concise formula. The formula is the deliverable only if it survives the unseen-run gate; no formula is assumed in advance.

**Track 1 implication.** This is a second-stage branch, after the reference run establishes which local statistics forecast progress. It should predict low-dimensional controls rather than 8M independent updates.

### 4. Function-space endpoint prediction

**Idea.** The target is not a particular arrangement of weights. On a fixed anchor set \(A\), save future logits, hidden states, or representation statistics \(f_{W_T}(A)\). Train the current model toward a predicted future function while preserving normal next-token loss:

\[
L = L_{\mathrm{NTP}} + \lambda_t D\big(f_{W_t}(A),\widehat f_{W_{t+K}}(A)\big).
\]

This can work even when two equally good networks use different hidden-unit permutations.

- **Demonstrated — related checkpoint generation, not function-space transport.** [G.pt](https://openreview.net/forum?id=JXkz3zm8gJ) models checkpoint distributions conditioned on weights and desired loss/error/return, and reports one-update optimization of unseen initializations for small MLP/CNN/RL architectures. It is direct evidence that checkpoint-conditioned generation can work, but its data contained millions of checkpoints across many small runs and it does not establish a function-space endpoint method for LM pretraining.
- **Demonstrated — fine-tuning only.** [On Task Vectors and Gradients](https://arxiv.org/abs/2508.16082) studies vision fine-tuning/task arithmetic and finds early gradient direction can matter strongly. It is inspiration for using early future-function signals, not support for from-scratch language pretraining geometry.
- **Plausible extrapolation.** Fixed anchors make endpoint targets invariant to
  many weight-space symmetries and could permit compact future storage. The
  implemented first branch instead retains full R0 snapshots and extracts
  selected logits/final residuals post hoc; it does not persist attention
  summaries.
- **Speculative hypothesis.** A future-logit/representation target on anchors will provide a better acceptance signal than raw delta MSE, especially across unseen seeds, because it scores behaviour rather than coordinates.

**Track 1 implication.** This is the best escape hatch if weight alignment is unstable. It should begin as a diagnostic/verification target rather than immediately becoming an expensive online teacher-distillation loop.

### 5. Symmetry-aware weight generation and hypernetworks

**Idea.** A transformer’s heads, MLP channels, and some equivalent internal bases can be re-indexed without changing the computed function. Before comparing independent trajectories, align/canonicalize them or use a representation/equivariant model that respects those transformations. Then a hypernetwork, flow, or graph model may generate a future weight state.

- **Demonstrated — alignment in vision networks.** [Git Re-Basin](https://arxiv.org/abs/2209.04836) gives algorithms to permute hidden units into an aligned weight-space basin and demonstrates zero-barrier linear connectivity for independently trained ResNets on CIFAR-10. It is compelling symmetry evidence, but not a guarantee that a full decoder-only transformer can be globally re-based by the same recipe.
- **Demonstrated — weight generation after canonicalization.** [DeepWeightFlow](https://arxiv.org/abs/2601.05052) uses re-basing/canonicalization before flow matching complete weights and reports high-performing generated weights across its tested architectures. The authors’ target settings are not 8M language models.
- **Plausible extrapolation.** For a fixed transformer architecture, head and channel matching using activation signatures from a fixed anchor set is a practical first approximation before any cross-seed delta averaging.
- **Speculative hypothesis.** An architecture-graph, permutation-equivariant trajectory predictor can transfer across width/depth variants more readily than a flat-vector predictor. This is a later Track 1 branch, not the first weekend experiment.

**Track 1 implication.** Within one continuous run, coordinates remain aligned by construction. Across seeds, curriculum runs, or structures, raw weight averages are prohibited until an alignment test passes. Function-space and spectral representations remain valid alternatives if re-basing does not.

### 6. Architecture and training-path co-design

**Idea.** Build a model and schedule whose trajectory is intentionally observable and predictable: normalized vectors, layer sharing/recurrence, progressive depth or width, precision changes, curriculum control, and scheduled parameter release can all alter the geometry that an accelerator must forecast.

- **Demonstrated — architecture can reduce steps.** [nGPT](https://arxiv.org/abs/2410.01131) normalizes vectors composing embeddings, MLP/attention projections, and hidden states, reporting 4–20× fewer steps to matched accuracy in its experiments. This is a step result, not an automatic wall-clock factor. Its matrices are collections of normalized vectors (a product of spheres), not one globally normalized matrix.
- **Demonstrated — curriculum ordering in from-scratch LM training.** The exact hard-to-easy paper is [On the Role of Corpus Ordering in Language Modeling](https://aclanthology.org/2021.sustainlp-1.15/). Its one-epoch transformer experiments found document-level hard-to-easy ordering outperformed its vanilla baseline by 1.7 average GLUE points after fine-tuning, while vanilla needed roughly twice as many steps to match it. This is the relevant document-ordering result; it is not the separate masking study.
- **Demonstrated — objective curriculum in small LM pretraining.** [Pre-Training Curriculum for Multi-Token Prediction](https://arxiv.org/abs/2505.22757) finds a reverse objective curriculum (MTP to NTP) improved NTP performance/output quality in its small-model experiments but did not retain self-speculative decoding benefits. This supports trying objective ordering as a separate lever, not conflating it with document ordering.
- **Plausible extrapolation.** nGPT is a strong **Branch A** candidate because per-vector normalization could make trajectory coordinates easier to fit. The claim that it will make them cleaner or more transferable is untested.
- **Speculative hypothesis.** Train a compact base depth first, then grow by inserting identity-initialized/residual-neutral blocks at predetermined milestones; forecast only the stable base layers until the added block has its own observed trajectory. This is a pathway-design experiment, not a claim that progressive depth is universally superior.

**Track 1 implication.** Co-design is not a default architecture choice. It is a knob deliberately held back until the baseline tells us whether the limiting problem is norm drift, nonlinearity, layer-specific dynamics, or data ordering.

## Cross-cutting constraints that expand the search space

### Weight coordinates are not universal

Within a run, \(W_{t+K}-W_t\) is meaningful. Across independent runs it may not be: functionally equivalent hidden units and heads can be permuted. The minimum safe procedure is:

1. Use raw deltas only within a single run.
2. For cross-run comparison, evaluate a candidate re-basing/alignment using a fixed anchor set and require that it preserves outputs before using aligned deltas.
3. If that fails, compare activation signatures, spectra/subspaces, or function-space targets instead of scalar coordinates.
4. Treat a cross-seed average as a hypothesis needing an output-preservation and acceptance check, never as a mathematically neutral operation.

### A poetry model needs conditional data, not only tags

`<PROMPT>`, `<THOUGHT>`, and `<POEM>` tokens establish modes but do not teach relation. Training records must contain genuine conditional sequences, for example:

```text
<PROMPT>
A poem about letting a beautiful moment pass without possession.
<THOUGHT>
Theme distilled from an attributed philosophical passage.
<POEM>
Target poem, title, or source excerpt.
```

Every prompt/poem relation also produces the primary prompt-only
`<PROMPT>...<POEM>` row used by the public generation interface. When an
attributed thought is available, the `<THOUGHT>` form above is an additional
training variant, never a replacement for that prompt-only relation.

Build the current corpus prompts deterministically from a clean source title,
falling back to source-author conditioning when a title is absent or unusable.
Store source, supplied edition/translation fields, transformation method,
rights status, and split at document/poem level. `user_supplied_personal_copy`
and `unknown` remain descriptive, not licence claims, and permit local
preparation; only an explicitly denied source is excluded. Never place a poem,
its paraphrase, or any identified duplicate across train and evaluation splits.
This project records provenance without asserting redistribution or publication
rights; synthetic fixtures require explicit opt-in.

### Three success claims, deliberately separated

| Level | Claim | Minimum evidence |
| --- | --- | --- |
| 1 | **Same-run trajectory compression.** The method reaches the studied reference run’s quality from the same initialization and same data order faster. | Reference endpoint and a replay using the pre-registered checkpoint-derived method. |
| 2 | **Transferable accelerator.** It accelerates an unseen seed or unseen data order on the same architecture. | The held-out endpoint is never used to fit the predictor; blind evaluation passes. |
| 3 | **Training law / general optimizer.** It transfers across a material change such as corpus mix/curriculum, width/depth, or architecture family. | One predeclared structural transfer succeeds, with no endpoint leakage. |

Level 1 is a worthwhile discovery. It must be reported as trajectory compression or endpoint distillation, not as a general optimizer.

## Recommended first and backup experiments

### Highest-information first: verified low-rank checkpoint transport

Use one 8M reference pretraining run and save a compact early/middle
weights-only checkpoint window. Freeze the deterministic anchor selection
before target training, then extract its logits/final residuals post hoc from
those snapshots during verification; the first branch does not write separate
per-cadence anchor feature files. Fit no neural predictor initially: compare
only (a) continued baseline training, (b) per-matrix linear finite-difference
forecast, and (c) low-rank temporal-Gram coefficient extrapolation. Each leap
is accepted only if the fixed verification loss and function-space drift gates
pass. Run one same-initialization replay to measure Level 1; reserve a fresh
data order with the same initialization for the raw Level 2 gate.

Why this first: it directly tests the central premise, reveals whether useful motion is low-dimensional, exposes unstable layers and phases, costs little additional GPU time, and creates the checkpoint dataset needed by every richer branch. It has no hidden auxiliary-model capacity that can disguise memorization.

### Backup: anchor-function target with local controller

If raw or spectral transport rarely passes verification, train a tiny layer-shared controller to choose a conservative coefficient for the baseline update using current loss slope, gradient/update norms, activation similarity, and per-layer spectral summaries. Train it against the reference run’s **future anchor-function improvement**, not raw final weights. It only adjusts a strong baseline optimizer and must beat its matched-quality wall-clock time on the held-out run.

Why this backup: it preserves the endpoint idea while avoiding cross-run coordinate alignment and avoids the cost of training a full weight-generating diffusion model.

## Deliberately deferred branches

Full checkpoint diffusion/flows, global hypernetworks, learned optimizers that output every parameter, broad architecture sweeps, and cross-architecture averaging are real research options, but they are deferred because they obscure the initial question. They become justified only after the reference trajectory establishes a stable signal that their extra capacity can exploit.
