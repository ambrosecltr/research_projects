# GENOME evaluation protocol and acceptance gates

> **Archived pre-recovery note.** Keep the functional metrics and leakage rules, but ignore
> learned-interpreter and decoder-training instructions. The active Runtime is deterministic.

## 1. Purpose

GENOME can appear to work while merely memorizing WT, ignoring decoder cost, exploiting the verifier, or producing low-MSE but nonfunctional weights. The evaluation protocol exists to prevent those false positives.

The core result is a **matched-quality cost comparison**, not a reconstruction screenshot or training loss.

---

## 2. Immutable evaluation assets

Genome Gate owns:

- frozen architecture and tokenizer manifests;
- `D_verifier_hidden`;
- R0 outputs on hidden anchors;
- final-window reference statistics;
- fixed poetry prompt suite and sampling seeds;
- acceptance-threshold manifest;
- raw checkpoint size and baseline training-cost records.

The compiler, interpreter trainer, latent refiner, patch allocator, and candidate selector do not have access to hidden verifier examples or outputs.

---

## 3. Candidate lifecycle

1. Train/fix representation using permitted data.
2. Generate one or more candidates.
3. Select candidates using only `D_probe` and declared compute.
4. Freeze each MGP and calculate its hash.
5. Submit frozen MGPs to Genome Gate.
6. Gate decodes and evaluates each frozen MGP exactly once for the main result.
7. Failed candidates may guide a later experiment only through released aggregate diagnostics, not hidden examples.

During development, a separate development verifier may be used. It must not be the hidden result set.

---

## 4. Structural validity metrics

Before executing the model, report:

- MGP version and hash;
- architecture/base/interpreter hash match;
- tensor-record completeness;
- missing/unexpected key count;
- shape and dtype mismatch count;
- tied-weight equality;
- NaN/Inf count;
- checksum status;
- decode determinism across two decodes;
- actual serialized bytes by component.

Any invalid structure is an automatic rejection.

---

## 5. Size and description-length metrics

### 5.1 Raw reference sizes

Report:

- logical raw WT bytes from dtype × numel;
- actual original checkpoint file bytes;
- canonical safetensors bytes;
- standard compressed archive bytes;
- W0 shared-base bytes.

### 5.2 Genome sizes

Report:

```text
manifest bytes
global/layer/tensor code bytes
structured factor bytes
codebook index/scale bytes
patch bytes
exact residual bytes
shared decoder/interpreter bytes
shared dictionaries/codebooks bytes
base bytes if stored
complete archive bytes
```

### 5.3 Ratios

\[
R_{payload}=\frac{B_{target\ payload}}{B_{WT}},
\]

\[
R_1=\frac{B_{target\ payload}+B_{shared}}{B_{WT}},
\]

\[
R_N=\frac{B_{target\ payload}+B_{shared}/N}{B_{WT}}.
\]

Report all three. The headline for G0 should normally include \(R_1\), not only the favorable amortized ratio.

### 5.4 Effective bits per child parameter

\[
\operatorname{bpp}=\frac{8B_{total}}{\operatorname{numel}(W_T)}.
\]

This is convenient but never replaces component-level accounting.

---

## 6. Parameter-space metrics

Compute for WT and candidate:

### Complete checkpoint

\[
\operatorname{NRMSE}_W=
\frac{\|\widehat W-W_T\|_2}{\|W_T\|_2+\epsilon}.
\]

### Learned displacement

\[
\operatorname{NRMSE}_\Delta=
\frac{\|\widehat\Delta-\Delta_T\|_2}{\|\Delta_T\|_2+\epsilon}.
\]

### Per tensor/role

- NRMSE;
- cosine similarity;
- relative norm error;
- top singular-value error;
- subspace angle for top-r singular spaces;
- spectral energy error;
- exact-bit equality fraction;
- residual entropy estimate.

Parameter metrics diagnose representation. They do not determine acceptance by themselves.

---

## 7. Primary functional metrics

### 7.1 Hidden validation loss and perplexity

\[
L_G=L_{NTP}(D_{verifier};\widehat W),
\qquad
PPL_G=e^{L_G}
\]

using the same token weighting and masking as Track 1.

### 7.2 Terminal-noise normalized gap

When final-window statistics exist:

\[
z_L=\frac{L_G-\mu_L}{\max(\sigma_L,\epsilon_L)}.
\]

Recommended default matched-quality criterion:

\[
z_L\le 2
\]

and no material regression on generation-health checks. The threshold must be frozen before hidden candidate evaluation.

If no terminal window exists, use a predeclared engineering tolerance such as:

\[
L_G\le L_{R0}+\max(0.01,0.01|L_{R0}|),
\]

but label it as a chosen tolerance.

### 7.3 Reference-logit divergence

On hidden anchors, for every predicted token position:

\[
D_{KL}=\frac1N\sum_i
D_{KL}\left(p_{R0,i}\|p_{G,i}\right).
\]

Also report:

- symmetric KL or Jensen–Shannon divergence;
- top-1 agreement;
- top-5/top-10 set overlap;
- logit cosine after centering;
- calibration/error by token-frequency bucket.

Calibrate the acceptance range using adjacent final-window R0 checkpoints where available.

### 7.4 Hidden-state similarity

At selected layers and token positions:

- residual-stream cosine;
- linear CKA;
- covariance spectrum distance;
- norm-ratio distribution;
- activation outlier rate.

Hidden agreement is diagnostic. A functionally strong alternative basin may have different internal coordinates.

---

## 8. Poetry-generation evaluation

NTP loss is necessary but insufficient for the project’s intended phenotype.

### Fixed generation protocol

For each prompt in `P_generation`:

- use fixed decoding settings;
- use the same set of random seeds for R0 and candidate;
- save raw token IDs and decoded text;
- include greedy and sampled generations;
- record generation latency separately from genome decode latency.

### Automated health metrics

- empty/invalid output rate;
- special-token/mode-transition validity;
- length distribution;
- repetition rate and repeated n-gram fraction;
- distinct-1/2/3;
- unique-line ratio;
- tokenizer unknown/error rate;
- stop-condition correctness;
- prompt-copy rate;
- corpus memorization/nearest-neighbour diagnostic where available;
- rhyme/meter proxies only if already meaningful for Track 1.

### Behavioural comparison

- R0 versus candidate log probability on each other’s sampled outputs;
- semantic embedding similarity distributions, used cautiously;
- style/classifier scores only as secondary diagnostics;
- blind human pairwise comparison by the user on a fixed subset.

Automated metrics must not be presented as proof of poetic quality. Include representative failures and outputs in the report.

---

## 9. Repair-to-quality curves

If a candidate does not initially pass, measure repair under a declared protocol.

### Repair methods

- full-weight continuation using the Track 1 baseline optimizer;
- LoRA/low-rank repair;
- genome-code refinement;
- genome-code plus patch refinement;
- compiler/linker iterations.

### Record at every repair point

- steps;
- tokens;
- forward/backward FLOPs estimate;
- GPU seconds and wall seconds;
- trainable parameter count;
- optimizer-state bytes;
- validation loss;
- hidden verifier evaluation only at predeclared checkpoints or on a development verifier.

For the hidden result, stopping must use `D_probe` or a fixed step budget. Do not stop based on hidden loss.

### Matched-quality repair cost

\[
C_{repair}^*=\min\{C: \text{candidate reaches frozen quality gate}\}.
\]

If the hidden gate cannot be queried during repair, estimate the first predeclared checkpoint that passes when evaluated after freezing the repair trajectory.

---

## 10. Complete child-generation compute

For one child candidate:

\[
\begin{aligned}
C_{child}={}&C_{fingerprint}+C_{early\ trajectory}
+C_{compile}+C_{sampling}\\
&+C_{decode}+C_{probe}+C_{link}
+C_{patch}+C_{repair}+C_{verification\ used\ for\ selection}.
\end{aligned}
\]

Report:

- tokens processed;
- forward FLOPs;
- backward FLOPs;
- GPU seconds;
- wall-clock seconds;
- peak GPU memory;
- CPU seconds for decomposition/encoding;
- disk I/O time when material.

Do not compare only optimizer step counts.

### Meta-training cost

Report separately:

\[
C_{meta}=C_{interpreter\ training}+C_{compiler\ training}+C_{linker\ training}+C_{checkpoint\ corpus\ creation}.
\]

Amortized cost over \(N\) children:

\[
\bar C_N=C_{child}+\frac{C_{meta}}N.
\]

Show break-even \(N\) where possible.

---

## 11. Baselines

Every relevant experiment includes the strongest applicable subset:

1. Unmodified R0.
2. W0/random initialization.
3. Continue ordinary Track 1 training from the same prefix.
4. Track 1 checkpoint-transport method from the same prefix.
5. Standard quantization/compression.
6. Per-tensor SVD.
7. Low-rank plus sparse.
8. LoRA repair at matched trainable parameters.
9. Full-weight repair.
10. Linear/quadratic extrapolation in a fitted latent trajectory.
11. Mean or nearest-neighbour genome from training runs.
12. Decoder with shuffled/random codes.

The nearest endpoint in the meta-training set is an important memorization baseline for G2.

---

## 12. Candidate sampling and selection

If the compiler generates \(n\) candidates:

- report \(n\);
- count all generation/decode/probe cost;
- use only `D_probe` to rank them;
- freeze the selection rule before hidden evaluation;
- report best, median, and distribution on development runs;
- do not report only the lucky hidden candidate without accounting for the search.

Best-of-N can be a legitimate algorithm. It is not a one-shot algorithm.

---

## 13. Research-level acceptance gates

### G0 gate: endpoint representation

Required:

- full byte accounting;
- valid decode;
- hidden functional metrics;
- comparison with transparent codecs;
- no claim beyond R0 representation.

Useful pass conditions:

- enters R0 terminal-quality band at a materially smaller payload; or
- requires only a small declared repair budget and improves the rate–repair Pareto frontier.

### G1 gate: same-run compilation

Required:

- compiler inference cannot access WT;
- conditioning set declared;
- compare from same early prefix/information;
- all compile/probe/repair cost counted;
- pure versus distillation mode labelled.

Pass when matched quality is reached at lower child cost than the strongest same-information baseline.

### G2 gate: hidden run

Required:

- target endpoint entirely withheld;
- split by run;
- no target-specific decoder training using hidden endpoint;
- alignment procedures trained/tuned without hidden endpoint;
- target candidate selected without hidden verifier.

Pass when the hidden run reaches the frozen quality band with lower child cost than ordinary training.

### G3 gate: structural transfer

In addition to G2, the held-out target includes a material task/architecture change and the compiler/interpreter contract supports it without target endpoint leakage.

### G4 gate: self-hosting

A child compiler generated by its parent must improve the hidden child-model Pareto frontier under equal evaluator and budget. See `09_RSI_AND_FUTURE_RESEARCH.md`.

---

## 14. Default result table

```text
candidate_id
research_level
conditioning_set
representation
payload_bytes
shared_bytes
single_model_ratio
amortized_ratio_N
bpp
decode_seconds
fingerprint_tokens/seconds
compile_seconds
probe_seconds
repair_tokens/steps/seconds
meta_training_gpu_hours
hidden_validation_loss
loss_z_score
perplexity
anchor_KL
top1_agreement
generation_health_summary
parameter_NRMSE_delta
pass/fail
failure_code
```

Accompany the table with rate–distortion and repair-to-quality curves.

---

## 15. Failure codes

Use stable failure labels:

```text
INVALID_FORMAT
HASH_MISMATCH
MISSING_TENSOR
SHAPE_MISMATCH
TIE_MISMATCH
NONFINITE_WEIGHTS
DECODE_NONDETERMINISTIC
BYTE_ACCOUNTING_ERROR
LOSS_GATE_FAIL
LOGIT_GATE_FAIL
GENERATION_HEALTH_FAIL
REPAIR_BUDGET_EXCEEDED
COMPUTE_BASELINE_NOT_BEATEN
TARGET_LEAKAGE
RUN_SPLIT_LEAKAGE
ALIGNMENT_NOT_VERIFIED
DECODER_MEMORIZATION
INTERPRETER_COST_DOMINATES
NO_RATE_DISTORTION_ADVANTAGE
UNKNOWN_FAILURE
```

A run can have multiple failure codes.

---

## 16. Leakage audit

Before accepting a result, answer:

- Did any target endpoint tensor enter compiler/interpreter input at inference?
- Did any final R0/target output enter refinement? If yes, is the result labelled distillation?
- Was the hidden verifier queried during candidate choice or stopping?
- Were checkpoints split by run rather than slice?
- Does a target ID select target-specific decoder parameters?
- Is the target embedded in a cache, fixture, constant, codebook, or preprocessing artifact?
- Did an alignment algorithm use hidden endpoint performance to select permutations?
- Was decoder size counted?
- Was best-of-N search counted?
- Was repair compute counted?

Any unreported yes invalidates the claimed level.

---

## 17. Statistical reporting

- Use identical prompt seeds across candidate and R0.
- For stochastic compiler outputs, evaluate multiple samples on development runs.
- Report confidence intervals or bootstrap intervals for token-level/verifier metrics where useful.
- For human poetry comparisons, randomize order and blind model identity.
- Do not overstate a difference smaller than final-window or seed-to-seed noise.
- For a single R0 result, call it a case study rather than a population estimate.

---

## 18. Required plots

At minimum:

1. hidden validation-loss gap versus total bytes;
2. anchor-logit KL versus total bytes;
3. matched-quality repair compute versus total bytes;
4. parameter NRMSE versus functional loss gap;
5. byte allocation by tensor role;
6. functional sensitivity versus delta energy by role;
7. prefix fraction versus total cost to matched quality for G1;
8. child cost versus ordinary-training cost, including amortization.

A good report makes it obvious whether a method compresses coordinates, preserves function, accelerates training, or only shifts cost into the interpreter.
