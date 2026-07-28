# GENOME experiment plan

## 1. Philosophy

The experiment sequence is designed to answer one uncertainty at a time with the least auxiliary-model capacity possible.

Do not begin with a giant compiler. A large model can memorize WT and make an uninformative result look impressive. The first goal is a trustworthy rate–distortion audit of R0.

Every phase has:

- a concrete hypothesis;
- required inputs;
- minimal baselines;
- an acceptance gate;
- a failure diagnosis;
- a definition of done.

---

# Phase 0 — freeze the Track 1 organism

## Hypothesis

R0 and its evaluation can be reproduced exactly enough to serve as a stable endpoint target.

## Tasks

1. Build the specimen freezer from `04_TRACK1_DATA_CONTRACT.md`.
2. Convert WT and W0 to canonical safetensors without altering values.
3. Freeze architecture, tokenizer, corpus, training, and tensor manifests.
4. Create `D_genome_fit`, `D_fingerprint`, `D_probe`, `D_verifier_hidden`, and `P_generation`.
5. Save R0 outputs on hidden verifier anchors.
6. Measure final-window noise if checkpoints exist.
7. Build hashes and an immutable `R0_specimen_id`.

## Tests

- load the canonical checkpoint and reproduce final validation loss;
- compare original and canonical logits on anchors;
- verify every state-dict key, shape, dtype, tie, and checksum;
- deliberately corrupt one tensor and prove the evaluator detects degradation.

## Gate P0

Pass only when R0 can be reconstructed and evaluated with no unexplained differences.

---

# Phase 1 — evaluator and functional sensitivity map

## Hypothesis

A small number of tensor families or directions may dominate the functional sensitivity of R0, allowing genome bits to be allocated intelligently.

## Experiments

### 1.1 Whole-model perturbation calibration

Apply controlled perturbations to WT:

- Gaussian noise scaled by tensor norm;
- uniform quantization at 8/6/4/3/2 bits;
- random coordinate zeroing;
- low-rank truncation of Delta-T;
- interpolation between W0 and WT.

Measure the complete evaluator response.

### 1.2 Tensor-family replacement

For diagnosis only, construct hybrids:

- W0 embeddings + WT body;
- WT embeddings + W0 body;
- one W0 layer inside WT;
- one WT layer inside W0;
- reset norms only;
- reset attention only;
- reset MLP only.

This yields a rough sensitivity ordering.

### 1.3 Delta energy map

For each tensor:

\[
e_m=\frac{\|\Delta_m\|_F^2}{\sum_j\|\Delta_j\|_F^2}.
\]

Compare delta energy with functional sensitivity. High energy is not automatically high importance.

### 1.4 Gradient-weighted sensitivity

On `D_probe`, calculate:

\[
s_m=\left|\left\langle\nabla_{W_m}L_P(W_T),\Delta_m\right\rangle\right|
\]

and blockwise/low-rank approximations. The endpoint gradient may be small; also measure sensitivity at compressed/corrupted candidates.

## Gate P1

Evaluator produces stable, monotonic-enough diagnostics and identifies which perturbations correspond to meaningful quality changes.

## Definition of done

A report ranks tensor roles by:

- bytes;
- delta energy;
- quantization sensitivity;
- replacement sensitivity;
- repair leverage.

---

# Phase 2 — classical rate–distortion audit of Delta-T

## Hypothesis

Delta-T has a shorter useful structured description than WT and can preserve R0 function at a meaningful byte fraction.

## Mandatory baselines

### B0: raw checkpoint

- original WT bytes;
- zipped/zstd-compressed WT bytes;
- original W0 plus full Delta-T bytes.

### B1: quantized complete WT

Evaluate standard quantization without Delta-T to establish a deployment-compression baseline.

### B2: quantized Delta-T

Regenerate/load W0 and add quantized Delta-T.

### B3: per-tensor truncated SVD

For every matrix delta:

\[
\Delta_m\approx U_{m,r}\Sigma_{m,r}V_{m,r}^\top.
\]

Test ranks:

```text
1, 2, 4, 8, 16, 32, 64
```

and energy targets:

```text
50%, 70%, 80%, 90%, 95%, 97%, 99%, 99.5%
```

Store vectors/scalars directly or quantized.

### B4: global byte-budget SVD allocation

Instead of the same rank everywhere, allocate the next singular component with the best estimated improvement per byte. Compare:

- singular energy per byte;
- probe-loss improvement per byte;
- gradient-weighted residual per byte.

### B5: low-rank plus sparse

\[
\Delta_m\approx L_m+S_m.
\]

Test sparse residual fractions:

```text
0, 0.001%, 0.01%, 0.05%, 0.1%, 0.5%, 1%
```

Compare largest-magnitude and gradient-weighted positions.

### B6: spectral

Test DCT/FFT coefficient budgets by tensor. Compare deterministic low-frequency order with largest-magnitude coefficients.

### B7: Kronecker and tensor train

Run on large matrix families where shapes factor cleanly. These are secondary baselines but may expose structure missed by SVD.

### B8: role-shared dictionary

Learn block dictionaries separately for:

- embeddings/LM head;
- Q/K/V/O;
- MLP up/gate/down;
- norm/bias vectors.

Represent blocks by codebook indices and optional residual scales.

## Budgets

Evaluate at actual total payload ratios:

```text
0.05%, 0.1%, 0.25%, 0.5%, 1%, 2%, 5%, 10%, 20%, 40%, 80%
```

Not every method will support every exact point; interpolate only in plots, not in metric tables.

## Metrics

At every point:

- total and per-model bytes;
- validation loss/perplexity;
- hidden verifier logit KL;
- top-k agreement;
- generation suite;
- parameter error by role;
- decode seconds;
- repair-to-gate curve.

## Gate P2

Continue to neural genomes when either:

1. a structured representation reaches the R0 terminal-quality band at a useful byte fraction; or
2. no representation passes, but there is a smooth rate–distortion curve and a small repair budget closes the gap; or
3. tensor-family diagnostics reveal a clear target for a neural residual decoder.

## Failure branches

- If WT and Delta-T compress equally: initialization removal may not be the main advantage.
- If embeddings dominate failure: give them a dedicated row/code decoder or retain them at higher precision.
- If norm vectors dominate: store them directly; their byte cost is small.
- If SVD has good MSE but poor function: use functional allocation and patching.
- If all small perturbations collapse quality: inspect evaluator, dtype, tied weights, and load logic before concluding high irreducibility.

---

# Phase 3 — shared neural genome auto-decoder

## Hypothesis

A shared decoder plus compact model/layer/tensor codes can represent Delta-T more efficiently than hand-chosen tensor decompositions.

G0 has full access to WT. Genome codes are optimized directly.

## Experiment 3.1: tensor-local coordinate decoder

Train one shared coordinate/block decoder across R0 tensors.

Suggested starting configuration:

```text
block size: 16x16
model code: 64 or 128 dims
layer code: 32 or 64 dims
per-tensor code: 32, 64, or 128 dims
role embedding: 16 or 32 dims
decoder: 4-6 layer MLP, hidden 256-512, gated activation
output: normalized delta block
training dtype: fp32 or mixed precision with fp32 accumulation
```

Use role-specific normalization of Delta-T blocks.

Loss progression:

1. normalized delta reconstruction;
2. add task loss on sampled batches;
3. add reference-logit loss on `A_fit_ref`;
4. add code rate/quantization penalty.

## Experiment 3.2: hierarchical codes

Compare:

- per-tensor codes only;
- global + tensor;
- global + layer + tensor;
- global + layer + tensor + sparse block exceptions.

The hierarchy is useful only if it lowers total bytes at equal function.

## Experiment 3.3: base-weight conditioning

Compare decoder with and without compressed W0 block features.

## Experiment 3.4: hybrid decoder

Decode:

\[
\widehat\Delta=L_{SVD}+D_\psi(z)+S.
\]

Use the neural field for the residual left by a cheap structured component. This may be easier than asking it to learn all scales at once.

## Experiment 3.5: code quantization

Test fp16, int8, vector-quantized, and entropy-coded codes. Train with quantization-aware noise.

## Essential anti-memorization tests

Even for G0:

- hold out random blocks during some training passes and evaluate their reconstruction;
- train the interpreter across multiple trajectory checkpoints with checkpoint-specific codes;
- compare decoder parameter count with WT size;
- train a same-size decoder on randomized/permuted Delta-T as a negative control;
- ensure target-specific codes, not hidden target-specific decoder branches, carry endpoint identity.

## Gate P3

The neural/hybrid genome must improve the functional rate–distortion frontier or latent-refinement speed after counting the decoder.

A decoder that merely replaces a 100 MB checkpoint with a 150 MB target-specific network has not succeeded.

---

# Phase 4 — latent-space optimization and compile-and-polish

## Hypothesis

Even if a genome cannot be predicted exactly, optimizing a low-dimensional genome may reach matched quality substantially faster than updating all 50M child parameters.

## Experiment 4.1: recover fitted genome from corruption

1. Take a successful G0 genome.
2. Quantize/noise/drop portions of its codes.
3. Optimize only codes on `D_probe`.
4. Measure steps/tokens/FLOPs/seconds to recover gate quality.

## Experiment 4.2: random or mean-code start

Start from:

- zero codes;
- mean codes across trajectory checkpoints;
- early-checkpoint code;
- compiler prediction later.

Compare latent optimization against:

- full-weight AdamW repair;
- LoRA repair at matched trainable parameter count;
- low-rank Delta-T factor optimization;
- continued Track 1 training.

## Experiment 4.3: multiscale latent refinement

Optimize in order:

1. global code;
2. layer codes;
3. tensor codes;
4. sparse/low-rank patch.

This tests whether broad function can be corrected before local details.

## Experiment 4.4: learned linker

Only after direct latent refinement works, generate training pairs:

```text
corrupted genome + probe evidence -> correction toward successful genome
```

Train a small linker and compare it with direct gradient steps.

## Gate P4

Latent repair reaches matched quality with lower total child compute or wall-clock than the strongest declared repair baseline.

---

# Phase 5 — weight language and self-supervised pretraining

## Hypothesis

Model checkpoints contain recurring local and global motifs that can be tokenized and learned without natural language.

This phase is optional before G1 if the direct compiler can train from fitted codes.

## Tokenization

1. Canonicalize/alignment where required.
2. Convert Delta-W matrices to normalized blocks.
3. Learn a vector-quantized codebook.
4. Represent a checkpoint as architecture tokens plus block tokens and scales.
5. Optionally learn BPE-style merges over recurring block-token patterns.

## Pretraining tasks

- masked block token prediction;
- missing tensor completion;
- missing layer completion;
- next-checkpoint prediction;
- endpoint-token prediction;
- tensor role/layer prediction;
- checkpoint order prediction;
- performance-bin prediction;
- same-run versus different-run classification;
- corruption-to-repair token generation;
- function-signature prediction.

## Negative controls

- shuffled blocks;
- randomized role labels;
- randomized checkpoint order;
- random matrices matched in marginal distribution.

A useful weight language model should fail these controls rather than merely model scalar histograms.

---

# Phase 6 — same-run endpoint compiler, G1

## Hypothesis

A limited prefix of the R0 training process contains enough information to predict a useful endpoint genome.

## Target representation

Use a successful fitted G0 genome as one supervised target, but account for non-uniqueness. Better options:

- fit multiple genomes from different initial codes and train against a distribution;
- train through the frozen interpreter using functional loss;
- predict a coarse genome, then refine on `D_probe`.

## Inputs

Test cumulatively:

1. architecture + W0 only;
2. add static dataset fingerprint;
3. add initial gradient sketches;
4. add 0.1% trajectory;
5. 0.25%;
6. 0.5%;
7. 1%;
8. 2%;
9. 5%;
10. 10%;
11. 20%.

## Training examples from one trajectory

Use windows such as:

\[
(W_{t_0},\ldots,W_{t_k})\to p_T
\]

but avoid pretending they are independent. Vary:

- observed checkpoint cadence;
- masked tensor families;
- corrupted/noisy observations;
- prefix endpoint target, such as predicting later intermediate genomes as well as WT;
- task fingerprint batches.

## Compiler losses

- genome-code regression after canonical code matching;
- decoded Delta-T loss;
- task loss through frozen interpreter;
- logit/hidden agreement on fit anchors;
- uncertainty calibration;
- rate and repair cost.

## Inference variants

- one-shot pure compile;
- compile + direct latent refinement;
- compile + learned linker;
- best-of-N genome sampling + allowed probe selection;
- compile + low-rank/sparse patch.

## Gate P6

A G1 candidate generated without access to WT at inference must beat a matched-information baseline:

- continue ordinary training from the same prefix;
- Track 1 transport method from the same prefix;
- fit/repair a generic genome from the same prefix;
- simple linear extrapolation in successful G0 latent space.

Report both total compute and wall-clock.

---

# Phase 7 — sibling runs and held-out compilation, G2

## Hypothesis

The compiler learns transferable structure rather than memorizing R0.

## Dataset design

Create run-level train/development/hidden splits. Hold the hidden endpoint and all its later checkpoints out of training.

Recommended controlled gates:

1. hidden data order, same W0;
2. hidden W0 seed, same data order;
3. both seed and order hidden;
4. small corpus-mixture shift.

The first two isolate different sources of endpoint variation.

## Cross-run handling

Compare:

- function-space targets only;
- activation-signature alignment;
- verified head/MLP-channel re-basing;
- transformer-equivariant neural functional encoder;
- distributional genome generation.

Do not average raw endpoints before passing an alignment/function test.

## Gate P7

The hidden run reaches the predeclared quality band with less total child compute than ordinary training from the same information state.

The target run's endpoint cannot influence model selection, thresholds, or interpreter fine-tuning.

---

# Phase 8 — genome-native child architecture

## Hypothesis

A conventional dense transformer may have more endpoint degrees of freedom than GENOME can compactly infer. Co-designing the child to be generated from a compact genome may make endpoint compilation substantially easier.

## Child parameterization

Replace free matrices with:

\[
W_l=W_{0,l}+D_\psi(z_g,z_l,z_{tensor},coordinates)+S_l.
\]

Train the child from scratch by updating genome codes and possibly the shared decoder, not 50M independent scalars.

Experiments:

- fixed random substrate + low-rank generated modulation;
- recurrent/shared-depth decoder with layer controls;
- generated MLP/attention matrices, directly stored embeddings/norms;
- progressive release of sparse exception parameters;
- nGPT/normalized-vector or other trajectory-friendly variants only after baseline evidence.

Compare final quality, code size, training stability, and compile predictability against the standard Track 1 model.

A genome-native architecture is a new child architecture and must not replace the R0 baseline retroactively.

---

# Phase 9 — self-hosting, G4

Proceed only after G2 or a strong genome-native result. The complete self-hosting loop is specified in `09_RSI_AND_FUTURE_RESEARCH.md`.

---

# Default experiment matrix

Keep the first pass small:

| Axis | Initial values |
| --- | --- |
| Base | WT direct, W0 + Delta-T |
| Block size | 16, 32 |
| Tensor formula | quantized, SVD, SVD+sparse, spectral, neural block field, hybrid |
| Tensor code dim | 32, 64, 128 |
| Layer code dim | 0, 32, 64 |
| Global code dim | 0, 64, 128 |
| Code dtype | fp16, int8 |
| Patch | none, low-rank, sparse |
| Loss | Delta only, Delta+task, Delta+task+logit |
| Prefix | 0, 0.5%, 1%, 2%, 5%, 10% |

Do not run the full Cartesian product. Use the next experiment that most clearly separates competing explanations.

---

# Recommended decision tree

```text
Can R0 be reproduced? -- no --> fix specimen/evaluator
          |
         yes
          v
Does Delta-T show structured rate-distortion? -- no --> test function-space/role sensitivity
          |                                      |
         yes                                     v
          |                              Is small repair effective?
          v                                      |
Does neural/hybrid decoder beat baselines?        +-- yes --> compile-and-polish path
          |                                      +-- no  --> genome-native architecture branch
    no ---+--> keep transparent codec and test latent optimization
          |
         yes
          v
Can low-dimensional codes be optimized quickly? -- no --> redesign representation
          |
         yes
          v
Can early trajectory predict codes? -- no --> richer fingerprint / time-field model
          |
         yes
          v
Does it transfer to a hidden run? -- no --> alignment/distribution/run diversity
          |
         yes
          v
Architecture transfer and self-hosting
```

---

# Experiment completion rule

Do not declare a phase complete because a training loss decreased. A phase is complete only when:

- candidate MGPs are frozen;
- byte accounting is final;
- hidden Genome Gate evaluation has run once per frozen candidate set;
- results include baselines and failure cases;
- the decision log states the next branch and why.
