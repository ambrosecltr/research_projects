# GENOME system architecture

> **Archived pre-recovery note.** The learned-interpreter architecture in this file is not active.
> Use `../../RECOVERY.md` and `00_README.md`.

## 1. End-to-end pipeline

```text
Track 1 artifacts
      |
      v
[Specimen Freezer] ---> immutable R0/W0 manifests and evaluator assets
      |
      +--------------------+
      |                    |
      v                    v
[Architecture Graph]  [Data/Trajectory Fingerprinter]
      |                    |
      +---------+----------+
                v
         [Genome Compiler]
                |
                v
       initial Model Genome p0
                |
                v
      [Genome Interpreter]
                |
                v
        candidate phenotype
                |
                v
         [Allowed Prober]
                |
                v
     [Latent Linker / Refiner]
                |
                +---- repeat K times ----+
                v                        |
       [Patch Allocator] <---------------+
                |
                v
       frozen candidate MGP
                |
                v
         [Genome Gate]
                |
        accept / reject / diagnose
```

The system is deliberately modular. G0 uses the freezer, interpreter, codecs, prober, and gate without a learned compiler. G1 adds an early-trajectory compiler. G2 adds multiple independent runs and alignment/equivariance requirements.

---

## 2. Component A: Track 1 specimen freezer

The freezer converts the Track 1 project into an immutable model specimen.

Outputs:

```text
specimen/R0/
  architecture.json
  tensor_inventory.json
  initialization.json
  training_recipe.json
  corpus_manifest.json
  tokenizer_manifest.json
  final_checkpoint.safetensors
  initial_checkpoint.safetensors       # if retained
  checkpoints/index.jsonl              # optional trajectory
  optimizer_states/index.jsonl         # optional
  metrics/history.parquet
  anchors/probe_ids.json
  anchors/verifier_ids.json
  anchors/reference_outputs/
  hashes.json
```

Responsibilities:

- map model code into a stable architecture/tensor manifest;
- capture W0 or enough information to regenerate it;
- save WT in a non-pickle canonical form;
- identify tied tensors;
- hash all artifacts;
- reproduce final Track 1 validation metrics;
- create immutable data splits for genome fitting, probing, and verification;
- fail if the reconstructed R0 does not match the original.

No genome work should proceed until the freezer passes its reproducibility tests.

---

## 3. Component B: architecture graph

The architecture graph describes functional relationships, not merely state-dict order.

### Node types

Use two linked levels:

1. **Operation nodes:** embedding, normalization, attention, MLP, residual addition, output head.
2. **Tensor nodes:** each trainable parameter tensor.

### Tensor-node features

- semantic role;
- layer index and normalized depth;
- shape, rank, and log dimensions;
- fan-in/fan-out;
- dtype;
- initialization family and scale;
- tied-group ID;
- parameter count;
- base and delta norms when allowed;
- whether the tensor is matrix/vector/scalar;
- attention head count and head dimension where applicable;
- MLP expansion ratio;
- input/output activation stream IDs.

### Edge types

- tensor parameterizes operation;
- operation consumes/produces activation stream;
- residual connection;
- tensor is paired with another tensor, such as Q/K or up/down;
- tied/aliased;
- adjacent depth;
- same semantic role across layers;
- same attention head group;
- base-to-output dependency.

The first implementation may use a fixed graph for the Track 1 architecture. Build a correct graph before making it architecture-generic.

---

## 4. Component C: weight specimen encoder

The encoder converts weights or deltas into manageable tokens/features.

### Initial deterministic features

For each tensor:

- Frobenius/L2/L1/L-infinity norms;
- mean, variance, skew proxy, kurtosis proxy;
- row/column norm distributions;
- top singular values and explained energy;
- effective rank;
- update-to-base norm ratio;
- cosine with recent deltas;
- blockwise means/variances;
- quantization entropy estimates;
- spectral coefficient energy curves.

These are useful even before a learned encoder exists.

### Patch tokenization

For matrix tensors:

1. pad to a multiple of block size;
2. divide into \(b_r\times b_c\) blocks, initially 16×16 or 32×32;
3. normalize using declared tensor-role statistics;
4. flatten/project each block into a token;
5. add tensor, layer, role, row-block, and column-block embeddings.

Do not concatenate all scalar weights into one context.

### Learned weight encoder options

- small block transformer;
- SANE-style sequential subset encoder;
- graph neural functional network;
- role-specific CNN over matrix patches;
- invariant/equivariant transformer for cross-run stages.

G0 does not require a learned encoder if genome codes are optimized directly.

---

## 5. Component D: dataset and trajectory fingerprinting

The compiler cannot generate corpus-specific learning without corpus evidence. The fingerprinting service creates compact evidence while keeping the hidden verifier isolated.

### Static dataset fingerprint

Input: deterministic batches from a permitted fingerprint split.

Channels:

- token/byte frequency sketches;
- sequence-length and packing statistics;
- conditional-mode ratios such as prompt/thought/poem/prose;
- per-batch loss under W0 or an early checkpoint;
- random-projected per-tensor gradients;
- gradient norm and covariance spectra by tensor role;
- activation mean, variance, quantiles, and covariance sketches;
- residual-stream and attention entropy statistics;
- Fisher-diagonal or empirical-Fisher summaries;
- optional learned set encoder over token sequences.

### Early trajectory fingerprint

Input: selected checkpoints from the first fraction of training.

Channels:

- encoded Delta-W at each observed time;
- finite differences and acceleration estimates;
- optimizer moment summaries;
- training/validation loss slopes;
- layerwise update norms;
- singular-value trajectories;
- anchor-logit trajectories;
- activation-space regime signatures.

### Fingerprint budgets

The fingerprint itself consumes compute and must be budgeted. Every experiment declares:

- number of batches/tokens;
- number of forward/backward passes;
- storage bytes;
- elapsed time and FLOPs;
- fraction of ordinary training cost.

---

## 6. Component E: genome planner/compiler

The compiler maps conditioning evidence to genome codes and formula choices.

### Fixed-architecture G1 compiler

For the first learned compiler, the architecture is constant. Use a modest model:

1. encode dataset/trajectory evidence into \(z_D\);
2. create one token per tensor using the frozen tensor inventory;
3. add role/layer/shape embeddings;
4. run bidirectional transformer or graph-attention blocks;
5. emit global/layer/tensor codes and optional formula gates.

Conceptually:

\[
H_0=\operatorname{TensorTokens}(\mathcal G)+\operatorname{Broadcast}(z_D,z_\tau),
\]

\[
H_{k+1}=\operatorname{GraphTransformer}_k(H_k,E_\mathcal G),
\]

\[
p=\operatorname{Heads}(H_K).
\]

### Outputs

- global code;
- per-layer code;
- per-tensor code;
- optional code uncertainty;
- optional mixture component;
- optional primitive/opcode probabilities;
- patch budget allocation;
- predicted endpoint metrics for calibration.

### Avoiding target averaging

If multiple good endpoints are incompatible, use:

- mixture-of-experts heads;
- latent noise sampled once per candidate;
- flow/diffusion in compact genome space;
- best-of-N sampling selected only by the allowed probe.

Do not begin with checkpoint-space diffusion. If a generative model is needed, generate compact genomes.

---

## 7. Component F: shared genome interpreter

The interpreter expands codes into Delta-W tensors.

### Recommended first neural interpreter

A role-conditioned block decoder:

```text
inputs:
  global code
  layer code
  tensor code
  tensor role embedding
  normalized layer depth
  tensor shape embedding
  block row/column coordinates
  optional compressed W0 block features

network:
  FiLM-conditioned MLP or small transformer

output:
  one 16x16 or 32x32 delta block
```

The decoder is shared across layers and, where practical, across tensor roles. Small role-specific output heads are acceptable and must be counted.

### Why block decoding

- far shorter sequence than scalar prediction;
- efficient batching;
- local spatial correlations within matrices;
- streamable reconstruction;
- simple codebook and residual integration;
- natural patch-level loss and sensitivity analysis.

### Role-specific handling

- **Embeddings/LM head:** block field plus frequency-aware or row-code branch.
- **Attention matrices:** shared attention-role decoder with Q/K/V/O identity embedding.
- **MLP matrices:** shared up/gate/down decoder.
- **Norm/bias vectors:** one-dimensional field or direct quantized storage.

### Base-weight conditioning

Two decoder modes should be compared:

\[
\widehat\Delta=D_\psi(z,coordinates,role)
\]

versus:

\[
\widehat\Delta=D_\psi(z,coordinates,role,E_0(W_0\text{ block})).
\]

Conditioning on W0 may help preserve initialization-specific endpoint directions, but it increases decode compute.

---

## 8. Component G: prober

The prober executes a candidate on a small, permitted set. It produces evidence, not the final scientific verdict.

Measurements:

- NTP loss and per-token loss histogram;
- reference-logit KL on allowed anchors;
- top-k token agreement;
- hidden-state cosine/CKA summaries;
- residual-stream norm ratios;
- attention entropy, max probability, and dead-head indicators;
- activation saturation/outlier statistics;
- projected gradients by tensor role;
- approximate sensitivity of loss to each genome code or tensor family;
- fixed-prompt generation health checks.

Probe selection must be deterministic and separate from the hidden verifier.

---

## 9. Component H: latent linker/refiner

### Stage 1: direct latent optimization

Freeze the interpreter and optimize genome codes on the probe objective:

```python
for refinement_step in range(K):
    candidate = interpreter.decode(genome_codes)
    loss = probe_objective(candidate)
    loss.backward()          # gradients flow to codes, not full child weights
    latent_optimizer.step()
```

This is still optimization, but the dimensionality may be orders of magnitude smaller than WT. Measure whether it reaches matched quality faster than ordinary full-weight repair.

### Stage 2: learned linker

Train a model to predict latent corrections from probe residuals:

\[
\Delta p=\mathcal L_\omega(p,E).
\]

Training examples can be generated by corrupting fitted genomes, decoding them, probing them, and learning the correction toward a successful genome.

### Stage 3: active probe selection

Let the linker choose the next batches or synthetic probes that maximally reduce genome uncertainty. Count the added probe compute.

---

## 10. Component I: patch allocator

After broad structure is decoded, allocate a small budget to residual error.

Initial methods:

1. **Per-family substitution test:** temporarily replace one generated tensor family with R0 to measure maximum recoverable gap. Diagnostic only; R0 tensors are not allowed in final G1/G2 candidates.
2. **Residual SVD:** low-rank factorize WT minus decoded tensor for G0.
3. **Gradient-weighted residual:** rank residual components by projected impact on probe loss.
4. **Top-coordinate sparse patch:** store largest residual values, compared with largest gradient-weighted values.
5. **Latent-plus-patch joint optimization:** optimize patch factors and genome codes under a byte penalty.

The patch is expected to shrink as the interpreter improves.

---

## 11. Component J: Genome Gate

The gate is the immutable evaluator defined in `06_EVALUATION_PROTOCOL.md`.

It receives a frozen MGP and produces:

- validity result;
- exact byte accounting;
- decode profile;
- hidden verifier metrics;
- poetry-generation suite outputs;
- repair-to-threshold curve if repair is part of the declared experiment;
- acceptance decision and failure classification.

The compiler, linker, and candidate selector cannot query the hidden gate during training or candidate selection.

---

## 12. G0 operating mode: auto-decoder

In G0, the full endpoint is available.

```text
WT and W0
   |
   v
optimize genome codes + interpreter to reconstruct Delta-T/function
   |
   v
quantize genome
   |
   v
add optional patch
   |
   v
Genome Gate
```

G0 establishes representability and rate–distortion. It does not establish prediction.

Recommended training pattern:

- train shared interpreter plus codes over many R0 trajectory checkpoints or tensor subsets;
- hold out blocks/tensors/checkpoints when testing whether the interpreter generalizes internally;
- test code budgets and quantization;
- compare with non-neural codecs.

---

## 13. G1 operating mode: same-run compiler

```text
early R0 evidence -> compiler -> initial genome -> decode -> probe/link -> frozen MGP -> hidden gate
```

The final WT is withheld from the compiler at inference, but it may have been used as supervised training signal during compiler development. This is endpoint distillation from one known run, not independent-run transfer.

Useful training targets:

- fitted G0 genome codes;
- direct functional loss through the interpreter;
- residual correction from early checkpoints;
- distribution over genomes from multiple fitted solutions.

---

## 14. G2 operating mode: held-out run

Train on sibling runs and withhold the target endpoint entirely.

Required changes:

- independent initializations/data orders;
- cross-run symmetry handling;
- architecture graph and base-state conditioning;
- compiler output distribution rather than one averaged endpoint;
- target-run probe and hidden verifier separation;
- no target-specific interpreter fine-tuning unless counted as repair and declared.

A target-specific latent refinement on permitted data is allowed as compile-and-polish, but it must be included in child cost.

---

## 15. Failure localization architecture

Implement a candidate surgery harness:

- generated all tensors;
- R0 embeddings + generated body;
- generated embeddings + R0 body;
- one R0 layer at a time;
- one generated layer at a time;
- R0 norms only;
- R0 attention only;
- R0 MLP only;
- generated model with exact sparse residual at increasing budgets.

These hybrids are diagnostics and must not be reported as final generated candidates. They reveal which tensor families dominate functional failure.

---

## 16. Memory and execution strategy

For a 50M child, full materialization is manageable, but build the interpreter to support future scale.

Modes:

1. **Eager decode:** materialize complete state dict, then load model.
2. **Layer streaming:** decode one layer to CPU/GPU, execute or save, then release intermediate decoder buffers.
3. **Generate-and-cache:** expand tensors once at model load and cache standard weights.
4. **On-demand execution:** decode immediately before each layer execution; later research only.

Profile decode separately from inference. A tiny stored genome with extremely expensive decode may not be useful.

---

## 17. Artifact integrity

All generated artifacts should have SHA-256 hashes and parent links:

```text
specimen hash -> interpreter hash -> compiler hash -> MGP hash -> evaluation report hash
```

The evaluation report records the exact MGP hash. Changing a single byte creates a new candidate ID.

Do not store arbitrary executable Python objects in research artifacts. Prefer JSON, safetensors, Parquet, NPZ without pickle, and compressed binary streams with documented codecs.
