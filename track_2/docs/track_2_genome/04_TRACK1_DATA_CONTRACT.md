# Track 1 data and artifact contract for GENOME

> **Archived pre-recovery note.** Track 1 is legacy G0 and future evaluation-only. It is not an
> active public-life compiler-training target.

## 1. Relationship to Track 1

Track 1 produces the first controlled training organism. Its fully trained poetry model becomes **R0**, the reference phenotype for GENOME.

Track 1 and Track 2 answer different questions:

- Track 1 studies whether an observed training path can be traversed faster.
- Track 2 studies whether the endpoint/function can be encoded and eventually compiled without replaying most of that path.

The bundled Track 1 map is at [`reference/track_1_research_map.md`](reference/track_1_research_map.md). GENOME must preserve its distinctions between same-run trajectory compression, held-out-run acceleration, and general transfer.

---

## 2. Required R0 specimen

The specimen freezer must produce or locate the following.

### Absolutely required for G0

1. Final R0 model checkpoint, WT.
2. Exact model implementation or a stable construction function.
3. Architecture/configuration manifest.
4. Tokenizer files and special-token definitions.
5. Validation/evaluation data that reproduces the reported final metrics.
6. Training corpus provenance and split hashes.
7. Tensor inventory with names, shapes, dtypes, and tied groups.
8. The final Track 1 validation loss/perplexity and fixed prompt behaviour.

### Strongly required

1. Initial checkpoint W0.
2. Initialization seed and initialization procedure.
3. Exact training configuration and data order.
4. Final-window checkpoints and metrics.
5. Periodic trajectory checkpoints from early and middle training.
6. Optimizer state summaries or complete states at selected checkpoints.
7. Existing fixed anchors or enough information to recreate them.

### Useful later

1. Multiple independent Track 1 runs.
2. Alternative seeds with the same data order.
3. Alternative data orders with the same seed.
4. Curriculum/objective variants.
5. Width/depth variants.
6. Failed or unstable runs.

Failures are valuable training examples for performance prediction and genome gating.

---

## 3. Base-state hierarchy

Use the strongest available base mode and report it.

### Mode A: exact seed replay

Regenerate W0 from:

```text
architecture code and config
framework version
initialization functions and parameters
master seed and per-device seeds
PRNG algorithm
state-dict/tensor creation order
precision and device used during initialization
```

Verify every tensor checksum against a saved W0 checkpoint.

### Mode B: saved W0 shared base

Store W0 once as a shared artifact and encode Delta-T. This is valid for testing conditional compression and the genome representation.

### Mode C: earliest available checkpoint

If W0 is unavailable and cannot be reconstructed, use the earliest checkpoint \(W_{t_0}\):

\[
\Delta_{t_0\to T}=W_T-W_{t_0}.
\]

This result must be labelled **partial-trajectory endpoint representation**, not full initialization-to-endpoint generation.

### Mode D: no base

Encode WT directly. Keep this only as a baseline. It wastes capacity on initialization entropy and is not the preferred GENOME formulation.

---

## 4. Frozen architecture manifest

Example:

```json
{
  "model_family": "track1_poetry_decoder",
  "implementation": "module.path:ModelClass",
  "parameter_count": 50000000,
  "vocab_size": 0,
  "context_length": 0,
  "n_layers": 0,
  "hidden_size": 0,
  "n_heads": 0,
  "head_dim": 0,
  "mlp_size": 0,
  "activation": "...",
  "normalization": "...",
  "position_encoding": "...",
  "tie_embeddings": true,
  "bias_policy": "...",
  "dtype": "...",
  "tensor_inventory_file": "tensor_inventory.json",
  "source_commit": "..."
}
```

Populate from the real project; do not infer values from the approximate 50M label.

The manifest hash is embedded in every MGP and evaluation report.

---

## 5. Tensor inventory

For every state-dict entry, store:

```json
{
  "canonical_index": 17,
  "name": "blocks.2.attn.q_proj.weight",
  "role": "q_proj",
  "layer_index": 2,
  "shape": [512, 512],
  "dtype": "bfloat16",
  "numel": 262144,
  "bytes": 524288,
  "requires_grad": true,
  "tied_group": null,
  "initialization": {"family": "normal", "std": 0.02},
  "state_sha256_W0": "...",
  "state_sha256_WT": "..."
}
```

Semantic roles must be assigned explicitly. Do not rely on substring matching inside the decoder after the inventory is frozen.

Suggested roles:

```text
embedding
attention_norm
q_proj
k_proj
v_proj
o_proj
mlp_norm
gate_proj
up_proj
down_proj
final_norm
lm_head
bias
other
```

---

## 6. Checkpoint trajectory index

Store one JSONL row per checkpoint:

```json
{
  "run_id": "track1_R0",
  "checkpoint_id": "step_001000",
  "step": 1000,
  "tokens_seen": 12345678,
  "fraction_of_final_tokens": 0.03125,
  "checkpoint_path": "...",
  "checkpoint_sha256": "...",
  "optimizer_path": "...",
  "train_loss": 0.0,
  "validation_loss": 0.0,
  "learning_rate": 0.0,
  "wall_seconds": 0.0,
  "gpu_seconds": 0.0,
  "data_cursor": "...",
  "rng_state_hash": "..."
}
```

Use `tokens_seen` as the primary training-progress coordinate. Steps can be misleading when batch size, accumulation, or sequence length changes.

---

## 7. Terminal reference window

Do not define endpoint equivalence using a single noisy scalar if Track 1 saved final-window checkpoints.

Let the last \(J\) accepted checkpoints be \(W_{T-J+1},\ldots,W_T\). Compute:

\[
\mu_L=\frac1J\sum_j L_{verify}(W_j),
\qquad
\sigma_L=\operatorname{std}_j L_{verify}(W_j).
\]

Also compute terminal distributions for:

- anchor-logit KL between adjacent checkpoints;
- generation metrics;
- hidden-state drift;
- layerwise parameter movement.

This terminal noise defines a meaningful matched-quality band. If no terminal window exists, create a fixed tolerance and label it as an engineering threshold rather than empirical run noise.

---

## 8. Data partitions for Track 2

Track 2 needs more than the original train/validation split. Derive immutable subsets without violating document/poem boundaries.

### 8.1 `D_genome_fit`

May be used to optimize G0 genome codes/interpreter and train G1/G2 compilers. It may contain training examples from the original Track 1 train split.

### 8.2 `D_fingerprint`

Used to compute dataset/task fingerprints. It must be deterministic and declared. For a new run, fingerprinting this data is part of child-generation compute.

### 8.3 `D_probe`

Small set used for candidate selection, direct latent refinement, linker evidence, and repair. It must not overlap the hidden verifier at poem/document level.

### 8.4 `D_verifier_hidden`

Immutable hidden set used only by Genome Gate. It cannot be used to:

- train the interpreter/compiler;
- optimize genome codes;
- choose among candidates;
- tune thresholds;
- decide patch allocation;
- stop repair.

### 8.5 `P_generation`

Fixed conditional-prompt suite for poetry generation. Include:

- prompts sampled from held-out themes;
- manually authored prompts;
- short and long prompts;
- concrete and abstract prompts;
- prompts requiring the project’s `<PROMPT>`, `<THOUGHT>`, and `<POEM>` modes as applicable;
- adversarial repetition and mode-transition checks.

Split all data at the original document/poem level and preserve provenance.

---

## 9. Reference-output sets and leakage modes

Reference outputs from R0 are useful but must be labelled by how they are used.

### `A_fit_ref`

R0 logits/hidden states usable during G0 representation fitting. This is allowed because G0 explicitly has access to WT.

### `A_probe_ref`

R0 outputs on the allowed probe set. Using these at candidate inference creates **endpoint function distillation**.

Use separate labels:

- **G1-Pure:** no WT or final R0 outputs are available when compiling/refining the candidate; only corpus/task loss and early evidence are allowed.
- **G1-Distill:** final R0 outputs on declared probe examples are available at inference. This may be valuable but is a weaker information setting.

### `A_verify_ref`

R0 outputs on the hidden verifier. These are available only to Genome Gate for final scoring.

Never silently use `A_probe_ref` and call the result pure endpoint prediction.

---

## 10. R0 preprocessing products

For W0, WT, and each trajectory checkpoint selected for study, compute:

### Per tensor

- Delta from W0;
- raw and delta norms;
- update-to-base norm ratio;
- mean, variance, min, max;
- row and column norm quantiles;
- singular values and cumulative energy for matrices;
- effective rank;
- DCT/FFT energy curves;
- quantization error at 8/6/4/3/2 bits where implemented;
- entropy/compressed bytes of raw and delta representations;
- non-finite count;
- checksum.

### Per model

- total parameter and byte count;
- fraction of delta energy by tensor role/layer;
- global delta norm;
- cosine between adjacent checkpoint deltas;
- validation loss/perplexity;
- fixed anchor outputs;
- fixed generation outputs with seeds.

Store derived arrays separately from immutable checkpoints.

---

## 11. Corpus of model lives

The long-term training unit is not only a checkpoint. It is a **model life**:

```text
architecture graph
initialization/base state
training corpus and data order
optimizer and schedule
checkpoint trajectory
optimizer-state trajectory
loss/metric history
gradient sketches
activation summaries
anchor outputs
final endpoint and quality
compute and wall-clock records
```

A single life yields many self-supervised tasks:

- masked block reconstruction;
- missing tensor reconstruction;
- checkpoint t to checkpoint t+k;
- early prefix to endpoint;
- corrupted genome to repaired genome;
- tensor role prediction;
- endpoint quality prediction;
- function-signature prediction;
- detect whether two checkpoints share a run;
- predict whether a jump will pass verification.

However, many slices from one life are not independent runs. Run-level splits are mandatory for G2 claims.

---

## 12. Minimum data by research level

### G0

One endpoint plus W0 is sufficient.

Recommended additional examples:

- trajectory checkpoints from R0;
- masked blocks/tensors;
- synthetic corruptions;
- multiple fitted genomes at different budgets.

### G1

One complete trajectory can train a same-run endpoint completion system, but overfitting risk is extreme. Use trajectory windows, block masking, and synthetic corruptions. Be explicit that the endpoint lineage is known.

### G2

At least one target run must be withheld by run ID. A practical initial collection is:

- several sibling runs used for training;
- one development run for threshold/configuration choices;
- one untouched hidden run for the result.

More independent runs are better than denser checkpoint saving from one run.

### G3

Hold out at least one material change: corpus mix, objective, width, depth, or architecture family.

### G4

The unit is a compiler generation with hidden child-model tasks and an immutable evaluator.

---

## 13. Cheap sibling-run strategy

If full 50M sibling runs are too expensive, create a hierarchy of model organisms:

1. **Micro organisms:** 0.5M–3M parameters, many seeds and corpus subsets.
2. **Small organisms:** 3M–15M parameters, moderate number of runs.
3. **Track-scale organisms:** the 50M Track 1 architecture, few carefully selected runs.

Keep architecture ratios and tensor roles as similar as practical. The compiler can pretrain on smaller lives, then adapt to the 50M lineage.

Useful controlled variations:

- initialization seed only;
- data order only;
- poetry/prose mix;
- learning-rate multiplier;
- optimizer family;
- context length;
- width/depth;
- tokenizer variant only after same-tokenizer experiments.

Do not change several variables simultaneously in the first sibling dataset.

---

## 14. External checkpoint corpora

External data can pretrain the weight encoder/interpreter’s general grammar, but it does not replace Track 1-specific data.

Potential sources include:

- Pythia and PolyPythias trajectories;
- the Transformer-NFN checkpoint dataset;
- G.pt checkpoint datasets for small architectures;
- SANE/model-zoo datasets;
- public hypernetwork/weight-generation model zoos.

Use external corpora for:

- block autoencoding;
- weight-token codebooks;
- tensor-role representations;
- trajectory-order prediction;
- general checkpoint quality prediction;
- symmetry-aware pretraining.

Do not claim direct poetry endpoint knowledge from unrelated checkpoints.

Every external source needs a license/provenance record and conversion manifest.

---

## 15. Data augmentation in weight space

### Safe within-run augmentations

- additive quantization noise;
- block masking;
- low-rank truncation;
- spectral coefficient dropping;
- sparse corruption;
- latent-code corruption;
- interpolation between adjacent same-run checkpoints, evaluated rather than assumed valid;
- random selection of trajectory windows.

### Function-preserving symmetry augmentation

Later stages may apply verified head/channel permutations and other exact symmetries. Every transform must pass:

\[
\max_{x\in A}\|f_{T(W)}(x)-f_W(x)\|\le\epsilon_{numeric}.
\]

If the check fails, the augmentation is invalid.

### Prohibited pseudo-augmentation

Do not arbitrarily permute scalar coordinates, rows, columns, heads, or layers and assume the model function is preserved.

---

## 16. Run-level split schema

```json
{
  "split_version": "1",
  "train_run_ids": ["micro_0001", "micro_0002"],
  "development_run_ids": ["small_0001"],
  "hidden_run_ids": ["track50m_seed_hidden"],
  "rules": {
    "checkpoint_slices_inherit_run_split": true,
    "no_endpoint_from_hidden_run_in_training": true,
    "no_hidden_verifier_access": true
  },
  "sha256": "..."
}
```

A checkpoint from a hidden run remains hidden even if it is early. The only allowed hidden-run inputs are those explicitly granted to the compiler at inference, such as W0, data fingerprint, or a predeclared early prefix.

---

## 17. Storage and precision policy

- Preserve immutable checkpoints in their original effective dtype and optionally fp32-converted analysis copies.
- Compute Delta-T in fp32 or fp64 to avoid subtraction artifacts, then evaluate target output dtypes separately.
- Use safetensors for model arrays when practical.
- Store metrics in Parquet/JSONL.
- Compress archival copies, but report uncompressed logical bytes and actual archive bytes separately.
- Never delete source checkpoints after deriving deltas without a verified backup.
- Record tensor endianness and dtype conventions for exact residual tests.

---

## 18. Preflight checklist

Before Phase 1 experiments, answer yes to all:

- [ ] R0 loads without warnings or missing/unexpected keys.
- [ ] R0 reproduces its frozen final validation result.
- [ ] W0 exists or the fallback base mode is declared.
- [ ] `WT - W0` can be computed for every trainable tensor.
- [ ] Tied weights are identified and verified.
- [ ] Tokenizer and special tokens are hashed.
- [ ] Corpus/document split hashes are recorded.
- [ ] `D_probe` and `D_verifier_hidden` do not overlap.
- [ ] R0 reference outputs are saved for the hidden verifier.
- [ ] Tensor inventory order is stable across repeated construction.
- [ ] Every immutable artifact has a checksum.
- [ ] The evaluator can score an unmodified copy of R0 and a deliberately corrupted copy.
