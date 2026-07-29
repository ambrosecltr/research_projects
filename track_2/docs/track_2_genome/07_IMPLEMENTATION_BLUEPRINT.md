# GENOME implementation blueprint

> **Archived pre-recovery note.** Do not implement the auto-decoder or learned-interpreter plan in
> this file. Use the active code map and recovery plan.

## 1. Suggested repository layout

Place the documentation pack at the project root or under `docs/track_2_genome/`. Suggested code layout:

```text
track_2_genome/
  README.md
  configs/
    specimen.yaml
    evaluation.yaml
    codecs/
      svd.yaml
      spectral.yaml
      hybrid.yaml
    autodecoder.yaml
    compiler.yaml
  genome/
    __init__.py
    types.py
    hashing.py
    io.py
    specimen.py
    tensor_inventory.py
    architecture_graph.py
    splits.py
    metrics.py
    evaluator.py
    sensitivity.py
    fingerprint.py
    trajectory.py
    bit_accounting.py
    mgp/
      manifest.py
      serializer.py
      interpreter.py
      validation.py
      opcodes.py
    codecs/
      base.py
      raw.py
      quantized.py
      svd.py
      lowrank_sparse.py
      spectral.py
      kronecker.py
      tensor_train.py
      codebook.py
    neural/
      block_decoder.py
      coordinate_decoder.py
      autodecoder.py
      compiler.py
      linker.py
      entropy_model.py
    repair/
      latent.py
      low_rank.py
      sparse.py
      full_weight.py
  scripts/
    freeze_track1_specimen.py
    verify_specimen.py
    analyze_delta.py
    build_sensitivity_map.py
    run_rate_distortion.py
    fit_autodecoder.py
    quantize_genome.py
    refine_genome.py
    train_compiler.py
    compile_candidate.py
    evaluate_candidate.py
    make_report.py
  tests/
    test_specimen_roundtrip.py
    test_tensor_inventory.py
    test_mgp_roundtrip.py
    test_tied_weights.py
    test_bit_accounting.py
    test_decoder_determinism.py
    test_split_leakage.py
    test_corruption_detected.py
  artifacts/
    specimens/
    interpreters/
    compilers/
    genomes/
    evaluations/
    reports/
```

Do not require this exact structure if the existing Track 1 repository has established patterns. Preserve the conceptual module boundaries.

---

## 2. Core data types

Use typed dataclasses or validation models.

### `TensorSpec`

```python
@dataclass(frozen=True)
class TensorSpec:
    canonical_index: int
    name: str
    role: str
    layer_index: int | None
    shape: tuple[int, ...]
    dtype: str
    numel: int
    nbytes: int
    tied_group: str | None
    initialization: dict[str, Any]
```

### `ModelSpecimen`

```python
@dataclass(frozen=True)
class ModelSpecimen:
    specimen_id: str
    architecture_manifest_path: Path
    tensor_inventory_path: Path
    tokenizer_manifest_path: Path
    corpus_manifest_path: Path
    training_recipe_path: Path
    base_checkpoint_path: Path | None
    final_checkpoint_path: Path
    checkpoint_index_path: Path | None
    split_manifest_path: Path
    hashes_path: Path
```

### `GenomeComponent`

```python
@dataclass
class GenomeComponent:
    opcode: str
    payload_keys: list[str]
    arguments: dict[str, Any]
```

### `TensorGenomeRecord`

```python
@dataclass
class TensorGenomeRecord:
    tensor_name: str
    base_source: str
    components: list[GenomeComponent]
    output_dtype: str
```

### `GenomeProgram`

```python
@dataclass
class GenomeProgram:
    manifest: dict[str, Any]
    payload_tensors: dict[str, Tensor]
    patch_tensors: dict[str, Tensor]
```

### `EvaluationReport`

```python
@dataclass(frozen=True)
class EvaluationReport:
    candidate_id: str
    mgp_sha256: str
    validity: dict[str, Any]
    bytes: dict[str, int]
    compute: dict[str, float]
    parameter_metrics: dict[str, Any]
    functional_metrics: dict[str, Any]
    generation_metrics: dict[str, Any]
    decision: str
    failure_codes: tuple[str, ...]
```

---

## 3. Stable interfaces

### Codec

```python
class GenomeCodec(Protocol):
    name: str

    def fit(
        self,
        base_state: Mapping[str, Tensor],
        target_state: Mapping[str, Tensor],
        tensor_specs: Sequence[TensorSpec],
        budget: GenomeBudget,
        probe: ProbeEvaluator | None,
    ) -> GenomeProgram: ...

    def decode(
        self,
        program: GenomeProgram,
        base_state: Mapping[str, Tensor],
    ) -> dict[str, Tensor]: ...
```

### Interpreter

```python
class GenomeInterpreter(nn.Module):
    def decode_tensor(
        self,
        spec: TensorSpec,
        base_tensor: Tensor,
        global_code: Tensor,
        layer_code: Tensor | None,
        tensor_code: Tensor,
        block_size: tuple[int, int],
    ) -> Tensor: ...
```

### Compiler

```python
class GenomeCompiler(nn.Module):
    def forward(
        self,
        architecture_graph: ArchitectureGraphBatch,
        dataset_fingerprint: Tensor,
        trajectory_fingerprint: Tensor | None,
        base_features: Tensor | None,
    ) -> CompilerDistribution: ...
```

### Probe evaluator

```python
class ProbeEvaluator(Protocol):
    def evaluate_state(
        self,
        state_dict: Mapping[str, Tensor],
        *,
        require_grad_sketch: bool,
    ) -> ProbeEvidence: ...
```

### Genome Gate

```python
class GenomeGate:
    def evaluate_frozen_mgp(
        self,
        mgp_path: Path,
        interpreter_path: Path,
        specimen: ModelSpecimen,
    ) -> EvaluationReport: ...
```

---

## 4. Configuration policy

Every executable command accepts a complete config file and saves the resolved config.

Example `autodecoder.yaml`:

```yaml
project: GENOME
research_level: G0
specimen_id: track1_R0
seed: 1701

base:
  mode: saved_W0

representation:
  block_rows: 16
  block_cols: 16
  global_code_dim: 128
  layer_code_dim: 64
  tensor_code_dim: 64
  code_dtype_final: int8
  structured_residual: svd
  svd_rank: 8
  sparse_patch_fraction: 0.0001

interpreter:
  hidden_dim: 512
  depth: 5
  activation: silu
  role_embedding_dim: 32
  shape_embedding_dim: 32
  condition_on_base_block: true

loss:
  delta: 1.0
  task: 0.1
  logit: 0.1
  hidden: 0.0
  gradient: 0.0
  rate: 0.0001

training:
  optimizer: adamw
  learning_rate: 0.0003
  weight_decay: 0.01
  max_updates: 0          # replace with actual decision
  mixed_precision: true
  grad_clip_norm: 1.0
  save_every: 0

evaluation:
  probe_manifest: artifacts/specimens/track1_R0/splits.json
  hidden_gate_during_training: false
```

Zero placeholders must be replaced before execution. The resolved config is hashed.

---

## 5. Phase 0 implementation sequence

### Step 0.1 — locate Track 1 entry points

Find:

- model constructor;
- tokenizer loader;
- checkpoint loader;
- dataset/evaluator constructor;
- training config and seed;
- final checkpoint and any snapshots.

Write these paths into a Track 1 adapter rather than importing ad hoc throughout Track 2.

### Step 0.2 — canonical state export

```python
model = construct_track1_model(config)
load_checkpoint(model, source_checkpoint)
state = canonicalize_state_dict(model.state_dict())
save_safetensors(state, destination)
```

Canonicalization may normalize key prefixes but may not reorder tensor values or change dtype unless the conversion is explicitly verified.

### Step 0.3 — tensor inventory

Walk the constructed model and map parameters to semantic roles. Save canonical order. Assert repeated construction returns the same inventory.

### Step 0.4 — evaluation reproduction

Run R0 through the Track 1 evaluator and the new Gate evaluator. Differences must be explained and reduced to declared numeric tolerance.

### Step 0.5 — split freeze

Create document-level IDs for fit/fingerprint/probe/hidden/generation sets and save hashes.

---

## 6. Delta analysis implementation

```python
def compute_delta(
    base: Mapping[str, Tensor],
    target: Mapping[str, Tensor],
    inventory: Sequence[TensorSpec],
) -> dict[str, Tensor]:
    delta = {}
    for spec in inventory:
        a = base[spec.name].detach().to(torch.float32).cpu()
        b = target[spec.name].detach().to(torch.float32).cpu()
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch for {spec.name}")
        delta[spec.name] = b - a
    return delta
```

For each matrix, use randomized SVD only after validating it against exact SVD on representative tensors. Store spectra, not full factors, during analysis unless required by a candidate.

---

## 7. Classical codec implementation order

1. `DENSE_DELTA` round-trip sanity codec.
2. symmetric int8/int4 Delta-T.
3. per-tensor SVD with fixed rank.
4. global budget allocator over SVD components.
5. low-rank plus sparse residual.
6. DCT/spectral codec.
7. role-shared block codebook.
8. Kronecker/TT only if earlier results justify them.

Each codec must implement:

- fit;
- decode;
- exact byte count;
- per-tensor error report;
- MGP serialization round trip;
- deterministic output.

### SVD budget allocator pseudocode

```python
components = []
for tensor in matrix_tensors:
    U, S, Vh = factorize(delta[tensor])
    for rank_index, sigma in enumerate(S):
        component_bytes = bytes_for_next_singular_component(tensor)
        estimated_value = sigma.square().item() / component_bytes
        components.append((estimated_value, tensor, rank_index))

components.sort(reverse=True)
selected = select_until_budget(components, budget_bytes)
program = encode_selected_singular_components(selected)
```

A later allocator replaces singular energy with probe-measured benefit per byte.

---

## 8. Functional sensitivity harness

Implement a state-dict compositor:

```python
def compose_state(
    base: Mapping[str, Tensor],
    replacements: Mapping[str, Tensor],
) -> dict[str, Tensor]:
    out = {k: v.clone() for k, v in base.items()}
    for key, value in replacements.items():
        assert out[key].shape == value.shape
        out[key] = value.to(out[key].dtype)
    restore_tied_groups(out)
    return out
```

Use it to substitute tensor roles/layers and evaluate. Mark all composites as diagnostic artifacts so they cannot be mistaken for final genomes.

---

## 9. Neural block auto-decoder

### Block dataset

Each training item contains:

```text
checkpoint/model ID
tensor ID and role
layer ID
block row/column
tensor shape
base block or compressed base features
target normalized Delta block
```

Do not duplicate huge block arrays on disk unless required. Generate block views lazily from memory-mapped safetensors.

### Normalization

For role \(r\), estimate robust scale \(s_r\) from training deltas:

\[
y=\operatorname{clip}(\Delta/s_r,-c,c).
\]

Record \(s_r\), clipping, and inverse transform in the interpreter manifest.

### Forward pass

```python
def decode_block(meta, codes, base_features):
    x = concat(
        codes.global_code,
        codes.layer_code[meta.layer],
        codes.tensor_code[meta.tensor],
        role_embedding(meta.role),
        shape_embedding(meta.shape),
        coordinate_embedding(meta.block_row, meta.block_col),
        base_features,
    )
    return output_head(meta.role, trunk(x)).reshape(meta.block_shape)
```

### Training stages

1. Delta block MSE only.
2. Assemble sampled tensors and add tensor-level losses.
3. Periodically assemble complete model and add task/logit loss.
4. Add code quantization and entropy penalty.
5. Freeze interpreter; refit/quantize codes for final candidates.

Complete-model functional loss is expensive. Use it periodically rather than for every block batch.

---

## 10. Latent refinement implementation

Genome codes must be distinct `nn.Parameter` objects; interpreter weights remain frozen.

```python
for parameter in interpreter.parameters():
    parameter.requires_grad_(False)

codes = load_or_initialize_codes()
optimizer = torch.optim.AdamW(codes.parameters(), lr=latent_lr)

for step, batch in enumerate(probe_loader):
    optimizer.zero_grad(set_to_none=True)
    state = interpreter.decode_state(codes, base_state, inventory)
    with functional_model_state(model, state):
        output = model(batch.input_ids)
        loss = ntp_loss(output, batch.labels)
    loss.backward()
    optimizer.step()
```

Repeated complete state construction may be expensive. Optimizations:

- use stateless/functional calls;
- cache decoded tensors between accumulation steps when valid;
- decode only affected layers for layer-local probe losses;
- use differentiable parameter injection;
- optimize multiscale code groups sequentially.

Measure actual wall-clock, not theoretical latent dimension alone.

---

## 11. Dataset fingerprint implementation

### Fixed random gradient projections

For each tensor role, define a reproducible sketch seed. Avoid storing a dense random matrix. Use a hash/PRNG-driven signed projection or CountSketch:

\[
\widetilde g[j]=\sum_{i:h(i)=j}s(i)g[i].
\]

This gives \(O(d)\) sketching with \(k\)-dimensional output.

Store:

- sketch algorithm/version;
- hash seed;
- sketch dimension;
- normalization;
- batch IDs and token count.

### Set aggregation

Start with mean, variance, and quantile aggregation of batch fingerprints. Only add a learned DeepSets/attention aggregator when simple statistics are insufficient.

---

## 12. Early trajectory encoder

For checkpoint times \(t_i\), create features per tensor:

```text
relative progress
delta norm
finite-difference norm
cosine to previous delta
selected singular values
projected delta sketch
loss and LR
optimizer moment norms
```

The simplest compiler baseline is a per-role temporal model followed by global graph attention. Also test linear extrapolation in fitted genome-code space:

\[
\widehat p_T=p_{t_k}+(T-t_k)\widehat{\dot p}_{t_k}.
\]

This transparent baseline is mandatory before a large temporal transformer.

---

## 13. Compiler training loop

```python
for batch in run_level_loader:
    conditioning = build_conditioning(
        architecture=batch.architecture_graph,
        dataset_fingerprint=batch.dataset_fingerprint,
        trajectory=batch.allowed_prefix,
        base=batch.base_features,
    )

    distribution = compiler(conditioning)
    genome = distribution.rsample_or_mean()
    candidate_state = frozen_interpreter.decode_state(genome, batch.base_state)

    loss = (
        lambda_code * code_target_loss(genome, batch.fitted_genome)
        + lambda_delta * normalized_delta_loss(candidate_state, batch.target_state)
        + lambda_task * task_loss(candidate_state, batch.fit_batches)
        + lambda_logit * reference_logit_loss(candidate_state, batch.fit_anchors)
        + lambda_rate * distribution_rate_loss(distribution)
    )

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    clip_grad_norm_(compiler.parameters(), max_norm)
    optimizer.step()
```

For large targets, decode sampled tensor subsets for delta loss and perform complete functional passes at a lower cadence.

Run-level batching must not mix hidden-run endpoint data into training.

---

## 14. MGP serialization

Canonical rules:

- sorted JSON keys;
- stable float formatting for metadata;
- safetensors payload keys sorted;
- no pickle;
- SHA-256 over canonical manifest and payload files;
- archive creation after hashes are recorded;
- actual archive byte count stored in evaluation report, not inside the hashed manifest if that creates recursion.

Round-trip test:

```text
program in memory
 -> serialize
 -> destroy in-memory object
 -> load from files
 -> decode twice
 -> compare tensors/checksums
 -> evaluate smoke batch
```

---

## 15. Bit accounting implementation

Never estimate a tensor payload from latent dimensions when a serialized artifact exists. Count file bytes and also compute logical component bytes.

```python
@dataclass(frozen=True)
class BitBreakdown:
    manifest: int
    codes: int
    factors: int
    codebook_indices: int
    scales: int
    patch: int
    exact_residual: int
    interpreter: int
    dictionaries: int
    base: int
```

Assert:

```python
sum(target_specific_components) == measured_target_payload_bytes
```

Allow container/format overhead as a separate field.

---

## 16. Tests that must exist before result runs

### Specimen

- original/canonical R0 logits match;
- W0/WT keys and shapes match;
- tensor order stable;
- hidden split disjoint by source document/poem;
- tied groups correct.

### MGP

- dense-delta codec recovers WT exactly at target dtype;
- decode deterministic;
- malformed records rejected;
- missing tensors rejected;
- output ties preserved;
- serialized and in-memory decode match.

### Metrics

- R0 receives expected score;
- W0/random receives worse score;
- deliberate tensor corruption is detected;
- KL/top-k calculations match small hand examples;
- byte accounting matches actual files.

### Leakage

- training loader refuses hidden run IDs;
- compiler input schema cannot contain target endpoint path;
- hidden verifier loader is accessible only through Gate entry point;
- target-specific interpreter checkpoint is flagged and counted.

### Neural decoder

- one tiny tensor can be overfit;
- one complete tiny model can be decoded and executed;
- gradients reach genome codes;
- frozen interpreter does not update during latent refinement;
- quantized-code decode matches declared quantization.

---

## 17. Logging and reproducibility

Each run directory:

```text
runs/<run_id>/
  resolved_config.yaml
  environment.json
  git_state.json
  parent_artifacts.json
  stdout.log
  metrics.jsonl
  checkpoints/
  candidates/
  evaluation_development.json
  decision_log.md
```

`environment.json` should include:

- Python and PyTorch versions;
- CUDA/ROCm/MPS versions;
- GPU/CPU model;
- deterministic flags;
- dependency lock hash;
- relevant environment variables.

Record dirty git diff hash or patch when the worktree is not clean.

---

## 18. Reporting script

`make_report.py` should consume immutable evaluation reports and produce:

- Markdown summary;
- CSV/Parquet result table;
- rate–distortion plots;
- repair curves;
- tensor-role allocation chart;
- representative generation outputs;
- experiment lineage.

Do not hand-copy headline metrics from console output.

---

## 19. Performance considerations

- Keep immutable model states on CPU or memory-mapped storage when not executing.
- Batch block decoding by role and shape.
- Avoid Python loops per scalar.
- Use chunked SVD/randomized methods for huge embeddings after validation.
- Cache architecture/coordinate embeddings.
- Profile data loading, decode, state injection, and model execution separately.
- Preserve a CPU-only smoke path for tests.
- A full 50M decode is small enough for an initial eager implementation; optimize only after measuring the bottleneck.

---

## 20. First vertical slice

The first complete slice should do exactly this:

1. Load canonical W0 and WT.
2. Compute Delta-T.
3. Encode each matrix with fixed-rank SVD and each vector directly.
4. Serialize an MGP.
5. Destroy in-memory factors.
6. Reload and decode the MGP.
7. Load the candidate model.
8. Run probe and development evaluation.
9. Report actual bytes, decode time, loss, and KL.

Only after this works should the agent add global budget allocation, sparse patches, neural decoders, or compilers.
