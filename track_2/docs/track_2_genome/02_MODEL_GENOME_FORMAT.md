# Model Genome Program specification

> **Archived pre-recovery note.** Learned-interpreter fields and decoder-first recommendations in
> this file are historical only. Active MGPs use the deterministic Runtime defined by
> `../../RECOVERY.md`.

## 1. Purpose

A **Model Genome Program (MGP)** is the compact, executable output of GENOME. It is not required to be human-readable symbolic mathematics. It is a versioned program/data structure that a shared interpreter can deterministically expand into a target model state.

The format must support three progressively richer representations:

1. **Structured formula genomes** — explicit low-rank, spectral, Kronecker, dictionary, and sparse components.
2. **Neural genomes** — compact latent codes decoded by a shared coordinate/block network.
3. **Hybrid genomes** — explicit structured components plus a neural residual and a sparse exception patch.

The initial implementation should serialize MGPs as a directory or archive containing JSON metadata and tensor payloads in `safetensors` or another non-executable format. A custom packed binary format is deferred until the representation is useful.

---

## 2. Design requirements

An MGP must be:

- **Complete:** sufficient to reconstruct a runnable candidate when combined with declared shared artifacts.
- **Deterministic:** the same MGP, base state, interpreter, and environment contract produce the same candidate.
- **Countable:** every target-specific byte can be measured.
- **Streamable:** individual tensors or blocks can be decoded without materializing all weights at once.
- **Versioned:** old results remain decodable after the implementation evolves.
- **Integrity checked:** hashes identify the architecture, base initialization, interpreter, tokenizer, and tensor inventory.
- **Non-executable by default:** do not use Python pickle as the canonical exchange format.
- **Explicit about conditioning:** the header declares all information required from outside the file.

---

## 3. Conceptual reconstruction equation

For tensor \(m\):

\[
\widehat W_m
=
B_m
+
\operatorname{Scale}_m
\left(
\operatorname{Structured}_m
+
\operatorname{Neural}_m
\right)
+
\operatorname{Patch}_m,
\]

where \(B_m\) is normally the corresponding W0 tensor.

A practical hybrid is:

\[
\widehat W_m=W_{0,m}
+\alpha_m
\left[
U_m\operatorname{diag}(a_m)V_m^\top
+\sum_q c_{m,q}(A_{m,q}\otimes B_{m,q})
+\mathcal T_m^{-1}(M_m\odot C_m)
+D_\psi(z_m)
\right]
+S_m.
\]

Not every term must be present. Absent terms cost zero payload bytes apart from their opcode/header.

---

## 4. File layout for MGP v0

Recommended development layout:

```text
candidate.mgp/
  manifest.json
  genome.safetensors
  patch.safetensors              # optional
  exact_residual.bin             # optional diagnostic
  metrics_at_creation.json       # not required for decoding
  README.md                       # optional human summary
```

`manifest.json` is canonical and sorted before hashing. The payload file stores only declared tensors.

Example header:

```json
{
  "format": "MGP",
  "version": "0.1.0",
  "project": "GENOME",
  "research_level": "G0",
  "candidate_id": "g0_hybrid_budget_0100bp_run_0007",
  "created_utc": "2026-07-27T00:00:00Z",
  "architecture_manifest_sha256": "...",
  "tensor_inventory_sha256": "...",
  "tokenizer_sha256": "...",
  "base_mode": "seed_replay",
  "base_checkpoint_sha256": null,
  "initialization_manifest_sha256": "...",
  "interpreter_id": "genome_interpreter_v0",
  "interpreter_sha256": "...",
  "decoder_dtype": "float32",
  "output_dtype": "bfloat16",
  "tensor_order": ["tok_embeddings.weight", "blocks.0.attn.q_proj.weight"],
  "tied_groups": [["tok_embeddings.weight", "lm_head.weight"]],
  "records": [],
  "bit_accounting": {},
  "conditioning_contract": {}
}
```

The development format may be verbose. Compression claims must use actual packed byte counts or a clearly defined entropy estimate, not character counts of pretty-printed JSON.

---

## 5. Tensor record

Every state-dict tensor has one record, including copied/seed-generated tensors. A record contains:

```json
{
  "name": "blocks.4.mlp.down_proj.weight",
  "role": "mlp_down",
  "layer_index": 4,
  "shape": [512, 2048],
  "dtype": "bfloat16",
  "base_source": "W0",
  "decode_mode": "hybrid",
  "scale": 0.0371,
  "components": [
    {"opcode": "LOW_RANK", "rank": 12, "payload_prefix": "t004.lr"},
    {"opcode": "NEURAL_BLOCK_FIELD", "code_key": "t004.z", "block": [16, 16]},
    {"opcode": "SPARSE_PATCH", "nnz": 384, "payload_prefix": "t004.sp"}
  ],
  "output_checksum": "..."
}
```

Required metadata:

- exact tensor name;
- semantic role;
- layer/depth position where applicable;
- shape and dtype;
- base source;
- ordered operations;
- payload keys;
- expected output checksum for frozen candidate artifacts.

---

## 6. Core opcodes

### 6.1 `BASE_COPY`

Use the base tensor unchanged.

\[
\widehat W_m=B_m.
\]

Useful for tensors whose learned delta is negligible or whose effect is deliberately excluded.

### 6.2 `BASE_SEED`

Generate the tensor from the pinned initialization algorithm and random stream. The record includes a random-stream offset or tensor-generation index when necessary.

### 6.3 `DENSE_DELTA`

Store the complete delta directly. This is a sanity baseline, not a compact genome.

### 6.4 `QUANTIZED_DELTA`

Store a quantized delta and scale/zero-point metadata. Support at minimum:

- symmetric int8;
- symmetric int4 packed two values per byte;
- per-tensor scale;
- per-row or per-block scales as a later option.

This establishes a conventional compression baseline.

### 6.5 `LOW_RANK`

For a matrix delta:

\[
\Delta_m\approx U_m\operatorname{diag}(a_m)V_m^\top.
\]

Payload options:

- dense \(U,a,V\);
- quantized factors;
- factors generated from shared bases plus small coefficients.

The record must count all factor and scale bytes.

### 6.6 `SHARED_BASIS`

Use a shared role-specific or global dictionary:

\[
\Delta_m\approx\sum_{j=1}^{J}c_{m,j}B_{r(m),j}.
\]

The bases are part of the shared interpreter artifact unless target-specific. Coefficients are target-specific genome bytes.

### 6.7 `KRONECKER_SUM`

\[
\Delta_m\approx\sum_{q=1}^{Q}c_q(A_q\otimes B_q).
\]

The chosen factor shapes must multiply to the tensor shape. Record padding/cropping explicitly if used.

### 6.8 `TENSOR_TRAIN`

Reshape a tensor into a higher-order tensor and store tensor-train cores. Record the factorization of each original dimension and all TT ranks.

### 6.9 `SPECTRAL`

Store selected transform coefficients:

\[
\Delta_m\approx\mathcal T^{-1}(M\odot C).
\]

Initial transforms:

- DCT-II by row/column;
- 2D FFT with conjugate-symmetry handling;
- learned fixed orthogonal dictionary later.

The mask may be implicit through a deterministic coefficient ordering, avoiding index bytes.

### 6.10 `BLOCK_CODEBOOK`

Split the tensor into fixed blocks and encode each block using a learned codebook index plus optional residual scale:

\[
\Delta_m[B_b]\approx s_bQ[k_b].
\]

Count codebook bytes as shared interpreter bytes and indices/scales as genome bytes.

### 6.11 `NEURAL_COORD_FIELD`

A shared decoder emits scalar values from coordinates and codes:

\[
\Delta_m[i,j]=D_\psi(z_g,z_l,z_m,\bar i,\bar j,e_{role}).
\]

The MGP stores only codes and decoder-selection metadata. The interpreter weights are shared.

### 6.12 `NEURAL_BLOCK_FIELD`

A shared decoder emits a block:

\[
\Delta_m[B_{u,v}]=D_\psi(z_g,z_l,z_m,e_{u,v},e_{role}).
\]

This is the preferred neural mode for GPU efficiency.

### 6.13 `SPARSE_PATCH`

Store coordinate/value corrections. Index encoding choices:

- sorted flat indices with delta coding;
- block-local bitmaps;
- run-length encoding;
- top-k deterministic positions with only values stored, when positions can be reproduced from a declared sensitivity rule.

### 6.14 `LOW_RANK_PATCH`

A small residual factorization:

\[
S_m=A_mB_m^\top.
\]

Useful when the main decoder captures broad structure but misses a few sensitive directions.

### 6.15 `COPY_FROM_TIED`

For tied parameters, decode one owner tensor and alias/copy it exactly. Do not spend genome bits twice.

### 6.16 `EXACT_RESIDUAL`

Optional diagnostic-only payload that transforms the decoded candidate into the exact target bit pattern. The residual representation, codec, byte order, target dtype, and application order must be declared. `EXACT_RESIDUAL` is counted in full and is never required for functional GENOME success.

---

## 7. Formula-selection grammar

A simple abstract grammar is:

```text
MODEL      := HEADER GLOBAL_CODE LAYER* TENSOR* PATCH* END
LAYER      := LAYER_META LAYER_CODE
TENSOR     := TENSOR_META BASE OP* CHECKSUM
BASE       := BASE_COPY | BASE_SEED
OP         := QUANTIZED_DELTA
           | LOW_RANK
           | SHARED_BASIS
           | KRONECKER_SUM
           | TENSOR_TRAIN
           | SPECTRAL
           | BLOCK_CODEBOOK
           | NEURAL_COORD_FIELD
           | NEURAL_BLOCK_FIELD
PATCH      := SPARSE_PATCH | LOW_RANK_PATCH | EXACT_RESIDUAL
```

The first implementation does not need to train a model to emit opcodes. Select formula families through an offline rate–distortion search. Later, the compiler may emit discrete opcode tokens and continuous arguments.

---

## 8. Global, layer, and tensor codes

Recommended hierarchy:

```text
global_code: [d_g]
layer_codes: [n_layers, d_l]
tensor_codes: [n_tensors, d_t]
block_codes: optional sparse/per-block exceptions
```

A tensor code should not redundantly store metadata that the architecture graph already provides. The decoder receives:

- role embedding;
- normalized layer depth;
- shape embedding;
- fan-in/fan-out;
- tensor orientation/convention;
- global and layer codes;
- base-weight features if enabled.

For initial G0 experiments, codes are free trainable parameters optimized directly for R0. For later levels, the compiler predicts them.

---

## 9. Quantizing genome codes

A floating-point latent is not automatically a compact genome. Evaluate:

- fp32;
- fp16/bfloat16;
- int8 with scale;
- vector-quantized codebook indices;
- entropy-coded discrete latents.

Training may use continuous codes and quantization noise or a straight-through estimator. Final byte accounting uses the serialized representation, not the training representation.

A useful objective is:

\[
\mathcal L=\mathcal L_{function}+\beta\sum_i -\log_2 p_\eta(q_i),
\]

where \(q_i\) are quantized code symbols and \(p_\eta\) is an entropy model.

---

## 10. Exception patches and bit allocation

The patch exists because a compact global formula may miss a small number of functionally important directions.

Allocate patch bits by estimated benefit per bit:

\[
\operatorname{priority}(c)=
\frac{\widehat{\Delta D_f}(c)}{B(c)+\lambda C(c)}.
\]

Candidate corrections may be:

- one scalar coordinate;
- a row/column block;
- one singular direction;
- one attention head;
- one normalization vector;
- one embedding subset;
- one low-rank tensor residual.

The initial allocator may greedily test tensor-family substitutions or use gradient magnitude projected through decoder residuals. A learned allocator is later work.

---

## 11. Deterministic decode contract

Each MGP declares:

- interpreter code/version hash;
- interpreter weight hash;
- base mode and base hash;
- architecture and tensor inventory hash;
- decoding dtype;
- output dtype and rounding mode;
- tensor/block traversal order;
- transform normalization conventions;
- padding/cropping conventions;
- random generator and seed if any stochastic decoder is used;
- whether stochastic sampling is prohibited for this frozen candidate.

For a published/frozen candidate, decode must be deterministic. Generative sampling happens before freezing the MGP.

Pseudo-decoder:

```python
def decode_mgp(mgp, architecture, base_state, interpreter):
    validate_hashes(mgp, architecture, base_state, interpreter)
    output = {}

    for record in mgp.records_in_canonical_order():
        if record.is_tied_non_owner:
            continue

        tensor = load_base(record, base_state)
        for component in record.components:
            tensor = apply_component(
                tensor=tensor,
                component=component,
                mgp=mgp,
                interpreter=interpreter,
            )

        tensor = cast_with_declared_rounding(tensor, record.dtype)
        assert tensor.shape == tuple(record.shape)
        output[record.name] = tensor

    restore_ties_and_aliases(output, mgp.tied_groups)
    validate_output_checksums_when_present(output, mgp)
    return output
```

---

## 12. Bit accounting

Report four rates:

### Per-model payload

\[
B_{payload}=B(p)+B(S)+B(\text{target-specific metadata}).
\]

### Single-model total

\[
B_1=B_{payload}+B(\psi)+B(\text{shared dictionaries})+B(\text{base if stored}).
\]

### Amortized total over N generated models

\[
B_N=B_{payload}+\frac{B(\psi)+B(\text{shared dictionaries})+B(\text{shared base})}{N}.
\]

### Exact-recovery total

\[
B_{exact}=B_1+B(\text{lossless residual}).
\]

Also report actual archive bytes and raw checkpoint bytes. Do not claim compression using only the latent dimensionality.

---

## 13. Canonical tensor ordering

Use the frozen Track 1 tensor inventory. Ordering must never depend on dictionary iteration or incidental module construction.

Canonical key:

```text
(layer_index_or_-1, role_order, tensor_name)
```

Example role order:

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
other
```

Record all exceptions in the architecture manifest.

---

## 14. Tied weights and aliases

If Track 1 ties token embeddings and the LM head, the candidate must preserve that relationship unless an experiment explicitly tests untying and counts the additional parameters.

The manifest identifies:

- owner tensor;
- aliases;
- whether storage is shared or copied;
- whether gradients were tied during R0 training.

A checksum test must verify equality after decode.

---

## 15. Validation before model execution

An MGP is invalid if any of these fail:

- unknown format/version;
- architecture mismatch;
- base/interpreter hash mismatch;
- missing or duplicate tensor record;
- shape/dtype mismatch;
- illegal transform shape;
- non-finite payload value;
- index out of range;
- tied-group inconsistency;
- output contains NaN/Inf;
- byte accounting does not match serialized payload;
- undeclared target-specific artifact is required.

Fail closed. Do not silently coerce malformed genomes.

---

## 16. Initial MGP versions

### MGP 0.1 — transparent G0 audit

- base W0;
- quantized delta;
- SVD/low-rank;
- spectral coefficients;
- sparse residual;
- JSON + safetensors;
- no learned compiler.

### MGP 0.2 — neural auto-decoder

- global/layer/tensor codes;
- shared block decoder;
- optional structured branch;
- direct latent optimization;
- quantized codes.

### MGP 0.3 — compiled genome

- compiler-predicted codes;
- dataset/trajectory conditioning manifest;
- probe/link history;
- frozen final MGP.

### MGP 1.0 — stable cross-run format

Only declare 1.0 after at least one hidden-run compilation, stable decoding tests, and backwards-compatible tooling.
