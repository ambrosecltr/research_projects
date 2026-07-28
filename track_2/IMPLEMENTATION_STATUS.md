# GENOME implementation status

## Implemented and tested

### Exact Track 1 boundary

- Concrete adapter for `poetry_50m/track_1` rather than a placeholder template.
- Track 1's own `ModelConfig`, `DecoderOnlyTransformer`, and `seed_everything` are used.
- Trainer checkpoints (`format_version: 2`) and weights-only snapshots (`poetry50m.weights.v1`) are supported.
- The exact packed-record validation/test loss path is reproduced, including `loss_mask` and supervised-token weighting.
- Production parameter counts are checked: 50,343,424 for GPT and 54,596,096 for nGPT.
- Exact Track 1 tensor roles cover fused QKV, attention output, fused SwiGLU input, MLP output, nGPT scales, and residual rates.
- A preflight command reports R0 progress and required artifacts.
- Partial checkpoints are rejected as WT while `require_complete_endpoint: true`.
- W0/WT model-config, training-config, step, and run-manifest lineage are validated before freezing.
- Final trainer checkpoint, final trajectory snapshot, run manifest, and train receipt hashes are reconciled.
- Imported `poetry50m` package origin is pinned to the configured Track 1 checkout.

### R0 specimen foundation

- Canonical W0/WT export to safetensors.
- W0 is generated twice and both state hashes must agree when no explicit base checkpoint is supplied.
- Architecture, tokenizer, corpus, training-recipe, split, environment, tensor-inventory, and hash manifests.
- Stable semantic tensor roles and layer indices.
- Tied-storage discovery and exact tied-value restoration.
- Immutable specimen integrity verification.
- Fail-closed artifact boundaries: exact format versions, complete file manifests,
  path-traversal rejection, symlink rejection, and undeclared-file rejection.

### Model Genome Program v0.1

- Canonical JSON manifest plus safetensors payloads.
- Manifest/payload integrity hashes.
- Architecture, inventory, and W0 contract hashes.
- Complete tensor records in frozen canonical order.
- Deterministic decoding and fail-closed validation.
- Shared-payload support for global/layer neural codes.
- Actual canonical MGP file-byte accounting.
- Dense Delta-T, symmetric int8, packed int4, low-rank, sparse-patch, tied-copy, and neural-block opcodes.

### Transparent G0 codecs

- Bit-exact dense sanity codec using fp64 Delta-T arithmetic before target-dtype casting.
- Per-tensor symmetric int8 Delta-T.
- True packed int4 Delta-T, two values per byte.
- Per-matrix fixed-rank SVD with explicit vector handling.
- Global singular-component byte-budget allocation.
- Reusable full-SVD workspace: every unique matrix is factorized once per frontier.
- Shared SVD factorization time is measured once and separated from candidate fit time.
- Low-rank plus largest-residual sparse patch.
- Parameter error by tensor and semantic role.

### Evaluation and experimentation

- Adapter-driven task loss and perplexity.
- Anchor-logit KL and top-k agreement.
- Compact anchor storage, avoiding retained full-vocabulary logits for every token.
- R0/W0/candidate comparisons.
- Development Genome Gate reports.
- Delta norms, role energy, singular spectra, effective rank, and perturbation utilities.
- Rate–distortion orchestration with unique labels, immutable outputs, atomic publication,
  and a hash-bound `rate_distortion_context.json`.
- Latent-only and full-weight repair baselines.
- Markdown/CSV/JSON report generation.
- Immutable run-context/environment helpers.
- Evaluation-only Track 1 checkpoint export for the existing generation/blind-judgment suite.

### Model-native evidence and G1/G2 scaffolding

- Architecture graph with role/layer/tie relations.
- Source-document-level deterministic split utilities.
- Deterministic CountSketch gradient fingerprints from W0.
- Trajectory norm/cosine/statistical features and a transparent code-space extrapolator.
- Persistent run-level model-life index.
- Role-conditioned neural block auto-decoder with global/layer/tensor codes.
- External shared neural interpreter artifacts with hashes and byte accounting.
- Fixed-size probabilistic genome compiler baseline.
- Run-level compiler training dataset and safetensors compiler artifact.

### Validation performed

- **Twenty-eight automated tests pass.**
- Python bytecode compilation passes for the package, examples, scripts, and tests.
- The complete tiny causal-LM pipeline runs from training through R0 freeze, MGP serialization, object destruction/reload, decoding, execution, and report generation.
- Dense decoding reproduces the target exactly at its declared dtype.
- Int8 and packed-int4 candidates preserve the tiny model's function closely.
- Neural block decoding can overfit a held toy tensor.
- Gradient fingerprinting, latent-code repair, rate–distortion orchestration, and reporting have been smoke-tested.
- The concrete poetry50m adapter's seed replay, checkpoint parsing, completion gate, and exact tensor-role mappings are unit-tested with an isolated Track 1 fixture.
- W0/WT lineage, completion-receipt reconciliation, compact-anchor loss parity, config-relative paths, duplicate-key rejection, and evaluation-checkpoint export are tested.
- Specimen, MGP, compiler, and interpreter artifact loaders are tested against path traversal and malformed manifests.
- The rate sweep is tested for one-time SVD reuse and atomic failure without a partial final directory.

## Pending only because the real full endpoint is not yet available

These are experiment executions, not missing software design:

1. Freeze the completed R0 checkpoint into the immutable specimen.
2. Confirm its 50,343,424-parameter inventory and W0 identity against the real files.
3. Reproduce the known Track 1 validation/test metrics from canonical WT.
4. Measure the real R0 SVD/int4/low-rank-plus-sparse rate–distortion frontier.
5. Fit the first role-conditioned neural genome to R0.
6. Measure compile-and-polish repair curves.
7. Produce independent sibling model lives for G1/G2 compiler training.

## Deliberate current limitations

- The transparent codec set does not yet include DCT/spectral, Kronecker, tensor-train, or learned role codebooks. Add them after the R0 SVD/int4 frontier identifies the relevant failure mode.
- The neural auto-decoder currently trains primarily on normalized block reconstruction. Complete-model functional losses should be selected from measured R0 cost and sensitivity.
- The generic Gate is a development evaluator. The final hidden verifier should remain isolated at the repository/process level.
- Exact cross-hardware bitwise neural-interpreter decoding is not claimed.
- The baseline compiler consumes fixed-size evidence vectors. A graph or Set Transformer compiler is a measured later upgrade, not a prerequisite for G0.
- Useful G1/G2 training requires multiple independent model lives. Slices from one R0 trajectory are not treated as independent runs.

## Next milestone

When R0 finishes, run:

```bash
genome track1-preflight --config configs/poetry50m_track1.yaml
genome freeze --config configs/poetry50m_track1.yaml
genome verify --specimen artifacts/specimens/track1_R0 --config configs/poetry50m_track1.yaml
genome analyze --specimen artifacts/specimens/track1_R0 --output artifacts/analysis/R0
genome rate-distortion --specimen artifacts/specimens/track1_R0 \
  --config configs/poetry50m_track1.yaml \
  --output artifacts/rate_distortion/R0 \
  --ranks 0,1,2,4,8,16,32,64
```

The milestone is complete only when `verification.json` proves canonical WT reproduces Track 1 while W0 is materially worse, and the rate–distortion report records function versus actual artifact bytes.
