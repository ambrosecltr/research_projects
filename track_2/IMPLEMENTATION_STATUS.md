# GENOME implementation status

## Proven before PolyPythia Round One

- Canonical W0/WT tensor inventories and tied-weight handling.
- Deterministic MGP serialization, integrity validation, and decoding.
- Exact known-endpoint round trips through the Track 1 adapter.
- Dense, quantized, SVD, sparse, and single-life neural G0 mechanics.

These results prove known-endpoint representation. They do not prove endpoint prediction.

## Implemented for PolyPythia Round One

### Immutable source and split

- Ten standard non-deduplicated Pythia 14M lives.
- Training seeds 0–7, development seed8, hidden seed9.
- All 154 branches per life pinned by commit.
- Every selected weight pinned by exact LFS SHA-256 and byte count.
- Pile data-order files and tokenizer files pinned by commit and content identity.
- Intermediate checkpoints remain catalogued but are not downloaded by the primary endpoint experiment.

### GPT-NeoX compatibility

- Strict native-to-canonical and canonical-to-native key conversion.
- Support for historical `embed_out` and current Transformers `lm_head` output names.
- Explicit exclusion of regenerated historical attention-mask and rotary buffers.
- Exact tensor and real-model logit round-trip regression tests.
- Canonical endpoint artifacts in safetensors.

### Neural Genome Decoder

- One role-conditioned block decoder shared across independent model lives.
- Per-life global, layer, tensor, and block genome codes.
- W0 block values are decoder inputs; WT values are reconstruction targets only.
- Vector and scalar tensors use the neural block opcode instead of dense endpoint payloads.
- Frozen-decoder development code fitting for seed8.
- Decoder audit across seeds0–8 with byte accounting, parameter metrics, Wikitext behavior, and logit comparisons.

### GENOME Compiler

- Endpoint-free evidence from W0, architecture, tokenizer, data order, and training recipe.
- No endpoint hashes, fitted target codes, early weights, or intermediate weights in compiler inputs.
- Training through the frozen decoder against Delta-T block loss.
- No arbitrary latent-code label matching.
- Blockwise code generation avoids a model-sized flat output head.
- One-shot hidden prediction and immutable prediction seal.

### Hidden evaluation

- Hidden seed9 WT cannot be materialized before a valid prediction seal.
- The predicted genome is decoded and run before reveal.
- Full pinned Wikitext comparison of W0, predicted WT, and true WT.
- Six pinned zero-shot LM Evaluation Harness tasks:
  LAMBADA OpenAI, PIQA, Winogrande, ARC-Easy, SciQ, and document-level Wikitext.
- No repair, polishing, or early training prefix in the primary result.

## Local validation

- 45 automated tests pass locally; the one additional CUDA Runtime test is skipped on the Mac.
- Python bytecode compilation passes.
- Focused Ruff checks pass for all new and modified Round One modules.
- A complete synthetic learned path passes:
  shared decoder training, development-code fitting, compiler training, hidden prediction, seal creation, MGP decoding, GPT-NeoX assembly, and forward execution.
- The exact 14M block sampler measured about 0.0084 seconds per 256-block CPU batch on the local Mac. This is about 421 seconds of sampler work for 50,000 updates before GPU compute.

## Pinned source totals

The current local source plan reports:

- full 1,540-checkpoint catalogue: 78,251,476,072 bytes;
- sealed primary endpoint materialization: 962,921,344 bytes;
- post-seal endpoint materialization: 1,016,252,936 bytes.

The large catalogue total is provenance, not the primary download size.

## Paid Round One results so far

Two decoder designs were rejected before compiler training:

- V1 used role-wide scales and had no block codes. Mean parameter relative L2 was
  `0.662834`; mean Wikitext loss gap was `81.079589`.
- V2 corrected QKV roles, padded-vector loss, per-tensor scales, and block coordinates.
  Mean parameter relative L2 was `0.643889`; mean Wikitext loss gap was `87.325305`.

Both designs made decoded models much worse than W0, including on fitted training lives.
They proved that one global/layer/tensor code hierarchy did not have enough target-specific
capacity. The hidden seed9 endpoint remained sealed, and no compiler was trained through either
failed decoder.

V3 adds one 16-value latent code per 16-by-16 block. Pythia 14M has 55,552 untied blocks, so the
raw block-code payload is 3,555,328 bytes per life before later quantization. The blockwise
Compiler predicts these codes from allowed evidence and W0 features through the frozen decoder.
Real V3 quality is not yet claimed.

## Not yet claimed

- Real shared-decoder reconstruction quality on PolyPythia.
- Real one-shot hidden seed9 quality.
- A compact rate-quality result after counting the shared decoder.
- Cross-size transfer or evidence that the compiler did not learn a 14M-only rule.
- Transfer to the local 8M Track 1 model or another architecture family.

Those claims require the paid Round One execution and its immutable result artifacts.

## Track 1 status

The previous 50M Track 1 boundary is legacy G0 integration. The new 8M model needs a full-life record from its true random W0 through pretraining and SFT to final WT. The pretrain-to-SFT segment can be retained as auxiliary stage-local data. It must not replace the full random-W0-to-final-WT record.
