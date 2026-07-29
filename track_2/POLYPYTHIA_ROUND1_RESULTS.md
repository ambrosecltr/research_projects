# PolyPythia Round One results

## Outcome

Round One is complete.

The result has two parts:

1. **Neural Genome Decoder: passed.**
   One shared decoder plus a canonical fp16 residual represented seeds0–8 with near-WT
   behavior.
2. **GENOME Compiler: failed hidden transfer.**
   The Compiler produced a complete runnable seed9 model, but that model was much worse than W0.

Round Two is not yet a memorization confirmation. Seed9 already showed that the Round One
Compiler did not transfer to an unseen seed.

## Fixed protocol

- Family: standard non-deduplicated Pythia 14M.
- Decoder and Compiler training: seeds0–7.
- Development: seed8.
- Hidden one-shot evaluation: seed9.
- Compiler inputs: W0, architecture, tokenizer identity, dataset-order identity, and training
  recipe.
- Forbidden inputs: WT, endpoint hashes, fitted endpoint codes, and early or intermediate
  checkpoints.
- Primary prediction: one shot.
- Early training prefix: not used.
- Repair or polishing: not used.
- Hidden WT was revealed only after both the prediction seal and pre-reveal Runtime record existed.

## Representation findings

The first three decoder designs failed.

| Design | Main change | Mean parameter relative L2 | Mean Wikitext loss gap | Result |
|---|---|---:|---:|---|
| V1 | Global, layer, and tensor codes | 0.662834 | 81.079589 | Failed |
| V2 | Correct QKV roles, scales, masks, and coordinates | 0.643889 | 87.325305 | Failed |
| V3 | One free 16-value code per 16×16 block | 0.389099 | 89.719535 | Failed |

V3 improved parameter error but made model behavior worse. Seed8 decoded loss was `193.530336`.
Seed8 W0 loss was `11.055402`, and WT loss was `5.461259`.

The measured rate frontiers explained the failure:

- width-16 role-conditioned block PCA kept `47.9460%` of centered delta energy;
- input embedding width 16 kept only `6.97%`;
- input embedding width 128 kept only `51.77%`;
- fp16 whole-tensor SVD rank 112 still produced seed8 loss `14.018015`;
- fp16 full rank 128 produced loss `5.519440` at `28,994,048` payload bytes;
- direct fp16 Delta-T produced loss `5.460485` at `28,135,424` payload bytes;
- direct int8 produced loss `7.263859`;
- direct int4 produced loss `57.704032`.

V4 therefore used a learned structured decoder plus a canonical fp16 residual:

```text
residual block =
  true normalized Delta-T block
  - frozen structured decoder prediction
```

The residual is deterministic. It is not a fitted latent label. The Compiler is trained through
the frozen decoder against endpoint blocks.

## V4 Neural Genome Decoder result

The shared structured component trained for 50,000 updates:

- normalized block MSE at update 1: `0.953145`;
- update 25,000: `0.528109`;
- update 50,000: `0.467642`.

Seed8 hierarchy fitting trained for 20,000 updates:

- normalized block MSE at update 1: `0.783827`;
- update 10,000: `0.638190`;
- update 20,000: `0.599121`.

The canonical residual then preserved the remaining endpoint change.

Full seeds0–8 decoder audit:

- mean parameter relative L2: `0.000137383`;
- worst parameter relative L2: `0.000195750`;
- mean Wikitext loss gap: `0.000105020`;
- worst Wikitext loss gap: `0.001437497`;
- mean anchor-logit KL: `0.000131592`;
- worst anchor-logit KL: `0.000511927`;
- training-life top-1 agreement: `0.9866` to `0.9946`;
- seed8 top-1 agreement: `0.98125`.

Size:

- one target-specific MGP: about `28.53 MB`;
- logical target-specific payload: `28,453,504` bytes;
- shared decoder: `4,614,112` bytes;
- W0 state: `56,278,040` bytes.

This is a valid high-rate shared-decoder result. It is not yet an aggressive compression result.

## GENOME Compiler result

The Compiler trained for 50,000 updates through the frozen V4 decoder.

- training normalized block MSE: `1.749143 → 0.150069`;
- seed8 development MSE: `1.637307 → 0.453779`;
- best logged seed8 MSE: `0.415970` at update 11,500;
- compiler artifact size: about `51 MB`.

The training-development gap showed weak transfer before hidden reveal, but the fixed hidden
experiment continued.

## Hidden seed9 sequence

Before prediction, seed9 contained only W0. Its WT canonical file was `null`.

The Compiler then:

1. produced one fp16 residual genome;
2. wrote an immutable prediction seal;
3. decoded the genome through the MGP Runtime;
4. assembled a 14,067,712-parameter model;
5. completed a finite-logit forward pass;
6. revealed seed9 WT only after these records existed.

The Runtime decode took `1.8746` seconds.

## Primary hidden Wikitext result

All states used the same 140 batches and 286,720 tokens.

| State | Mean loss | Perplexity |
|---|---:|---:|
| W0 | 11.072410 | 64,370.48 |
| Predicted WT | 78.254531 | 9.67e33 |
| True WT | 5.253024 | 191.14 |

Predicted-versus-true:

- loss gap: `73.001507`;
- parameter relative L2: `1.144849`;
- anchor-logit KL: `74.292494`;
- top-1 agreement: `0`;
- top-5 agreement: `0`.

The predicted model was runnable, but it was not a trained-quality model. It was worse than W0.

## Pinned zero-shot task result

| Task and metric | W0 | Predicted WT | True WT |
|---|---:|---:|---:|
| ARC-Easy normalized accuracy | 0.2475 | 0.2803 | 0.3093 |
| LAMBADA accuracy | 0.0000 | 0.0000 | 0.0877 |
| PIQA normalized accuracy | 0.5027 | 0.4902 | 0.5539 |
| SciQ normalized accuracy | 0.2240 | 0.2220 | 0.4700 |
| Winogrande accuracy | 0.4972 | 0.4854 | 0.4964 |
| Wikitext word perplexity | 311,956.49 | 1.91e39 | 132.63 |

The multiple-choice results are mostly near chance. LAMBADA and Wikitext clearly show that the
predicted model did not learn the hidden endpoint behavior.

## What Round One proves

Proven:

- the strict hidden protocol worked;
- one shared V4 Neural Genome Decoder can represent nine varied Pythia 14M endpoints at the
  measured fp16 residual rate;
- the GENOME Compiler can emit a complete genome without WT;
- the deterministic MGP Runtime can expand that predicted genome into a runnable model.

Failed:

- the Compiler did not predict a useful hidden seed9 endpoint;
- the result did not transfer across an unseen seed and data order;
- Round One did not show training replacement.

Not tested by this result:

- cross-size transfer;
- cross-family transfer;
- the local Track 1 8M model;
- post-training recipes;
- repair, refinement, or polishing.

## Evidence limitation found

The current dataset-order evidence is mainly a SHA-derived vector of file identities, file sizes,
and a normalized data-order seed. It does not contain corpus statistics, W0 gradient probes,
activation probes, or any learned relationship between two data orders.

Architecture, tokenizer, and training-recipe evidence are constant across all ten lives. The
Compiler therefore had eight examples from which to map a mostly identity-like data-order digest
and W0 sketch to about 14 million endpoint values. The hidden failure is consistent with fitting
the eight visible life contexts without learning the training process.

This is an evidence-backed limitation of the current Compiler input and training set. It does not
show that the full GENOME idea is impossible.

## Round Two status

Do not describe the next run as “extra proof that the successful result did not memorize.” There
is no successful hidden result to confirm yet.

The next research work must first improve Compiler transfer. The main options supported by this
result are:

- replace hash-like dataset evidence with real corpus/order statistics and W0 gradient or
  activation probes;
- use the already pinned intermediate checkpoints as additional training supervision while
  keeping inference one-shot and prefix-free;
- add more independent complete lives before another hidden endpoint;
- give the Compiler cross-block or tensor-level context instead of predicting each block only
  from one local W0 block and one life context.

A new hidden seed or model life can then test the improved Compiler. Cross-size Round Two should
follow only after a hidden same-family endpoint works.

## Immutable artifacts

All production artifacts are on RunPod network volume `oom8nxsk5d` under
`/workspace/genome`. The Track 2 pod `pbqjwjoe1dtrnl` was stopped after evaluation.

| Artifact | Path | File or content SHA-256 |
|---|---|---|
| Source archive | `control/track2-d5cb97b.tar.gz` | `fcc28c046d41fb602bd1061af49495c72805e87d9bffb34f7194b4e78c947dce` |
| Source plan content | `control/source-plan.json` | `1cbbda9ef933d9ec79587e4d01da0e85f939af5443374f2e257911cb409225f5` |
| Block rate file | `runs/block-rate-9b2f68b.json` | `00951679ca6a2b2e06a8239925d79bdc571e66513d2f23ed49d69b2a39163be2` |
| Tensor rate file | `runs/tensor-rate-aab275f.json` | `0834eefb67cd7a7e28962c7757ff35c9b46eff472ac0ead61887968d13610d68` |
| SVD frontier file | `runs/development-svd-frontier-9320a76.json` | `7407f9e461abb606254b4d9a248013c7ec77fb6e725df03588cbe8fba60fcd20` |
| Quantized frontier file | `runs/development-quantized-frontier-5b16411.json` | `10db824c0d04994d9d9c6ffe15c6457130dc6863f5b10b6bc97fc9ab2edd8c9f` |
| V4 decoder manifest | `runs/decoder-v4/manifest.json` | `ad49600a8ef1e73cb5c2fa11b70a4724bbc8c4839f95c6f9715827f33934c97c` |
| V4 decoder audit file | `runs/decoder-evaluation-v4.json` | `c2b64b53ab0da1cd2a3dacdcb9ad35ab3e0d5f3c8043847530eb68a09b79d567` |
| Compiler manifest | `runs/compiler-v4/manifest.json` | `10bf1f05cd2d0342516d8516c2658110a1a792748a1f283321fb393383585e5a` |
| Hidden prediction manifest | `runs/hidden-prediction-v4/manifest.json` | `7315e2cfb35249086688086d6b6dc22f3aa827092ce0db1a6cc7beb22aea5e2b` |
| Hidden prediction seal | `runs/hidden-prediction-v4/prediction_seal.json` | `362fbd45abad3ed5b26cda5d591bbb55889e9d0f5bb8beeea35c481c4521ce73` |
| Pre-reveal Runtime manifest | `runs/hidden-runtime-v4/manifest.json` | `21d5a15ba6a8ce32b414d62bdb5445192b13285472c88730bde7f8583a49ae25` |
| Revealed corpus manifest | `data/canonical-revealed-v4/manifest.json` | `46c15aa0d2401e8a013b668d026080cc91c676299a8d785943e230846d014d7f` |
| Hidden Wikitext file | `runs/hidden-evaluation-v4.json` | `cadcea8af0ea80ccf250c67f70ac40cc40c759ecb782d3c2f38319cb6a9be3e3` |
| Hidden LM Harness file | `runs/hidden-lm-eval-v4.json` | `768aa892bd12c97cb810d989548bb0cd96ab0f6e26494592e1e06003fe8ecaf7` |

The local production source commit is `d5cb97b`.
