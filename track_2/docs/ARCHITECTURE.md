# GENOME Compiler architecture

## Inputs

### Global task vector

A fixed-size vector combines:

- corpus token/bigram sketches;
- byte and length statistics;
- W0 loss/gradient/activation summaries;
- tokenizer statistics;
- numeric/categorical training-recipe features.

### Tensor records

One record per logical state tensor contains:

- role;
- layer position;
- shape and parameter count;
- W0 mean, variance, magnitude and norm;
- row/column W0 statistics;
- tie owner where applicable.

### Architecture graph

Edges connect tensors within one layer, adjacent layers and global embedding/output/norm tensors.

## Network

1. Project global and tensor features into `d_model`.
2. Apply graph message-passing blocks.
3. Prepend the global token.
4. Apply a bidirectional Transformer encoder.
5. Predict one primitive distribution and one rank distribution per tensor.
6. Use shared coordinate heads to generate factor rows/columns or vector values.
7. Allocate factors under the hard target byte budget.
8. Materialize one deterministic MGP.

The coordinate heads are shared across all tensors and model sizes. They are part of the compiler, not a second model or Runtime.

## Output complexity

For a selected rank-r matrix, generated values scale as:

```text
r * (output_width + input_width)
```

rather than:

```text
output_width * input_width
```

A manifest reserve is subtracted before payload allocation so serialized overhead cannot silently violate the 10% gate.

## Training

Teacher-forced training uses accepted compact programs.

- primitive cross-entropy;
- rank cross-entropy;
- decoded displacement loss;
- periodic task loss through the real model;
- prediction-dependent byte penalty.

The decoded displacement objective is invariant to rotations/sign changes that leave the factor product unchanged.

## Inference

Inference uses no WT and no target program. It receives only the hidden life’s W0, architecture, evidence and recipe. It produces a bounded MGP, then the deterministic Runtime produces the candidate state.
