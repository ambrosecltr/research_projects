# GENOME Track 2

GENOME means **Generative Endpoint Neural Operator for Model Emission**.

Track 2 tests whether a learned system can generate a complete trained model from:

- the model's true random initial weights, `W0`;
- its architecture and tokenizer;
- a fingerprint of its dataset and data order;
- its complete training recipe.

The primary result is one-shot generation. It does not use an early training prefix, WT, repair, or polishing.

## Three separate components

1. **MGP Runtime** is deterministic code. It applies a genome to W0 and assembles a runnable model.
2. **Neural Genome Decoder** is a shared learned model. It expands a compact genome code into the full weight change `Delta-T`.
3. **GENOME Compiler** is a learned model. It predicts a genome code from W0 and the allowed model-life description.

Do not use “decoder” and “compiler” as names for the same component.

## What is already proven

The deterministic MGP path can encode a known W0-to-WT change and decode it back into a model. That proves representation mechanics. It does not prove that the compiler can predict an unseen WT.

## PolyPythia Round One

Round One uses ten complete standard, non-deduplicated Pythia 14M model lives:

- training: seed0 through seed7;
- development: seed8;
- hidden one-shot evaluation: seed9.

The public source plan pins all 154 checkpoints for each life. The primary experiment only materializes W0 and WT because intermediate checkpoints are forbidden compiler inputs and are not consumed by the decoder objective.

The hidden sequence is strict:

1. materialize hidden seed9 W0 only;
2. train the decoder and compiler without hidden WT;
3. predict, hash, and seal the hidden genome;
4. decode it and run a forward pass;
5. verify the sealed prediction and runtime execution record;
6. only then materialize hidden WT and evaluate.

See [POLYPYTHIA_ROUND1.md](POLYPYTHIA_ROUND1.md) for the exact commands and artifact order.

## Install and test

```bash
cd /Users/ambrosecoulter/research_projects/track_2
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev,evaluation]'
.venv/bin/python -m pytest -q
```

The production dependency range matches the RunPod PyTorch 2.4 image.

## Main command group

```bash
.venv/bin/genome polypythia --help
```

The command group covers:

- immutable source planning;
- sealed endpoint download;
- canonical GPT-NeoX conversion;
- shared decoder training and development-code fitting;
- decoder reconstruction evaluation;
- compiler training through the frozen decoder;
- sealed hidden genome prediction and runtime execution;
- post-reveal Wikitext and LM Evaluation Harness comparisons.

## Track 1 boundary

The new local 8M Track 1 model is not decoder or compiler training data for PolyPythia Round One. It remains outside the training split and can be used later for integration or hidden transfer evaluation.

The old `configs/poetry50m_track1.example.yaml` file is retained only for the previous 50M G0 compatibility path. It is not the full-life record for the new 8M model. See [TRACK1_INTEGRATION.md](TRACK1_INTEGRATION.md).

## Scientific claims

Keep these claims separate:

- **Proven:** a known endpoint can be represented and decoded by MGP.
- **Round One decoder question:** one shared neural decoder can compactly represent varied 14M model lives.
- **Round One compiler question:** the compiler can predict hidden seed9 from allowed evidence without WT.
- **Round Two question:** the result transfers across more sizes and is not a 14M-family memorization effect.
- **Later question:** transfer across datasets, recipes, and architecture families.

Parameter error is diagnostic. A successful model result also needs model execution and matched functional evaluation.
