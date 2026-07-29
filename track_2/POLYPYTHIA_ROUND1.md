# PolyPythia 14M Round One runbook

## Questions

Round One answers two different questions.

1. Can one shared Neural Genome Decoder compactly represent the endpoints of several independent 14M model lives?
2. Can the GENOME Compiler predict a useful genome for hidden seed9 without seeing seed9 WT?

Round Two later adds model sizes. It tests whether the Round One result was only a 14M-family rule.

## Fixed split

| Use | Lives |
|---|---|
| Decoder and compiler training | seed0 through seed7 |
| Development and model selection | seed8 |
| Hidden one-shot result | seed9 |

The source is the standard non-deduplicated Pythia family. Do not mix it with `pythia-14m-deduped`.

## Local verification

```bash
cd /Users/ambrosecoulter/research_projects/track_2
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -e '.[dev,evaluation]'
.venv/bin/python -m compileall -q genome tests
.venv/bin/python -m pytest -q
```

Create the immutable source plan:

```bash
.venv/bin/genome polypythia plan \
  --config configs/polypythia_14m_round1.yaml \
  --output artifacts/polypythia_14m_round1/source-plan.json
```

The plan pins all 1,540 checkpoint branches. The primary download contains only W0 and WT.

## RunPod names

Use only these Track 2 names:

- network volume: `track2-polypythia-r1`;
- pod prefix: `track2-polypythia-r1-`;
- volume workspace: `/workspace/genome`.

Do not modify any Track 1 pod or volume.

## Volume layout

```text
/workspace/genome/
  control/
  research_projects/
  data/
    downloads/
    canonical-sealed/
    canonical-revealed/
    evaluation-texts/
  runs/
    decoder/
    development-code/
    decoder-evaluation.json
    compiler/
    hidden-prediction/
    hidden-runtime/
    hidden-evaluation.json
    hidden-lm-eval.json
```

Install the project in the pod's container environment. Keep Python environments off the network volume.

## Sealed execution

Set paths:

```bash
export GENOME_ROOT=/workspace/genome
export TRACK2_SRC=$GENOME_ROOT/research_projects/track_2
export ROUND1_PLAN=$GENOME_ROOT/control/source-plan.json
export ROUND1_DOWNLOAD=$GENOME_ROOT/data/downloads
export ROUND1_SEALED=$GENOME_ROOT/data/canonical-sealed
export ROUND1_RUNS=$GENOME_ROOT/runs

cd "$TRACK2_SRC"
python -m pip install -e '.[dev,evaluation]'
python -m pytest -q
python -m pip freeze > "$GENOME_ROOT/control/python-freeze.txt"
```

Download training and development endpoint pairs plus hidden W0:

```bash
genome polypythia download \
  --plan "$ROUND1_PLAN" \
  --output-root "$ROUND1_DOWNLOAD"
```

Prepare the sealed canonical corpus:

```bash
genome polypythia prepare \
  --config configs/polypythia_14m_round1.yaml \
  --plan "$ROUND1_PLAN" \
  --receipt "$ROUND1_DOWNLOAD/download-sealed.json" \
  --download-root "$ROUND1_DOWNLOAD" \
  --output "$ROUND1_SEALED"
```

Prepare the full pinned Wikitext test corpus before evaluation:

```bash
genome polypythia prepare-evaluation-texts \
  --config configs/polypythia_14m_round1.yaml \
  --cache-dir "$GENOME_ROOT/data/huggingface-cache" \
  --output "$GENOME_ROOT/data/evaluation-texts"
```

Train and audit the shared decoder:

```bash
genome polypythia train-decoder \
  --config configs/polypythia_14m_round1.yaml \
  --corpus "$ROUND1_SEALED" \
  --output "$ROUND1_RUNS/decoder" \
  --device cuda

genome polypythia fit-development-code \
  --config configs/polypythia_14m_round1.yaml \
  --corpus "$ROUND1_SEALED" \
  --shared-decoder "$ROUND1_RUNS/decoder" \
  --output "$ROUND1_RUNS/development-code" \
  --device cuda

genome polypythia evaluate-decoder \
  --corpus "$ROUND1_SEALED" \
  --shared-decoder "$ROUND1_RUNS/decoder" \
  --development-code "$ROUND1_RUNS/development-code" \
  --model-config "$ROUND1_DOWNLOAD/tokenizer/config.json" \
  --tokenizer "$ROUND1_DOWNLOAD/tokenizer" \
  --evaluation-texts "$GENOME_ROOT/data/evaluation-texts/texts.jsonl" \
  --output "$ROUND1_RUNS/decoder-evaluation.json" \
  --device cuda
```

Train the compiler and create the hidden one-shot prediction:

```bash
genome polypythia train-compiler \
  --config configs/polypythia_14m_round1.yaml \
  --corpus "$ROUND1_SEALED" \
  --shared-decoder "$ROUND1_RUNS/decoder" \
  --output "$ROUND1_RUNS/compiler" \
  --device cuda

genome polypythia predict-hidden \
  --corpus "$ROUND1_SEALED" \
  --shared-decoder "$ROUND1_RUNS/decoder" \
  --compiler "$ROUND1_RUNS/compiler" \
  --output "$ROUND1_RUNS/hidden-prediction" \
  --device cuda

genome polypythia execute-hidden \
  --corpus "$ROUND1_SEALED" \
  --shared-decoder "$ROUND1_RUNS/decoder" \
  --prediction "$ROUND1_RUNS/hidden-prediction" \
  --model-config "$ROUND1_DOWNLOAD/tokenizer/config.json" \
  --output "$ROUND1_RUNS/hidden-runtime" \
  --device cuda
```

At this point `hidden-prediction/prediction_seal.json` exists and the predicted model has completed a forward pass. Do not reveal hidden WT before both artifacts exist.

## Reveal and final evaluation

Download hidden WT only after the seal:

```bash
genome polypythia download \
  --plan "$ROUND1_PLAN" \
  --output-root "$ROUND1_DOWNLOAD" \
  --reveal-hidden \
  --prediction-seal "$ROUND1_RUNS/hidden-prediction/prediction_seal.json" \
  --runtime-execution "$ROUND1_RUNS/hidden-runtime/manifest.json"
```

Prepare a separate revealed corpus:

```bash
genome polypythia prepare \
  --config configs/polypythia_14m_round1.yaml \
  --plan "$ROUND1_PLAN" \
  --receipt "$ROUND1_DOWNLOAD/download-revealed.json" \
  --download-root "$ROUND1_DOWNLOAD" \
  --output "$GENOME_ROOT/data/canonical-revealed"
```

Run the full matched Wikitext evaluation:

```bash
genome polypythia evaluate-hidden \
  --sealed-corpus "$ROUND1_SEALED" \
  --revealed-corpus "$GENOME_ROOT/data/canonical-revealed" \
  --runtime-execution "$ROUND1_RUNS/hidden-runtime" \
  --model-config "$ROUND1_DOWNLOAD/tokenizer/config.json" \
  --tokenizer "$ROUND1_DOWNLOAD/tokenizer" \
  --evaluation-texts "$GENOME_ROOT/data/evaluation-texts/texts.jsonl" \
  --output "$ROUND1_RUNS/hidden-evaluation.json" \
  --device cuda
```

Run the full pinned zero-shot task suite:

```bash
genome polypythia evaluate-lm-harness \
  --config configs/polypythia_14m_round1.yaml \
  --sealed-corpus "$ROUND1_SEALED" \
  --revealed-corpus "$GENOME_ROOT/data/canonical-revealed" \
  --runtime-execution "$ROUND1_RUNS/hidden-runtime" \
  --model-config "$ROUND1_DOWNLOAD/tokenizer/config.json" \
  --tokenizer "$ROUND1_DOWNLOAD/tokenizer" \
  --output "$ROUND1_RUNS/hidden-lm-eval.json" \
  --device cuda
```

## Interpretation

A useful Round One result needs all of these:

- one shared decoder, not one decoder per life;
- development seed8 reconstruction with the decoder frozen;
- a hidden seed9 genome produced without WT;
- successful pre-reveal model execution;
- predicted WT materially better than W0;
- predicted WT close to true WT on matched functional evaluations;
- honest byte and compute accounting.

Round One does not prove cross-size or cross-family transfer. That is Round Two and later work.
