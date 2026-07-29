# General 8M runbook

This lineage is separate from the poetry 8M lineage.

## Storage

```text
artifacts/
  poetry_8m/
  general_8m/
    pretrain_v1/
    sft_v1/
runs/
  poetry_8m/
  general_8m/
    pretrain/
    sft/
```

The RunPod volume root is `/workspace/track_1`.

## Fixed source identities

- Pretraining: `openbmb/Ultra-FineWeb-L3` at commit
  `c68ab81ad03b2d2f476fa8ab3c72bed3528da359`.
- SFT: `TIGER-Lab/Fineweb-Instruct` at commit
  `edaa76cc99247babe39223abddcbebe35c158f50`.
- Fineweb-Instruct `train.jsonl`:
  `03de49736b6078943fe25b84a47189e483c29258e655756e62b26e1a61b34a52`.

## Pretraining corpus

`configs/data/general_8m_pretrain.json` builds:

- 439,456 Multi-Style rows and 48,832 QA rows for training;
- 488,288 total training rows;
- 500,006,912 training transitions from deduplicated source documents;
- one full pass at 15,259 optimizer steps with batch size 32;
- document-level 90/5/5 train, validation, and test assignment;
- exact UID and normalized-content deduplication;
- an 8,192-token byte-fallback BPE tokenizer with general chat role tokens.

The compact files use fixed-width little-endian `uint16` rows. Training memory-maps
the token file and shuffles row indices with the recorded data seed.

Preparation:

```sh
PYTHONPATH=src python scripts/prepare_general_corpus.py \
  --config configs/data/general_8m_pretrain.json \
  --output artifacts/general_8m/pretrain_v1 \
  --scratch /tmp/general8m-prep
```

Review validation:

```sh
PYTHONPATH=src python scripts/validate_general_pretrain.py \
  --prepared artifacts/general_8m/pretrain_v1 \
  --model-config configs/model/general_8m.yaml \
  --train-config configs/training/general_8m_pretrain_one_epoch.yaml \
  --data-seed 1521710748 \
  --output artifacts/general_8m/pretrain_v1/pretrain_plan.json
```

The model initialization seed is `82635239`. The independent training-order seed
is `1521710748`.

## Pretraining command

Run this only after the prepared artifact and hardware choice are reviewed:

```sh
PYTHONPATH=src python -m poetry50m.cli train \
  --prepared artifacts/general_8m/pretrain_v1 \
  --model-config configs/model/general_8m.yaml \
  --train-config configs/training/general_8m_pretrain_one_epoch.yaml \
  --run-dir runs/general_8m/pretrain/full-v1 \
  --batch-size 32 \
  --data-seed 1521710748
```

Track 1 writes the initial weights, configured intermediate trajectory snapshots,
the final trajectory snapshot, resumable checkpoints, the run manifest, telemetry,
and the training receipt.

Fixed held-out loss:

```sh
PYTHONPATH=src python scripts/evaluate_general_loss.py \
  --prepared artifacts/general_8m/pretrain_v1 \
  --model-config configs/model/general_8m.yaml \
  --checkpoint runs/general_8m/pretrain/full-v1/checkpoints/final.pt \
  --split validation \
  --batch-size 32 \
  --batches 128 \
  --output runs/general_8m/pretrain/full-v1/evaluation/validation-loss.json
```

## SFT preparation

The current reviewed preparation target is 20,000,000 supervised answer tokens.
It uses response-only loss and generates a one-pass training configuration from
the exact prepared row count.

```sh
PYTHONPATH=src python scripts/prepare_general_sft.py \
  --config configs/data/general_8m_sft.json \
  --tokenizer artifacts/general_8m/pretrain_v1/tokenizer.json \
  --output artifacts/general_8m/sft_v1
```

The SFT training seed is `2040893737`. Its independent data-order seed is
`1399503269`.

After pretraining, validate the exact base lineage:

```sh
PYTHONPATH=src python -m poetry50m.cli sft-validate \
  --mixture artifacts/general_8m/sft_v1 \
  --tokenizer artifacts/general_8m/pretrain_v1/tokenizer.json \
  --base-checkpoint runs/general_8m/pretrain/full-v1/checkpoints/final.pt \
  --base-manifest runs/general_8m/pretrain/full-v1/run.manifest.json \
  --base-receipt runs/general_8m/pretrain/full-v1/train.receipt.json \
  --model-config configs/model/general_8m.yaml \
  --train-config artifacts/general_8m/sft_v1/one_epoch_train_config.json \
  --batch-size 32 \
  --data-seed 1399503269 \
  --output artifacts/general_8m/sft_v1/sft_plan.json
```

SFT training:

```sh
PYTHONPATH=src python -m poetry50m.cli sft-train \
  --mixture artifacts/general_8m/sft_v1 \
  --tokenizer artifacts/general_8m/pretrain_v1/tokenizer.json \
  --base-checkpoint runs/general_8m/pretrain/full-v1/checkpoints/final.pt \
  --base-manifest runs/general_8m/pretrain/full-v1/run.manifest.json \
  --base-receipt runs/general_8m/pretrain/full-v1/train.receipt.json \
  --model-config configs/model/general_8m.yaml \
  --train-config artifacts/general_8m/sft_v1/one_epoch_train_config.json \
  --run-dir runs/general_8m/sft/full-v1 \
  --batch-size 32 \
  --data-seed 1399503269
```
