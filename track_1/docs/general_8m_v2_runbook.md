# General 8M V2 runbook

V2 keeps the V1 architecture, tokenizer, batch size, source revisions, source
order, and document split. It expands the V1 training selections and uses new
model and data-order seeds.

## Fixed design

- Model: `configs/model/general_8m.yaml`, 8,335,008 parameters.
- Tokenizer: the exact V1 8,192-token tokenizer.
- Pretraining: 1,500,020,736 unique training transitions, one epoch.
- Pretraining mix: 90% UltraFineWeb-L3 Multi-Style and 10% QA.
- Validation and test: the exact V1 document selections.
- SFT target: 60,000,000 supervised answer tokens from a superset of V1.
- Batch size: 32 for pretraining and SFT.

## Paths

```text
artifacts/general_8m/
  pretrain_v1/
  sft_v1/
  pretrain_v2/
  sft_v2/
runs/general_8m/
  pretrain/full-v1/
  pretrain/full-v2/
  sft/full-v1/
  sft/full-v2/
```

The RunPod volume root is `/workspace/track_1`.

## Preparation

```sh
PYTHONPATH=src python scripts/prepare_general_corpus.py \
  --config configs/data/general_8m_pretrain_v2.json \
  --tokenizer artifacts/general_8m/pretrain_v1/tokenizer.json \
  --output artifacts/general_8m/pretrain_v2 \
  --scratch /tmp/general8m-v2-prep

PYTHONPATH=src python scripts/prepare_general_sft.py \
  --config configs/data/general_8m_sft_v2.json \
  --tokenizer artifacts/general_8m/pretrain_v2/tokenizer.json \
  --output artifacts/general_8m/sft_v2
```

## Pretraining review and run

```sh
PYTHONPATH=src python scripts/validate_general_pretrain.py \
  --prepared artifacts/general_8m/pretrain_v2 \
  --model-config configs/model/general_8m.yaml \
  --train-config configs/training/general_8m_pretrain_v2_one_epoch.yaml \
  --data-seed 764905140 \
  --output artifacts/general_8m/pretrain_v2/pretrain_plan.json

PYTHONPATH=src python -m poetry50m.cli train \
  --prepared artifacts/general_8m/pretrain_v2 \
  --model-config configs/model/general_8m.yaml \
  --train-config configs/training/general_8m_pretrain_v2_one_epoch.yaml \
  --run-dir runs/general_8m/pretrain/full-v2 \
  --batch-size 32 \
  --data-seed 764905140
```

The model initialization seed is `396901101`.

## SFT review and run

After V2 pretraining finishes:

```sh
PYTHONPATH=src python -m poetry50m.cli sft-validate \
  --mixture artifacts/general_8m/sft_v2 \
  --tokenizer artifacts/general_8m/pretrain_v2/tokenizer.json \
  --base-checkpoint runs/general_8m/pretrain/full-v2/checkpoints/final.pt \
  --base-manifest runs/general_8m/pretrain/full-v2/run.manifest.json \
  --base-receipt runs/general_8m/pretrain/full-v2/train.receipt.json \
  --model-config configs/model/general_8m.yaml \
  --train-config artifacts/general_8m/sft_v2/one_epoch_train_config.json \
  --batch-size 32 \
  --data-seed 621155661 \
  --output artifacts/general_8m/sft_v2/sft_plan.json

PYTHONPATH=src python -m poetry50m.cli sft-train \
  --mixture artifacts/general_8m/sft_v2 \
  --tokenizer artifacts/general_8m/pretrain_v2/tokenizer.json \
  --base-checkpoint runs/general_8m/pretrain/full-v2/checkpoints/final.pt \
  --base-manifest runs/general_8m/pretrain/full-v2/run.manifest.json \
  --base-receipt runs/general_8m/pretrain/full-v2/train.receipt.json \
  --model-config configs/model/general_8m.yaml \
  --train-config artifacts/general_8m/sft_v2/one_epoch_train_config.json \
  --run-dir runs/general_8m/sft/full-v2 \
  --batch-size 32 \
  --data-seed 621155661
```

The SFT training seed is `134626435`.
