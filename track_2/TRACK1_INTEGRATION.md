# Track 1 integration boundary

## Current rule

The local Track 1 model is legacy G0 and future evaluation-only.

It is not part of the active public-life compiler-training split.

## Required full-life record

The 8M Track 1 record must describe one continuous model life:

```text
true random W0
  -> pretraining
  -> optional SFT, RL, RF, or other later stages
  -> final WT
```

The final WT is `sft-15m-two-epoch` for the current life. Its W0 is the random seed checkpoint from before pretraining. The pretrained checkpoint is not W0.

The pretraining-to-SFT transformation can also be retained as an auxiliary stage-local segment:

```text
pretrained checkpoint -> final SFT checkpoint
```

That auxiliary segment does not replace the full-life boundary.

## Exact data to retain

Do not infer missing values. Obtain them from the Track 1 artifacts and receipts:

- true random W0 state and hash;
- final WT state and hash;
- architecture and tokenizer files;
- pretraining and SFT dataset identities;
- data order and split identities;
- complete pretraining and SFT recipes;
- random seeds;
- stage boundaries and checkpoint hashes;
- environment and code revision;
- functional evaluation records.

## Existing legacy file

`configs/poetry50m_track1.example.yaml` describes the retired 50M integration path. Keep it while code or old G0 artifacts still reference it. Do not present it as the full-life record for the 8M model.

When the Track 1 agent finishes the exact 8M handoff, create a new multi-stage life record or migrate the legacy name only after all references are checked.

## Later use

After hidden same-family public-life compilation passes:

1. verify the 8M native-to-canonical-to-native conversion;
2. verify MGP encode/decode, tensor shapes, tensor ties, and model execution;
3. keep its endpoint outside decoder and compiler training for the chosen evaluation;
4. compile from W0 plus allowed evidence;
5. reveal WT only after the prediction is sealed.

That is compatibility verification and transfer evaluation. It is not a repeat of the known MGP arithmetic experiment.
