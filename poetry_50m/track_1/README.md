# poetry_50m / Track 1

This is a local, end-to-end personal research system: preparation preserves
rights and provenance; training is deterministic and resumable; evaluation is a
fixed three-seed suite; and transport is verified against held-out packed data
before it can alter a checkpoint. The current lineage is a three-source corpus:
Ultra-FineWeb-L3 English Multi-Style (synthetic rewriting, general prose), the
CC0-released Gutenberg line corpus (unconditional poetry NTP), and Poetry
Greats public-domain poems (conditional prompt-to-poem training). It excludes
BabyLM, Nano Wiki, Cerebras output, and every synthetic merge step. The
production model is an 8,335,008-parameter GPT with a 1,024-token context.

## Build and freeze the reviewed run

All artifacts live outside Git. The commit-pinned acquisition config and the
hash-priority selection policy are the corpus contract; do not replace either
source or selection with a convenient substitute.

```sh
uv sync --extra dev
uv run poetry50m corpus-acquire \
  --sources-config configs/data/huggingface_sources.json \
  --output artifacts/acquired
uv run poetry50m corpus-build \
  --acquisition artifacts/acquired \
  --sources-config configs/data/huggingface_sources.json \
  --selection-config configs/data/knowledge_corpus_selection.json \
  --output artifacts/corpus
uv run poetry50m prepare --corpus-manifest artifacts/corpus/manifest.jsonl \
  --prompts artifacts/corpus/prompts.jsonl \
  --thoughts artifacts/corpus/thoughts.jsonl \
  --pairings artifacts/corpus/pairings.jsonl \
  --config configs/data/corpus.json --output artifacts/prepared
uv run poetry50m plan-exposure --prepared artifacts/prepared \
  --model-config configs/model/track1_8m.yaml \
  --train-config configs/training/baseline.yaml --batch-size 4 \
  --tokens-per-parameter-per-pass 20 --passes 2 \
  --output artifacts/full-pretrain-plan
```

Review `artifacts/corpus/knowledge.receipt.json`, `artifacts/prepared/metadata.json`,
and `artifacts/full-pretrain-plan/receipt.json` before training. The plan must
show at least 333,400,320 unpadded data tokens and a 5/40/55 conditional/
general-prose/book-verse mix. Its `objective_exposure` records the actual mix,
each unique train pool's data-token count, and its planned repeat multiple.
Only after approval, use the derived frozen
`artifacts/full-pretrain-plan/train_config.json` with the same reviewed batch
size; `plan-exposure` itself never constructs a trainer or starts training.

R0 is the policy-free teacher: its observed trajectory may inform a method, but
not a sealed target. After inspecting R0, freeze
`configs/runs/personal_weekend.yaml` and verify that its
`trajectory_config_sha256` is the exact SHA-256 of
`configs/trajectory/first_branch.json`. Start R1/R2 only after that freeze.
Sealed training writes `run.manifest.json` and
`run.policy.commitment.json` before the first optimizer step. The companion
binds the policy, gates, exact held-out pack content, and selected supervised
token positions. Resume cannot switch a run between sealed and unsealed.

`analyze` receives at least two reference snapshots and a full target trainer
checkpoint at the same `W_t` step. `online` is retrospective within a
policy-bound target.
`level1` fits R0 but applies to distinct R1 with the same initialization and
data order; its current weights must exactly equal R0 at `W_t`. `level2` fits
R0 and evaluates a sealed R2 with the same initialization but an unseen
`--data-seed`. Level 1/2 reject continued endpoints. `--apply` is required to
alter R1. The post-leap gate clones the checkpoint optimizer, scheduler,
scaler, RNG, and exact future stream cursor. Its batch count, optimizer steps,
and retain/reset policy come only from the hash-bound run policy. It is
evaluated on separate fixed validation packs and never trains on that holdout.

```sh
uv run poetry50m analyze --prepared artifacts/prepared --model-config configs/model/track1_8m.yaml \
  --train-config configs/training/baseline.yaml --run-dir runs/r1 --batch-size 4 \
  --run-policy configs/runs/personal_weekend.yaml \
  --checkpoint runs/r1/checkpoints/final.pt \
  --snapshots runs/r0/trajectory/initial.pt runs/r0/trajectory/final.pt \
  --trajectory-config configs/trajectory/first_branch.json --target-step 500 \
  --scope level1 --reference-manifest runs/r0/run.manifest.json \
  --target-manifest runs/r1/run.manifest.json --method linear --apply \
  --output-dir runs/r1/transport
uv run poetry50m train --prepared artifacts/prepared --model-config configs/model/track1_8m.yaml \
  --train-config configs/training/baseline.yaml --run-dir runs/r1 --batch-size 4 \
  --run-policy configs/runs/personal_weekend.yaml --seal-endpoint \
  --resume runs/r1/transport/post_transport_checkpoint.pt
```

`endpoint-analyze` is a separate, offline R0 teacher diagnostic. It requires
initial, early, and endpoint snapshots (more snapshots add turning metrics).
It is endpoint-informed evidence and is forbidden as a sealed R2 fitting input:

```sh
uv run poetry50m endpoint-analyze \
  --snapshots runs/r0/trajectory/initial.pt runs/r0/trajectory/step_00000100.pt \
  runs/r0/trajectory/final.pt \
  --model-config configs/model/track1_8m.yaml \
  --output runs/r0/endpoint-geometry.json
```

`blind-pack` creates only blinded A/B rows and a separate unblinding key. Both
generation files must cover the same candidate-independent request IDs and
sampling settings. Each record retains its actual checkpoint content hash,
while `evaluation_manifest_id` appears only in the receipt. `blind-aggregate`
loads completed judgment JSONL (`comparison_id`, four `A`/`B`/`tie` rubric
fields, and `notes`) and writes unblinded tallies. `cost-report` accepts the six
ledger roles only as SHA-256-pinned references to immutable train/analysis
receipts plus one pinned resource receipt. It rejects invented or altered
timings. Unknown USD values remain JSON `null`.

A cost assembly has exactly `format_version`, `records`,
`resource_receipt`, and `amortized_uses`. `records` must name `reference`,
`analysis`, `checkpoint_io`, `verification_per_replay`, `replay`, and
`baseline_replay`; every value is
`{"receipt": "...", "sha256": "...", "estimated_cost_usd": null}`. Paths are
resolved relative to the assembly file. `estimated_cost_usd` is the total cost
attributable to that role, including accelerator and CPU cost; leave it `null`
when that total is unknown. The analysis receipt contains strict, non-overlapping
`analysis`, `checkpoint_io`, and `verification_per_replay` cost components.
Reference that same pinned receipt for those three roles; role-aware extraction
selects the corresponding component instead of counting the full command three
times. Obtain each hash with `shasum -a 256 <receipt>`, then run:

```sh
uv run poetry50m cost-report --input runs/cost-assembly.json \
  --output runs/cost-report.json
```

Train receipts distinguish cumulative and per-command processed/skipped/
supervised tokens, executed/virtual steps, trainer step time, full command
time through final artifact hashing, nullable accelerator time, backend-specific
memory semantics, and exact checkpoint/snapshot bytes and hashes. CUDA
`accelerator_seconds` is measured with CUDA events. MPS keeps that field `null`
because the workflow does not have a device-only MPS duration. On CUDA and MPS,
`device_active_wall_seconds` is synchronized elapsed time only for command
sections that exercise the accelerator; it includes host scheduling and is not
relabeled as accelerator time. CPU-only trajectory forecasting and checkpoint
I/O are excluded. Cost reports propagate an unknown unit as `null` while still
computing wall-clock, CPU, and available device-active totals, amortization, and
break-even. MPS peak working memory also remains `null`; current allocated
memory and its measurement semantics are recorded separately. Analysis receipts
separately bind the full analysis-source hash; the run manifest coordinate
signature includes only code that can alter initialized weights, optimizer
updates, scheduling, or prepared-stream coordinates.

## Input JSONL schemas

Every line is one UTF-8 JSON object. No blank records or unknown implicit
formats are accepted.

- Corpus manifest: `SourceDocument` with `document_id`, `provenance`, `text`,
  `blocks`, `source_path`, `metadata`, `raw_text`, and
  `transformation_lineage`; these eight top-level keys are all required and no
  others are accepted. `provenance` includes `work`, `author`, `licence`,
  `source`, `rights_status`, and evidence for public-domain/licensed/permission
  entries. Each block includes `block_id`, `kind`, and `text`;
  stanza/paragraph blocks retain their required IDs and spans. Synthetic
  material is rejected unless the data config explicitly sets
  `rights.allow_synthetic: true`.
  `SourceDocument` is also the split/leakage family: materialize one poem or one
  coherent philosophy passage per `document_id`, with stanzas/paragraphs below
  it. An omnibus book under one ID correctly remains wholly in one split.
- Prompts: `prompt_id`, `document_id`, `prompt`, `method` (`title`,
  `author_style`, `generic`, `theme`, `imagery`, `paraphrase`, or `passage`),
  `source_attribution`, and optional `poem_id`.
- Thoughts: `thought_id`, `document_id`, `text`, `method` (`passage`,
  `paraphrase`, or `editorial`), and `source_attribution`.
- Cross-document pairings: `pairing_id`, `target_document_id`,
  `target_block_id`, `prompt_id`, optional `thought_id`, and non-empty
  `transformation_lineage`.

Every prompt/poem relation produces a prompt-only
`<PROMPT>…<POEM>` training example matching the public generation interface.
When an attributed thought exists,
`<PROMPT>…<THOUGHT>…<POEM>` is an additional variant, never a replacement for
the prompt-only row.

Prepared output is canonical JSONL plus `metadata.json`; it records exact input
hashes, tokenizer hash, splits, pack hashes, and the allowed-synthetic policy.
All YAML/JSON configuration decoding is safe and rejects duplicate YAML keys
and unknown typed keys.

`score` writes first-pass per-pack loss. A non-shuffled curriculum must consume
that exact ledger with `--difficulty`; the CLI rejects missing or surplus pack
rows. Curriculum and ledger content are part of the stream identity, so a
checkpoint cannot silently resume under a different order.

The production data config assigns an intended **5/40/55 data-token mix**:
conditional Poetry Greats `0.05`, Ultra-FineWeb auxiliary prose NTP `0.4`, and
Gutenberg book-verse NTP `0.55`. The scheduler accounts for unpadded data tokens
after every batch, so varying pack lengths cannot silently turn this into a
batch-count ratio. Preparation fails if an enabled objective produces no
attributable training packs.

## Scope and storage

The 8M configuration is the production research target. A tiny fixture run
checks mechanics only: it cannot validate poetry quality, training scaling, or
transport utility. Keep acquired source snapshots and built corpus artifacts
outside the repository. The acquisition and build receipts bind every input,
revision, output, rights status, and transformation lineage. Budget at least
the source corpus size again for prepared JSONL and tokenizer artifacts; checkpoints and
trajectory snapshots each store all model weights, so retain only the cadence
needed for the chosen transport window.

Track 1 uses one production decoder: a conventional pre-norm causal transformer
with RMSNorm, RoPE, SwiGLU, and tied input/output embeddings.
`configs/model/track1_8m.yaml` is exactly **8,335,008** trainable parameters:
6 layers, width 288, 6 attention heads, FFN width 768, an 8,192-token
vocabulary, and a 1,024-token context.

The trainer is intentionally small but real: AdamW, cosine decay, gradient
accumulation, backend-aware mixed precision, deterministic seeding, checkpoint
resume (model, optimizer, scheduler, scaler, RNG and batch cursor), weights-only
trajectory snapshots, JSONL telemetry, per-example-loss callback, and optional
capture-cadence per-layer update geometry. A trajectory snapshot requires the
shared `SnapshotMetadata` contract (run/init/order/architecture/corpus/model/
tokenizer/code/training-config identities) and is atomically written in the
restricted `poetry50m.weights.v1` format used by trajectory analysis.

## Batch interface

The training stream yields mappings containing:

```python
{
    "input_ids": Tensor[batch, sequence],  # int32 or int64
    "targets": Tensor[batch, sequence],  # int32 or int64, -100 ignored
    "loss_mask": Tensor[batch, sequence],  # optional bool or float weights
    "example_ids": Sequence[str | int],  # optional, for the loss hook
    "data_token_count": int,  # optional exact non-padding input count
}
```

`input_ids` and `targets` are required by `Trainer`. `data_token_count` is the
exact non-padding count when a collator pads a batch; without it, the trainer
counts `input_ids.numel()` as a dense batch. Telemetry separately reports
processed data tokens and supervised loss tokens. Every example must retain at
least one target after applying `loss_mask`. A resumable stream implements
`__next__`, `state_dict()`, and `load_state_dict(state)`; `CyclingBatchStream`
is the small deterministic reference implementation.

## Minimal use

```python
from pathlib import Path
from poetry50m.model import DecoderOnlyTransformer, ModelConfig
from poetry50m.training import CyclingBatchStream, TrainConfig, Trainer

model = DecoderOnlyTransformer(
    ModelConfig(
        architecture="gpt",
        vocab_size=8192,
        max_seq_len=128,
        d_model=128,
        n_layers=2,
        n_heads=4,
        ffn_dim=512,
    )
)
trainer = Trainer(model, TrainConfig(max_steps=100, learning_rate=3e-4), Path("runs/demo"))
trainer.fit(CyclingBatchStream(batches))
```

The YAML configurations are plain validated data mappings. The project’s CLI
and data layers are responsible for decoding YAML and supplying batches; model
and training code intentionally do not own corpus policy.

## Validation

```sh
uv run python -m pytest
uv run python -m ruff format --check src tests
uv run python -m ruff check src tests
uv run python -m mypy src
```
