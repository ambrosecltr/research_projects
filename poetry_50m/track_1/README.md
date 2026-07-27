# poetry_50m / Track 1

This is a local, end-to-end personal research system: preparation preserves
rights and provenance; training is deterministic and resumable; evaluation is a
fixed three-seed suite; and transport is verified against held-out packed data
before it can alter a checkpoint. Track 1's first lineage is the pinned,
publicly downloadable knowledge corpus described by
[`configs/data/huggingface_sources.json`](configs/data/huggingface_sources.json)
plus a separately generated and audited synthetic poetry corpus. The production
model is an 8,335,008-parameter GPT with a 1,024-token context. A tiny synthetic
run is a pipeline and quality-review gate, not a poetry result.

## Install and run

### Synthetic prompt/poem corpus

`scripts/generate_synthetic_corpus.py` builds a resumable, auditable synthetic
corpus with Cerebras `gpt-oss-120b`. Generation requests rotate through
temperature, reasoning, and creative lanes and use distinct integer seeds.
A separate blind critique call scores every candidate. Local gates enforce
length, repetition, exact and 12-token deduplication, and 12-token overlap
against the original corpus. Accepted records retain synthetic provenance;
this does not claim that model output cannot reproduce memorized text.

Set `CEREBRAS_API_KEY` in the environment. Start with an eight-request,
32-candidate pilot:

```sh
PILOT="runs/synthetic-cerebras-pilot-$(date +%Y%m%d-%H%M%S)"
uv run python scripts/generate_synthetic_corpus.py plan-generation \
  --config configs/data/synthetic_cerebras_8m_v1.json --requests 8 --output "$PILOT"
uv run python scripts/generate_synthetic_corpus.py run-sync \
  --requests "$PILOT/generation.requests.jsonl" \
  --results "$PILOT/generation.results.jsonl" --concurrency 8
uv run python scripts/generate_synthetic_corpus.py ingest-generation \
  --config configs/data/synthetic_cerebras_8m_v1.json \
  --requests "$PILOT/generation.requests.jsonl" \
  --results "$PILOT/generation.results.jsonl" --output "$PILOT"
uv run python scripts/generate_synthetic_corpus.py run-sync \
  --requests "$PILOT/critic.requests.jsonl" \
  --results "$PILOT/critic.results.jsonl" --concurrency 8
uv run python scripts/generate_synthetic_corpus.py finalize \
  --config configs/data/synthetic_cerebras_8m_v1.json \
  --candidates "$PILOT/candidates.jsonl" \
  --critic-results "$PILOT/critic.results.jsonl" \
  --output "$PILOT/corpus"
```

Both synchronous stages append and flush one result at a time, so rerunning a
command resumes missing request IDs. Treat the cached request and result JSONL
as the reproducibility boundary: provider inference is not assumed to be
byte-deterministic. Calls are paced by independent request and model-token
buckets at 95% of the stated Cerebras limits by default. Inspect the acceptance
ledger and a representative sample before calculating or approving the full
request count.

For a better model served by an OpenAI-compatible endpoint, skip the paid
critic pass but retain all deterministic local quality gates. The base URL must
be the API root ending in `/v1`:

```sh
export SYNTH_API_KEY="..."
SYNTH_BASE_URL="https://provider.example/v1"
SYNTH_MODEL="provider/model-name"
SMOKE="runs/synthetic-openai-smoke-$(date +%Y%m%d-%H%M%S)"

uv run python scripts/generate_synthetic_corpus.py plan-generation \
  --config configs/data/synthetic_cerebras_8m_v1.json \
  --requests 2 --output "$SMOKE" \
  --model "$SYNTH_MODEL" --openai-compatible
uv run python scripts/generate_synthetic_corpus.py run-openai-compatible \
  --requests "$SMOKE/generation.requests.jsonl" \
  --results "$SMOKE/generation.results.jsonl" \
  --base-url "$SYNTH_BASE_URL" --api-key-env SYNTH_API_KEY \
  --concurrency 2 --requests-per-minute 60 --tokens-per-minute 100000
uv run python scripts/generate_synthetic_corpus.py ingest-generation \
  --config configs/data/synthetic_cerebras_8m_v1.json \
  --requests "$SMOKE/generation.requests.jsonl" \
  --results "$SMOKE/generation.results.jsonl" \
  --output "$SMOKE" --skip-critic
uv run python scripts/generate_synthetic_corpus.py finalize-local \
  --config configs/data/synthetic_cerebras_8m_v1.json \
  --candidates "$SMOKE/candidates.jsonl" \
  --output "$SMOKE/corpus"
```

If the endpoint uses the legacy completion-token field, add
`--max-tokens-field max_tokens` to `plan-generation`. If it does not implement
strict JSON Schema output, add `--response-format json-object` or, as a last
resort, `--response-format none`. The prompt still requires a JSON object.

Acquire and build the pinned knowledge sources, then merge them with an
approved synthetic corpus:

```sh
uv sync --extra dev
uv run poetry50m corpus-acquire \
  --sources-config configs/data/huggingface_sources.json \
  --output artifacts/acquired
uv run poetry50m corpus-build \
  --acquisition artifacts/acquired \
  --sources-config configs/data/huggingface_sources.json \
  --output artifacts/knowledge
uv run python scripts/generate_synthetic_corpus.py merge \
  --base-manifest artifacts/knowledge/manifest.jsonl \
  --base-prompts artifacts/knowledge/prompts.jsonl \
  --base-thoughts artifacts/knowledge/thoughts.jsonl \
  --base-pairings artifacts/knowledge/pairings.jsonl \
  --synthetic "$PILOT/corpus" --output artifacts/corpus
uv run poetry50m prepare --corpus-manifest artifacts/corpus/manifest.jsonl \
  --prompts artifacts/corpus/prompts.jsonl \
  --thoughts artifacts/corpus/thoughts.jsonl \
  --pairings artifacts/corpus/pairings.jsonl \
  --config configs/data/corpus.json --output artifacts/prepared
```

Do not scale corpus generation or repeat the full 100,000-step train solely
because these mechanics pass. First require a small held-out training probe to
improve prompt-keyword relevance and reduce empty prompt matches relative to
R0.

```sh
uv run poetry50m train --prepared artifacts/prepared --model-config configs/model/track1_8m.yaml \
  --train-config configs/training/baseline.yaml --run-dir runs/r0 --batch-size 4 \
  --until-step 250
uv run poetry50m train --prepared artifacts/prepared --model-config configs/model/track1_8m.yaml \
  --train-config configs/training/baseline.yaml --run-dir runs/r1 --batch-size 4 \
  --run-policy configs/runs/personal_weekend.yaml \
  --until-step 250 --seal-endpoint
uv run poetry50m train --prepared artifacts/prepared --model-config configs/model/track1_8m.yaml \
  --train-config configs/training/baseline.yaml --run-dir runs/r2 --batch-size 4 \
  --run-policy configs/runs/personal_weekend.yaml \
  --data-seed 7331 --until-step 250 --seal-endpoint
uv run poetry50m score --prepared artifacts/prepared --model-config configs/model/track1_8m.yaml \
  --train-config configs/training/baseline.yaml --run-dir runs/r0 --batch-size 4 \
  --checkpoint runs/r0/checkpoints/final.pt --output runs/r0/difficulty.jsonl
uv run poetry50m train --prepared artifacts/prepared --model-config configs/model/track1_8m.yaml \
  --train-config configs/training/baseline.yaml --run-dir runs/curriculum --batch-size 4 \
  --curriculum strict_hard_to_easy --difficulty runs/r0/difficulty.jsonl
uv run poetry50m generate --prepared artifacts/prepared --model-config configs/model/track1_8m.yaml \
  --train-config configs/training/baseline.yaml --run-dir runs/r0 --batch-size 4 \
  --checkpoint runs/r0/checkpoints/final.pt --suite configs/evaluation/prompt_suite.json \
  --evaluation-manifest-id track1-fixed-v1 --output runs/r0/generations.jsonl
uv run poetry50m metrics --prepared artifacts/prepared --suite configs/evaluation/prompt_suite.json \
  --records runs/r0/generations.jsonl --manifest runs/r0/generations.manifest.json \
  --output runs/r0/metrics.json
uv run poetry50m eval-loss --prepared artifacts/prepared --model-config configs/model/track1_8m.yaml \
  --train-config configs/training/baseline.yaml --run-dir runs/r0 --batch-size 4 \
  --checkpoint runs/r0/checkpoints/final.pt --split test --output runs/r0/test-loss.json
```

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

The production data config assigns an intended **80/20 objective-batch mix**:
conditional poetry `1.0` and auxiliary prose NTP `0.25`, or four conditional
batches per prose batch. This is a stream-scheduling ratio, not a raw-document
or token ratio. The prepared-artifact report is the record of the realized
packed supervised-token ratio; report that value with every run. Preparation
fails if this objective is enabled but no training prose exists.

## Scope and storage

The 8M configuration is the production research target. A synthetic tiny run
checks mechanics only: it cannot validate poetry quality, training scaling, or
transport utility. Keep acquired source snapshots and built corpus artifacts
outside the repository. The acquisition and build receipts bind every input,
revision, output, rights status, and transformation lineage. BabyLM-distilled
rows remain explicitly `unknown` for rights because their upstream
per-document provenance was not retained. Budget at least the source corpus
size again for prepared JSONL and tokenizer artifacts; checkpoints and
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
    "targets": Tensor[batch, sequence],    # int32 or int64, -100 ignored
    "loss_mask": Tensor[batch, sequence],  # optional bool or float weights
    "example_ids": Sequence[str | int],    # optional, for the loss hook
    "data_token_count": int,               # optional exact non-padding input count
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

model = DecoderOnlyTransformer(ModelConfig(
    architecture="gpt", vocab_size=8192, max_seq_len=128, d_model=128,
    n_layers=2, n_heads=4, ffn_dim=512,
))
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
