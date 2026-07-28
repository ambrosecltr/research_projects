# Chunked synthetic poetry SFT data

This pipeline creates one prompt and one generated poem per SFT example. It has no
judge or critic model. NanoWhale is not a requirement or encoded dependency; its
published training choices do not constrain this dataset.

The default corpus target is 15,000,000 **formatted sequence tokens**, measured with
the exact frozen `prepared_v3` Track 1 tokenizer. Receipts separately report:

- formatted tokens: prompt + response + four boundary tokens;
- training-input tokens: formatted tokens minus one;
- supervised tokens: response + EOS.

Two passes therefore expose `2 × training_input` input tokens and `2 × supervised`
loss-bearing targets. They are not automatically 30M supervised tokens.

## Generate independent chunks

Each chunk owns a disjoint global example-index range and one generator model. Use the
same seed across the corpus, change the model/provider between chunks, and never reuse
an index range. Assembly enforces one recipe, seed, and tokenizer across all chunks.

```bash
uv run python scripts/generate_synthetic_sft.py plan-chunk \
  --output artifacts/sft_synthetic/raw/chunk-000 \
  --model provider/model-a \
  --provider provider-name \
  --start-index 0 \
  --examples 1024

SYNTH_API_KEY=... uv run python scripts/generate_synthetic_sft.py \
  run-openai-compatible \
  --chunk artifacts/sft_synthetic/raw/chunk-000 \
  --base-url https://provider.example/v1 \
  --api-key-env SYNTH_API_KEY

uv run python scripts/generate_synthetic_sft.py finalize-chunk \
  --chunk artifacts/sft_synthetic/raw/chunk-000 \
  --tokenizer artifacts/prepared_v3/tokenizer.json \
  --output artifacts/sft_synthetic/final/chunk-000
```

`run-openai-compatible` is resumable. Successful responses are appended to
`results.jsonl`; rerunning sends only missing request IDs. A chunk plan cannot be
overwritten, its model and prompt assignments are hash-bound, and finalization fails
if any response is missing, reordered, or malformed. Dispatch also records the
non-secret base URL, rate settings, and timeout, and refuses to resume the same chunk
with different settings. Final receipts retain both the requested model and the model
identifier returned by the provider, when supplied.

Start each provider/model with a small exact-configuration smoke chunk and conservative
rate limits. The runner preserves completed requests after transient failures; rerun
the same command to send the remaining request IDs. It does not automatically retry
with backoff, and no universal idempotency header exists across OpenAI-compatible
providers, so a provider-side completion followed by a local timeout can be billed
again. Keep `examples-per-request` small enough that all poems and JSON fit inside the
single 4096-token response budget.

For the next model, use the next range:

```bash
uv run python scripts/generate_synthetic_sft.py plan-chunk \
  --output artifacts/sft_synthetic/raw/chunk-001 \
  --model other-provider/model-b \
  --provider other-provider \
  --start-index 1024 \
  --examples 1024
```

Providers that do not support strict JSON schema can use `--response-format
json-object`; verify that mode in the smoke first. Providers using the older token
field can add `--max-tokens-field max_tokens`.

Finalization uses no judge. It only applies deterministic structural gates: requested
line range, sane word count, no Markdown fence/control characters/refusal boilerplate,
no repeated line, and normalized exact-response deduplication. Rejected examples and
their reasons are retained in `rejections.jsonl`; accepted examples remain usable, and
later disjoint chunks fill the token shortfall.

## Check progress and assemble

```bash
uv run python scripts/generate_synthetic_sft.py summarize \
  --receipts artifacts/sft_synthetic/final/chunk-*/receipt.json \
  --output artifacts/sft_synthetic/progress.json

uv run python scripts/generate_synthetic_sft.py assemble \
  --receipts artifacts/sft_synthetic/final/chunk-*/receipt.json \
  --output artifacts/sft_synthetic/dataset-v1
```

`summarize` adds already accepted per-chunk counts before cross-chunk response
deduplication, so generate a modest buffer above the displayed target. Final assembly
is the authoritative deduplicated token count and fails closed if that count is short.

Assembly sorts by global example index, rejects overlapping ranges, removes exact
duplicate responses deterministically, and stops after the first complete example
that reaches 15,000,000 formatted tokens. It never truncates a poem. Use
`--target-metric supervised` only if the approved budget is changed to loss-bearing
response tokens rather than total formatted SFT tokens. Assembly exits with an error
if the validated, deduplicated input remains below target; `summarize` is the normal
progress command. `--allow-under-target` exists only for an explicitly requested
partial export.

The assembled JSONL retains `messages`, prompt controls, exact token counts, generator
model/provider provenance, chunk identity, request identity, recipe version, and seed.
The later SFT preparation step should encode these pairs with the existing
`<|bos|><|prompt|>...<|poem|>...<|eos|>` contract and supervise only poem plus EOS.
