# GENOME Pythia v1 experiment plan

## Gate 0 — clean workspace

Create a new RunPod network volume and initialize `/workspace/genome_v1`. Do not copy any pre-existing GENOME workspace.

Pass conditions:

- source, derived, compiler and run directories are separate;
- no hidden WT exists;
- exact branch commit and environment are recorded.

## Gate 1 — source audit and pinning

Use `configs/sources/pythia_v1.json`.

Resolve all W0 refs and all training/development WT refs to immutable Hugging Face commits. Do not resolve hidden 31M seed9 WT.

Pass conditions:

- 17 training lives, including Pythia 14M seed9;
- two development lives;
- one hidden life;
- Pythia 14M seed9 W0 and WT resolved for training-data preparation;
- every materialized file has SHA-256 and byte receipt;
- source directory is made read-only after materialization.

## Gate 2 — canonical lives

For every available life:

1. Load native GPT-NeoX state.
2. Convert to canonical state.
3. Convert back to native.
4. Require exact tensor equality.
5. Compare logits on fixed token sequences.
6. Export architecture graph and tensor inventory.
7. Write a canonicalization audit for each life.

No intermediate checkpoints are downloaded in v1.
The complete `GENOME_MODEL_LIFE` manifest is finalized in Gate 3 after its real
semantic evidence exists.

## Gate 3 — semantic evidence

Build one pinned deterministic sample from the exact standard-Pile Pythia token
binary, shared across lives, plus W0-specific response evidence. Record the
immutable dataset commit, source shard, aligned byte range and content hashes.

Corpus evidence:

- token unigram CountSketch;
- token bigram CountSketch;
- byte frequency;
- sequence-length histogram;
- tokenizer properties.

Raw-byte evidence is computed from deterministic tokenizer decoding of the exact
training tokens because the published Pythia binary contains tokens, not original
raw text. Record this limitation in the sample receipt.

Partition the pinned token sample by zero-based record parity. Even records are
the functional-refinement probe. Odd records are the fixed evaluation sample.
Record both hashes and counts. The two views must remain disjoint.

W0 evidence:

- fixed probe loss distribution;
- per-role gradient CountSketch;
- hidden-state moments and quantiles.

Recipe evidence:

- Pythia architecture and optimizer schedule;
- planned steps and tokens;
- context and global batch;
- known order metadata.

Pass conditions:

- repeated construction is identical;
- no WT is read;
- no hash bytes enter semantic tensors;
- hidden evidence construction cannot open hidden WT paths.

After these checks pass, finalize one complete `GENOME_MODEL_LIFE` manifest for
each life from its canonical artifacts, semantic evidence and pinned recipe.

## Gate 4 — compact target frontier

For every training and development life fit candidates at:

```text
1%, 2.5%, 5%, 7.5%, 10%, 15%, 20%
```

Start with globally budgeted randomized low rank plus bounded quantized vectors. Use CUDA SVD where useful. Functionally refine only compact coefficients through the Runtime.
If global energy allocation leaves eligible matrices at BASE_COPY and fails the
functional gate, evaluate a deterministic balanced variant that reserves rank
one per eligible matrix before allocating remaining bytes by energy.
If that still fails, evaluate a rank-balanced variant that allocates the same
singular-component index across eligible matrices before moving to the next
index.

Evaluate every candidate on the identical fixed odd-record Pythia evaluation
sample. Functional refinement may read only the even-record probe.

Save for each candidate:

- exact MGP bytes;
- primitive/rank table;
- decode time;
- parameter distortion;
- W0, candidate and WT loss;
- endpoint progress;
- logit KL and top-1 agreement;
- accept/reject reason.

Production target gate:

```text
serialized fraction <= 10%
endpoint progress >= 0.80
candidate beats W0
finite model
```

Both 14M seed8 and 31M seed8 must pass. Otherwise stop and improve the deterministic formula language. Do not train the compiler.

## Gate 5 — compiler corpus

Create `GENOME_COMPILER_CORPUS` from accepted training and development programs only.
The expected complete corpus has 19 records: 17 training and two development.
Pythia 14M seed9 contributes an endpoint-free compiler input and an accepted fitted target.

Each record contains:

- architecture graph;
- W0 state;
- semantic fingerprint;
- numeric recipe;
- accepted compact MGP;
- fixed functional probe JSONL;
- model config.

No hidden record is present.

## Gate 6 — tiny compiler smoke

Use a small subset of accepted lives.

Require:

- training loss decreases;
- primitive/rank accuracy improves;
- decoded-delta loss decreases;
- functional loss is finite;
- generated programs are structurally valid;
- actual program bytes remain bounded;
- compiler checkpoint and optimizer state are written;
- a stopped run resumes correctly;
- development examples never receive gradients.

## Gate 7 — production compiler training

Train the configuration in `configs/compiler/pythia_v1.yaml`.

Monitor:

- total, primitive, rank, decoded-delta and functional losses;
- predicted byte proxy and actual sampled MGP bytes;
- invalid-program count, which must remain zero;
- development loss and development endpoint progress;
- gradient norm, GPU memory and throughput.

Select the frozen compiler checkpoint using development lives only.

## Gate 8 — hidden one-shot compilation

For Pythia 31M seed9:

1. Load W0 and endpoint-free evidence.
2. Generate exactly one candidate in the primary experiment.
3. Serialize and audit it.
4. Execute it and save the candidate state.
5. Seal compiler, evidence, source-plan, MGP and state hashes.
6. Resolve and download hidden WT only after the seal.
7. Evaluate W0, candidate and WT identically.

Report:

```text
actual MGP bytes
byte fraction
decode and compile time
W0 loss
candidate loss
WT loss
endpoint progress
logit KL
top-1 agreement
```

The primary success condition is endpoint progress greater than zero.

## Gate 9 — only after hidden success

Then, and only then, consider:

1. Pythia 70M hidden-size transfer;
2. deduplicated-Pile dataset transfer;
3. staged/post-training recipe transfer;
4. another decoder-only architecture;
