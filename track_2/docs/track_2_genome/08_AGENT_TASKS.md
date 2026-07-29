# GENOME agent task cards

> **Archived pre-recovery task list.** These tasks are not active assignments. In particular, do
> not start the neural auto-decoder tasks.

## How to use this file

Give an implementation agent one task card at a time. Do not ask it to “implement GENOME” as one large job. Each card has a narrow objective, required inputs, outputs, tests, and explicit exclusions.

The agent must return:

```text
summary of what changed
files changed
commands/tests run
exact results
known limitations
artifacts created with paths/hashes
recommended next task
```

It must not claim completion without running the listed tests.

---

# Task 00 — repository orientation and Track 1 adapter plan

## Objective

Locate the exact Track 1 model, tokenizer, data, checkpoint, training configuration, and evaluation entry points. Produce an implementation plan without changing training behaviour.

## Read first

- `00_README.md`
- `AGENTS.md`
- `04_TRACK1_DATA_CONTRACT.md`
- `07_IMPLEMENTATION_BLUEPRINT.md`
- `reference/track_1_research_map.md`

## Required work

1. Inspect the repository structure.
2. Identify:
   - model constructor;
   - tokenizer construction/loading;
   - final checkpoint path/config;
   - initial checkpoint or initialization seed path;
   - checkpoint loading code;
   - validation/evaluation code;
   - dataset split and record IDs;
   - training metric logs;
   - periodic checkpoints and optimizer states.
3. Determine exact parameter count, architecture dimensions, dtypes, tied weights, and special tokens from code/artifacts.
4. Propose a single Track 1 adapter module that Track 2 will call.
5. List missing artifacts and the strongest available fallback mode.

## Output

Create:

```text
docs/track_2_genome/track1_repository_map.md
```

or the closest existing documentation path.

## Do not

- edit Track 1 training code;
- infer architecture dimensions from the “50M” name;
- launch training;
- create a new evaluator yet;
- use the hidden verifier concept before the repository map exists.

## Definition of done

The document includes exact file paths/symbols and a proposed adapter API. Every uncertainty is marked and tied to a concrete inspection or fallback.

---

# Task 01 — freeze the R0 specimen

## Objective

Create a canonical, immutable Track 1 specimen that reproduces R0.

## Required inputs

- Task 00 repository map;
- final Track 1 checkpoint;
- model/tokenizer/evaluator entry points.

## Required work

1. Implement `freeze_track1_specimen.py` or equivalent.
2. Export WT to canonical safetensors.
3. Export or reconstruct W0. If impossible, declare earliest-checkpoint fallback.
4. Generate architecture and tensor inventory manifests.
5. Identify and test tied tensors.
6. Hash tokenizer, corpus split, training recipe, checkpoints, and manifests.
7. Save a specimen directory matching `04_TRACK1_DATA_CONTRACT.md`.
8. Run R0 on a small existing validation batch and save the result.

## Tests

- no missing/unexpected state keys;
- canonical state equals source state at target dtype;
- repeated model construction yields the same inventory order;
- tied tensors are equal/aliased as expected;
- final checkpoint reproduces the existing evaluator metric within declared numerical tolerance;
- W0 checksums match seed replay if seed replay is claimed.

## Do not

- convert all weights to a lower precision silently;
- remove “unimportant” tensors;
- change key names without recording a key mapping;
- overwrite source checkpoints.

## Definition of done

A stable `specimen_id`, hashes, canonical WT/base artifacts, and a passing round-trip test exist.

---

# Task 02 — build data partitions and Genome Gate skeleton

## Objective

Create immutable fit/fingerprint/probe/hidden/generation partitions and a Gate that can score R0 and obvious failures.

## Required work

1. Split by original poem/document ID, not packed sequence.
2. Create:
   - `D_genome_fit`;
   - `D_fingerprint`;
   - `D_probe`;
   - `D_verifier_hidden`;
   - `P_generation`.
3. Save hashes and overlap checks.
4. Implement Gate loading that is inaccessible from ordinary training modules except through its explicit evaluation command.
5. Score:
   - R0;
   - W0/base;
   - one deliberately corrupted R0;
   - one quantized copy.
6. Save R0 reference logits/selected hidden statistics on hidden anchors.

## Tests

- no source-document overlap;
- hidden loader not imported by compiler training package;
- R0 score is reproducible;
- corruption produces a detectable metric change;
- fixed generation seeds reproduce token IDs.

## Do not

- expose hidden record IDs in normal training configs;
- tune thresholds using generated candidates;
- call a random token split a document-level split.

## Definition of done

Genome Gate can accept a state dict or frozen MGP, produce a structured report, and distinguish R0 from a corrupted model.

---

# Task 03 — Delta-T analysis and sensitivity map

## Objective

Characterize where learning changed R0 and which tensor families are functionally sensitive.

## Required work

1. Compute Delta-T in fp32.
2. Generate per-tensor statistics and singular/spectral summaries.
3. Implement role/layer replacement diagnostics.
4. Apply calibrated noise, quantization, zeroing, and low-rank truncation.
5. Produce tables/plots of:
   - bytes by role;
   - delta energy by role/layer;
   - effective rank;
   - perturbation sensitivity;
   - functional gap versus parameter error.

## Tests

- WT reconstructed from W0 + Delta-T matches at fp32 before cast;
- tied groups remain valid;
- every diagnostic candidate records exactly which tensors were changed;
- unchanged R0 passes as a zero-perturbation control.

## Do not

- report hybrid diagnostic models as GENOME candidates;
- assume largest delta energy means largest functional importance;
- perform cross-seed averaging.

## Definition of done

A sensitivity report identifies the first tensor families and formula types to test.

---

# Task 04 — implement the MGP round trip and dense/quantized codecs

## Objective

Prove the Model Genome Program format can serialize, reload, decode, and execute a candidate deterministically.

## Required work

1. Implement manifest validation and canonical ordering.
2. Implement opcodes:
   - `BASE_COPY`/base loading;
   - `DENSE_DELTA`;
   - `QUANTIZED_DELTA`;
   - `COPY_FROM_TIED`.
3. Implement exact byte accounting.
4. Serialize JSON + safetensors.
5. Decode after destroying the in-memory source object.
6. Evaluate decoded candidates.

## Tests

- dense Delta-T recovers WT at declared output dtype;
- decode twice gives identical tensors;
- malformed/missing/shape-mismatched records fail closed;
- tied tensors are restored;
- measured file bytes match accounting fields;
- decoded dense candidate reproduces R0 metrics.

## Do not

- use pickle as canonical format;
- read WT during decode except through the encoded dense-delta sanity payload;
- estimate bytes from tensor dimensions when serialized files exist.

## Definition of done

A frozen dense MGP and at least two quantized MGPs pass the round trip and Gate.

---

# Task 05 — SVD and global rate allocator

## Objective

Establish the first serious structured rate–distortion curve.

## Required work

1. Implement per-tensor truncated SVD for matrices.
2. Implement direct/quantized storage for vectors.
3. Implement fixed-rank and energy-threshold candidates.
4. Implement a global byte allocator over singular components.
5. Serialize each candidate as MGP.
6. Evaluate all declared budgets.
7. Compare complete-WT SVD and Delta-T SVD where meaningful.

## Tests

- reconstruction matches direct factor multiplication;
- reported ranks/bytes match payload;
- candidate quality generally improves as budget increases, with anomalies recorded;
- exact R0 endpoint is approached at full rank;
- no tensor is accidentally omitted.

## Do not

- select budget points based on hidden verifier results;
- ignore factor precision;
- use a single global flatten-and-SVD as the only result.

## Definition of done

A report contains bytes, loss, KL, parameter error, decode time, and repair curves across the SVD budget frontier.

---

# Task 06 — low-rank-plus-sparse, spectral, and role dictionary baselines

## Objective

Determine whether formula combinations beat pure SVD.

## Required work

1. Add sparse residual coding with sorted/delta-coded indices.
2. Compare largest-magnitude and probe-gradient-weighted patches.
3. Add DCT or other documented spectral codec.
4. Add a role-shared block codebook baseline.
5. Test hybrids such as SVD + sparse and SVD + spectral residual.
6. Produce a common Pareto plot against Task 05.

## Tests

- patches apply to the correct flattened coordinates;
- no index overhead is omitted;
- codebook bytes are counted as shared bytes;
- role normalization is reversible;
- all decoders are deterministic.

## Definition of done

The best transparent codec for R0 is identified, including which tensor roles require special treatment.

---

# Task 07 — neural block auto-decoder, G0

## Objective

Fit a shared block interpreter and compact codes to R0/trajectory checkpoints, without yet predicting codes.

## Required work

1. Build lazy block dataset views over Delta-W.
2. Implement role-conditioned block decoder.
3. Implement global/layer/tensor codes.
4. Train first on block reconstruction.
5. Add periodic complete-model task/logit losses.
6. Quantize codes and serialize a neural MGP.
7. Count decoder, code, and patch bytes.
8. Compare against the best transparent codec.

## Mandatory controls

- same decoder capacity on randomized Delta-W;
- shuffled role labels;
- no-base-conditioning variant;
- per-tensor-code-only versus hierarchy;
- held-out block reconstruction;
- interpreter parameter count and actual file bytes.

## Do not

- call fitted codes a compiler;
- make a target-specific output head for every tensor without counting it;
- report payload-only compression as the single-model result.

## Definition of done

At least one frozen neural/hybrid MGP is Gate-evaluated, and the report states whether it improves the full cost frontier.

---

# Task 08 — latent refinement and patch allocator

## Objective

Test whether optimizing compact genome variables can repair candidates faster than child-weight optimization.

## Required work

1. Freeze the interpreter.
2. Corrupt or initialize genome codes under declared schemes.
3. Optimize only global/layer/tensor codes on `D_probe`.
4. Compare with full-weight and LoRA repair at matched information/budget.
5. Add multiscale refinement.
6. Add sparse/low-rank patch allocation under a byte penalty.
7. Plot quality versus tokens/FLOPs/wall-clock.

## Tests

- interpreter gradients remain disabled;
- hidden verifier is not used for stopping;
- trainable parameter count is correct;
- all optimizer state/compute is counted;
- repair starts from exactly the frozen candidate.

## Definition of done

The result states whether genome-space optimization is a useful accelerator even before learned compilation.

---

# Task 09 — static dataset fingerprint

## Objective

Create a non-linguistic fingerprint of the poetry task/corpus suitable for compiler conditioning.

## Required work

1. Implement deterministic token/statistical sketches.
2. Implement per-role gradient CountSketch/random projections at W0 and selected early checkpoints.
3. Add activation/loss summary channels.
4. Aggregate fixed batches with simple statistics first.
5. Record fingerprint compute, tokens, and bytes.
6. Test reproducibility and sensitivity to corpus mixtures.

## Controls

- same corpus, different batch order;
- different corpus subset;
- shuffled tokens;
- poetry/prose ratio shift;
- same token histogram with altered sequence order where practical.

## Definition of done

Fingerprints are reproducible, distinguish controlled corpus changes, and have a complete compute contract.

---

# Task 10 — early trajectory encoder and transparent latent extrapolation

## Objective

Map early Track 1 checkpoints into fitted genome space and test simple endpoint forecasts before training a large compiler.

## Required work

1. Fit/encode genome codes for selected trajectory checkpoints.
2. Plot every code coordinate/subspace over token progress.
3. Test:
   - constant/nearest code;
   - linear extrapolation;
   - quadratic extrapolation;
   - low-rank temporal subspace extrapolation;
   - per-role extrapolation.
4. Decode and Gate candidates from multiple prefix fractions.
5. Compare with Track 1 transport and ordinary continuation.

## Do not

- use WT to choose the hidden candidate at inference;
- extrapolate optimizer moments as a default;
- hide failed loss spikes.

## Definition of done

The project knows whether the fitted genome coordinates make the endpoint trajectory easier to forecast than raw weights.

---

# Task 11 — train the G1 compiler

## Objective

Predict an initial endpoint genome from architecture, dataset fingerprint, and a predeclared early R0 trajectory prefix.

## Required work

1. Freeze a successful interpreter.
2. Build compiler conditioning records.
3. Train a modest bidirectional tensor-token/graph model.
4. Compare code regression with direct decoded functional training.
5. Calibrate uncertainty or a small mixture if endpoints/codes are multimodal.
6. Evaluate pure compile, compile + latent refinement, and compile + patch.
7. Compare from identical prefix information with ordinary continuation, Track 1 transport, and transparent latent extrapolation.

## Required labels

- `G1-Pure` if no final reference outputs are used at inference;
- `G1-Distill` if final reference outputs are supplied on allowed probe examples.

## Definition of done

A frozen compiler and MGP candidates have complete child/meta cost accounting and one hidden Gate result.

---

# Task 12 — sibling model-life generator

## Objective

Create controlled independent runs for G2 without turning the project into an enormous sweep.

## Required work

1. Define micro/small/track-scale organism configs.
2. Vary one factor at a time initially.
3. Save enough checkpoints for trajectory examples.
4. Capture complete model-life manifests.
5. Create run-level train/development/hidden split before compiler training.
6. Verify hidden endpoints are inaccessible to training loaders.

## Definition of done

The model-life corpus contains multiple independent endpoints and a permanently withheld run.

---

# Task 13 — G2 hidden-run compiler

## Objective

Test whether GENOME transfers to an unseen seed or data order.

## Required work

1. Train interpreter/compiler only on allowed runs.
2. Add symmetry-aware or function-space handling as needed.
3. Compile the hidden run from its allowed information.
4. Select candidates with its probe set only.
5. Freeze all candidates and run Genome Gate.
6. Compare with nearest-neighbour endpoint, mean genome, ordinary training, and Track 1 acceleration.

## Definition of done

The result is classified as pass/fail under the frozen G2 gate, with no target endpoint leakage.

---

# Task 14 — weight language model branch

## Objective

Learn a token language over model-weight blocks for representation, completion, and generation.

## Required work

1. Learn a block codebook on training runs only.
2. Tokenize architecture/tensor/block structure.
3. Train masked-block and checkpoint-order objectives.
4. Test missing tensor/layer completion.
5. Test endpoint-token prediction.
6. Compare discrete genome size and function with continuous codes.

## Definition of done

The weight LM beats shuffled/random controls and contributes to either representation or endpoint compilation.

---

# Task 15 — genome-native child model

## Objective

Train a child architecture whose weights are generated from compact codes by construction.

## Required work

1. Keep R0 as immutable baseline.
2. Define generated versus directly stored tensor families.
3. Train child genome codes/interpreter end-to-end from scratch.
4. Match parameter/FLOP scale as fairly as possible.
5. Measure endpoint quality, genome size, training cost, and predictability.
6. Test whether a compiler can infer final child codes more easily than R0 codes.

## Definition of done

A separate architecture result reports whether co-design improves the genome premise.

---

# Task 16 — self-hosting prototype

## Objective

Create the smallest scientifically valid parent/child compiler loop described in `09_RSI_AND_FUTURE_RESEARCH.md`.

## Preconditions

- a compiler can generate multiple child models;
- hidden child-model tasks exist;
- the compiler itself is representable by the genome system;
- evaluator and budget are immutable.

## Required work

1. Parent emits candidate child-compiler genomes or patches.
2. Decode candidates.
3. Evaluate each child on hidden model-compilation tasks.
4. Accept only Pareto improvements.
5. Archive complete lineage and costs.
6. Attempt one further generation using the accepted child.

## Definition of done

A child compiler generated by the parent is evaluated under the same hidden protocol. Improvement is reported only if it survives hidden tasks and total-cost accounting.

---

# Copy-paste handoff template for coding agents

```text
You are implementing one narrowly scoped part of Track 2 GENOME.

Read:
- AGENTS.md
- <relevant docs>

Implement Task <number and title> from 08_AGENT_TASKS.md.

Rules:
- Do not broaden scope into later phases.
- Do not modify immutable Track 1 behaviour.
- Do not use hidden verifier data except through the final Gate command.
- Preserve exact tensor names, shapes, dtypes, ties, and hashes.
- Count all serialized bytes and compute.
- Run every Definition of Done test.
- Record failures rather than hiding them.

Return:
1. Summary.
2. Files changed.
3. Commands/tests run and exact outcomes.
4. Artifacts and hashes.
5. Known limitations.
6. Whether the task's Definition of Done passed.
7. Recommended next task only; do not implement it yet.
```
