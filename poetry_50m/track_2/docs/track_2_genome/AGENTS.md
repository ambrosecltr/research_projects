# AGENTS.md — mandatory operating instructions for GENOME

This file is authoritative for agents implementing Track 2. Read the project documents before changing code. The user is intentionally pursuing a speculative idea. Do not reject a branch merely because it is unconventional. Convert uncertainty into a measurable experiment.

## Mission

Build and test **GENOME**, a system that represents and eventually compiles a trained neural model into a compact executable model genome. The first phenotype is the fully trained Track 1 poetry model, **R0**.

The immediate job is not recursive self-improvement. The immediate job is to answer, in order:

1. Can R0's learned displacement be represented compactly?
2. Can a shared interpreter decode that representation without hiding the checkpoint in its own parameters?
3. Can the genome be inferred from limited early-training or dataset evidence?
4. Does that inference transfer to a withheld run?

## Read in this order

1. `00_README.md`
2. `01_THEORY_AND_MATH.md`
3. `02_MODEL_GENOME_FORMAT.md`
4. `03_SYSTEM_ARCHITECTURE.md`
5. `04_TRACK1_DATA_CONTRACT.md`
6. `05_EXPERIMENT_PLAN.md`
7. `06_EVALUATION_PROTOCOL.md`
8. `07_IMPLEMENTATION_BLUEPRINT.md`
9. `08_AGENT_TASKS.md`

Use `10_REFERENCES.md` when selecting an implementation technique. Use `11_DECISION_LOG_TEMPLATE.md` for every experiment.

## Hard constraints

1. **Do not modify Track 1 behaviour while establishing the baseline.** Treat R0, its tokenizer, architecture, corpus split, and evaluation protocol as immutable inputs.
2. **Never train on the hidden verifier.** A probe set may be used for latent refinement; the verifier may not.
3. **Never call raw weight MSE alone success.** Always execute the generated model and measure task/function metrics.
4. **Never compare unaligned independent checkpoints as though neuron/head indices have universal meaning.** Same-run coordinates are aligned. Cross-run coordinates are not assumed aligned.
5. **Never omit decoder size or repair compute from the result.** Report single-target and amortized accounting separately.
6. **Never silently materialize WT inside a cache, constants file, code-generated array, test fixture, or initialization artifact.** That is target leakage.
7. **Never overwrite an experiment.** Use immutable run IDs and save config, code commit, environment, metrics, and artifact hashes.
8. **Never jump directly to a large diffusion model.** Implement transparent codecs and a simple auto-decoder first.
9. **Never require exact bitwise recovery before testing functional recovery.** Exact recovery is a separate diagnostic.
10. **Never describe G0/G1 results as general RSI or a universal training law.** Use the level names in `00_README.md`.

## Preferred implementation behaviour

- Make one small vertical slice runnable before creating abstractions for future architectures.
- Add assertions for every tensor name, shape, dtype, tie, and checksum.
- Use deterministic seeds and deterministic data selection wherever practical.
- Save JSONL/Parquet metrics as well as human-readable Markdown summaries.
- Keep the interpreter capable of decoding one tensor or one block at a time; do not require all generated weights to live in GPU memory simultaneously.
- Separate the **representation test** from the **prediction test**. A learned latent fitted directly to R0 is G0, not endpoint prediction.
- Use direct latent optimization before training a learned linker. It is cheaper and establishes whether the latent space is navigable.
- Fit the learned displacement `WT - W0`, not only the complete checkpoint.
- Keep embeddings and the LM head as a separately reported tensor family; they may dominate bytes and behave differently from transformer blocks.
- Preserve tied embeddings exactly when Track 1 ties them.

## Required output from every experiment

Every run must save at least:

```text
run_id
research_level
hypothesis
parent_run_id
code_commit
config_hash
environment_hash
R0_manifest_hash
training/probe/verifier split hashes
representation family
genome bytes
shared interpreter bytes
patch bytes
raw baseline bytes
decode seconds
compile/fitting seconds
verification seconds
repair steps/tokens/FLOPs/seconds
validation loss and perplexity
anchor-logit divergence
terminal-noise-normalized quality gap
parameter errors by tensor family
failure reason or acceptance decision
```

## What to do when something fails

Do not replace the method immediately with a much larger network. First localize the failure:

- reconstruct each tensor family separately;
- replace one generated tensor family at a time in R0;
- measure sensitivity to each family;
- compare complete weights versus Delta-T;
- compare raw, aligned, spectral, low-rank, and function-space errors;
- test whether a small sparse patch repairs the model;
- test whether latent refinement repairs it faster than full-weight training;
- inspect whether embeddings, normalization, attention, or MLP tensors dominate the gap.

Record the negative result before changing direction.

## Scope control

The first implementation does **not** need:

- a universal architecture grammar;
- large architecture sweeps;
- broad public checkpoint ingestion;
- exact cross-hardware deterministic decoding;
- a symbolic equation for every scalar;
- a self-improving compiler;
- publication-grade ablations.

It does need a trustworthy R0 specimen, a correct evaluator, meaningful codecs, exact byte accounting, and one reproducible rate–distortion curve.
