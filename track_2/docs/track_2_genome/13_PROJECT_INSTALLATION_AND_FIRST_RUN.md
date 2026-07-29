# Installing GENOME into the Track 1 project

> **Archived pre-recovery installation note.** Track 1 is no longer the first compiler-training
> target. Use `../../README.md` and the active recovery gates.

## 1. Recommended placement

Place this complete documentation directory inside the existing Track 1 repository:

```text
<project-root>/
  docs/
    track_2_genome/
      00_README.md
      AGENTS.md
      ...
```

The documents use paths relative to the GENOME documentation directory. Keep the files together.

If the repository already has a root `AGENTS.md`, do not overwrite it. Add a short pointer to the existing root instructions instead:

```markdown
## Track 2: GENOME

For any work under `track_2_genome/` or involving model-genome research, read
`docs/track_2_genome/AGENTS.md` and follow the ordered task cards in
`docs/track_2_genome/08_AGENT_TASKS.md`.
```

If the repository has no root agent-instruction file, either copy the pointer above into a new root `AGENTS.md` or explicitly name the GENOME documents in every agent handoff.

---

## 2. Preserve Track 1 before starting

Before an agent changes code:

1. Commit or otherwise snapshot the current Track 1 repository.
2. Record the branch and commit containing the fully trained Track 1 model.
3. Confirm the final checkpoint is backed up outside any directory that experiments may clean.
4. Confirm the tokenizer, configuration, metrics, and corpus manifests are retained.
5. Do not rename or move Track 1 artifacts until Task 00 has mapped them.

GENOME treats the final Track 1 endpoint as an immutable specimen named **R0**. The initial state is **W0**, the endpoint is **WT**, and the learned displacement is **Delta-T = WT - W0**.

---

## 3. First agent assignment

The first implementation agent receives **Task 00 only**. It must orient itself and produce a repository map. It must not begin codecs, model training, or architecture changes.

Copy-paste handoff:

```text
We are starting Track 2, called GENOME — Generative Endpoint Neural Operator for Model Emission.

Read these files completely before changing anything:
- docs/track_2_genome/00_README.md
- docs/track_2_genome/AGENTS.md
- docs/track_2_genome/04_TRACK1_DATA_CONTRACT.md
- docs/track_2_genome/07_IMPLEMENTATION_BLUEPRINT.md
- docs/track_2_genome/08_AGENT_TASKS.md
- docs/track_2_genome/reference/track_1_research_map.md

Implement Task 00 from 08_AGENT_TASKS.md and no later task.

Your job is to locate and document the exact Track 1 model constructor, tokenizer, final checkpoint, W0 or initialization path, configs, data manifests, evaluator, checkpoint history, metrics, tied weights, architecture dimensions, and reproducibility constraints. Propose one Track 1 adapter API. Do not edit Track 1 training behaviour and do not launch training.

Return all outputs required by the task card, including exact paths, unresolved questions, and the strongest fallback for any missing artifact.
```

---

## 4. Review Task 00 before Task 01

Check the repository map for:

- exact symbols and paths rather than guesses;
- real architecture dimensions, not values inferred from “50M”;
- final checkpoint and tokenizer location;
- whether W0 is stored or exactly reproducible;
- tied embeddings or other aliases;
- the original evaluation entry point;
- document/poem identity needed for split isolation;
- checkpoint and metric history;
- any missing artifact that changes the strength of the claim.

Only then assign Task 01, which freezes R0 into a canonical specimen.

---

## 5. First executable milestone

The first code milestone is not a neural genome. It is this verified vertical slice:

```text
source Track 1 checkpoint
    -> canonical R0 specimen
    -> W0/WT tensor inventory
    -> Delta-T
    -> dense MGP serialization
    -> destroy in-memory source representation
    -> reload MGP
    -> deterministic decode
    -> execute candidate
    -> reproduce R0 metrics
```

Until that round trip works, do not build SVD allocators, neural interpreters, compilers, or RSI machinery.

---

## 6. Initial task order

Use this order unless a documented dependency prevents it:

```text
Task 00  repository orientation
Task 01  freeze R0
Task 02  partitions and Genome Gate
Task 03  Delta-T/sensitivity analysis
Task 04  MGP round trip
Task 05  SVD rate–distortion
Task 06  structured hybrid codecs
Task 07  neural auto-decoder
Task 08  latent refinement
Task 09  dataset fingerprint
Task 10  trajectory-to-genome extrapolation
Task 11  G1 compiler
Task 12  sibling model lives
Task 13  G2 hidden-run compiler
```

Tasks 14–16 are research branches that require evidence from the earlier sequence.

---

## 7. Suggested branch and artifact policy

A practical branch pattern is:

```text
track2/task-00-repository-map
track2/task-01-freeze-r0
track2/task-02-genome-gate
...
```

Generated model artifacts should normally remain outside Git unless the repository already has a suitable large-file policy. Commit manifests, configurations, code, small evaluation reports, and hashes. Store checkpoints/MGP payloads in the project artifact area with stable IDs and backups.

Never place the only copy of R0 in a disposable experiment output directory.

---

## 8. Earliest useful scientific result

The earliest meaningful report is the **G0 R0 rate–distortion audit**:

- raw WT and Delta-T storage baselines;
- standard compressed/quantized baselines;
- per-tensor SVD;
- globally allocated SVD;
- low-rank plus sparse and spectral alternatives;
- actual serialized bytes;
- hidden validation/function metrics;
- decode time;
- repair-to-quality curves.

This answers whether R0 has a compact useful conditional description. It does not yet claim endpoint prediction.
