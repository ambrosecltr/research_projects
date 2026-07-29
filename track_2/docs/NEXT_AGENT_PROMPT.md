# Fresh-agent handoff prompt

You are continuing GENOME Track 2 in:

```text
/Users/ambrosecoulter/research_projects/track_2
```

Branch:

```text
track_2/pythia-seed9-training
```

Read these files before you change or run anything:

```text
track_2/AGENTS.md
track_2/README.md
track_2/docs/EXPERIMENT_PLAN.md
track_2/docs/RUNPOD_HANDOFF.md
track_2/VALIDATION.md
```

Keep the user updated in simple language. Explain what you are doing and why.

## Current resource state

Both RunPod pods are stopped. The network volume is intact.

```text
Network volume ID: 4kwmhcepgj
Network volume:    genome-pythia-v1
Region:            EU-RO-1
Size:              100 GB
Mount:             /workspace

CPU pod ID:        ufz8yhwxxmzufs
CPU state:         stopped

GPU pod ID:        g3ou14zh0wb7q7
GPU:               RTX 4090
GPU state:         stopped
```

Do not start the GPU first. Start the CPU pod only when you need to update or test
the volume checkout. Stop paid pods when the task is complete.

Do not delete or replace the network volume.

## Correct split

```text
Training and formula development:
  Pythia 14M seeds0–6,8,9
  Pythia 31M seeds0–6,8

Fresh development:
  Pythia 14M seed7
  Pythia 31M seed7

Hidden:
  Pythia 31M seed9
```

Seed8 affected formula design. Its historical results are training and
formula-development evidence. Never describe them as independent development
confirmation.

Pythia 14M seed9 is a training life. Its W0 and WT are available for target
preparation. It is never a hidden evaluation.

Pythia 31M seed9 is the only hidden life. Its WT is absent and must stay unresolved
until the compiler program and Runtime result are sealed.

## Work already complete

- The paid GPU was stopped after volume checks.
- The data split was corrected in code, config, tests, and documents.
- `DIRECT_VECTOR` is limited to one-dimensional fp16 tensors with at most 4,096
  values. Direct matrices are forbidden. Aggregate direct-vector payload bytes
  are reported.
- One formula-driven production command exists: `genome produce-target`.
- The immutable formula ID is:

```text
4f4e6d9d5d9ef7677dd955bb89be81dfedf161ecb010fdfd405475fdce46d155
```

- Every new evaluation and acceptance report binds run ID, formula ID, program ID,
  manifest SHA, payload SHA, W0 state ID, WT state ID, evaluation JSONL SHA,
  source-plan ID, and full code commit.
- The compiler-corpus builder recomputes and checks those bindings.
- Existing even records are refinement data.
- Existing odd records are formula-tuning data.
- A second immutable development verifier exists on the volume:

```text
/workspace/genome_v1/evidence/corpus/verifier/tokens.jsonl
/workspace/genome_v1/evidence/corpus/verifier/receipt.json
```

It uses another shard, has 128 batches, and has token SHA:

```text
6a9cfd8231943cc0603a7c40c9f6f0bfa02c7032e415ff452258c359a4e8cd99
```

- Compiler checkpoint selection now uses free-running generated MGP quality.
  Teacher-forced loss is diagnostic only.
- Hidden result tiers are predeclared:
  below 25% is weak signal only and 80% or more is strong.
- A small non-weight audit bundle is committed at:

```text
track_2/artifacts/audit/pythia_v1_formula_v2_historical
```

- Local result at handoff: 50 tests pass.
- No compiler corpus exists.
- No compiler training has started.
- No seed7 development target has run.
- No hidden result exists.

## Historical result warning

Old formula-v2 reports are audit evidence only. They predate the new artifact
bindings and cannot enter the compiler corpus.

Pythia 14M seed5 historically reached 73.38% progress and failed the 80% gate.
It must be rerun once with the formally frozen formula. Keep the rejection and all
diagnostics. Do not lower the gate.

The expected final compiler corpus is:

```text
16 accepted training targets
2 accepted seed7 development targets
18 total records
```

Pythia 14M seed5 is rejection evidence and is excluded from compiler supervision.

## Exact next actions

1. Confirm the branch is clean and contains the protocol-repair commit.
2. Confirm that commit was pushed.
3. Start the CPU pod.
4. Update `/workspace/genome_v1/repo` to the same branch commit.
5. Run all Track 2 tests on the CPU pod.
6. Check again that `/workspace/genome_v1/source/hf/pythia-31m-seed9/wt` does not
   exist.
7. Stop the CPU pod if you are not immediately using it.
8. Change only the formula status in
   `configs/targets/pythia_v1.yaml` from `formula-development` to `frozen`.
   Do not change any identity field or the formula ID.
9. Commit and push the formula-status change.
10. Update and retest the volume checkout.
11. Start the GPU.
12. Rerun Pythia 14M seed5 exactly once with `genome produce-target`.
13. Save the bound rejection and full diagnostics.
14. Regenerate every other training target with the exact same formula and command.
15. Do not run seed7 until all required training reports are present and bound to
    the same formula and source plan.
16. Run Pythia 14M seed7 exactly once on the independent 128-batch verifier.
17. Run Pythia 31M seed7 exactly once on the same verifier.
18. Do not change the formula after either seed7 result.
19. Build the 18-record compiler corpus and let the builder reverify all bindings.
20. Run compiler smoke and resume checks.
21. Start compiler training only after every earlier gate passes.

## Fixed prohibitions

Do not:

- change the 10% byte budget;
- lower the 80% development gate;
- add a residual;
- add a direct matrix;
- use seed8 as development confirmation;
- use the odd formula-tuning records as fresh development data;
- build the compiler corpus before bound seed7 results exist;
- start compiler training early;
- select a checkpoint using teacher-forced loss alone;
- resolve, download, or inspect Pythia 31M seed9 WT before sealing;
- push hidden repair as part of the one-shot result.

When you reach any failed gate, stop the dependent work and report the failure. A
failed gate is a valid result.
