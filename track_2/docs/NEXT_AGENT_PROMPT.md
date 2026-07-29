# Prompt for the RunPod implementation agent

You are taking over GENOME Track 2 from a clean forward-only branch.

Repository:

```text
https://github.com/ambrosecltr/research_projects
```

Branch:

```text
agent/genome-clean-start
```

Read, in order:

```text
track_2/AGENTS.md
track_2/README.md
track_2/docs/THEORY_AND_MATH.md
track_2/docs/ARCHITECTURE.md
track_2/docs/EXPERIMENT_PLAN.md
track_2/docs/SOURCE_MATRIX.md
track_2/docs/RUNPOD_HANDOFF.md
track_2/docs/IMPLEMENTATION_STATUS.md
```

Treat the checked-out Track 2 tree as the complete authoritative project.

Your job is to take the current implementation through the real Pythia source, compact-target and GPU-training gates.

Create a new 100 GB RunPod network volume named `genome-pythia-v1` or a close date-suffixed equivalent. Do not mount or copy any pre-existing GENOME volume.

Use only:

```text
Pythia 14M seeds0–7 and seed9 training
Pythia 14M seed8 development
Pythia 31M seeds0–7 training
Pythia 31M seed8 development
Pythia 31M seed9 fresh hidden
```

Pythia 14M seed9 is a training life. Materialize its W0 and WT, fit its compact
target, and include its endpoint-free input and accepted fitted target in compiler
training. Never describe it as a hidden evaluation.

Follow the gates in `docs/EXPERIMENT_PLAN.md` exactly:

1. initialize the new workspace;
2. resolve and pin sources without resolving hidden WT;
3. inspect actual storage before downloading;
4. materialize W0/WT pairs and receipts;
5. build canonical lives and verify native/canonical round trips;
6. build real corpus and W0-response evidence;
7. fit and functionally refine compact targets at the declared byte budgets;
8. require both development lives to pass the 10% byte and 80% endpoint-progress gates;
9. construct the compiler corpus;
10. run a tiny complete compiler smoke and resume test;
11. launch the production compiler run only after every gate passes;
12. freeze the selected compiler using development lives only;
13. compile exactly one hidden 31M seed9 candidate;
14. seal compiler/evidence/program/runtime hashes;
15. reveal hidden WT only after sealing;
16. report one-shot results before any repair.

Non-negotiable:

- one learned GENOME Compiler only;
- deterministic Runtime;
- no learned decoder;
- no dense or exact residual;
- no one-value-per-weight representation;
- no hidden WT or endpoint-derived data before sealing;
- no extra model family or size before hidden 31M success;
- stop rather than bypass a failed compact or functional gate.

Commit code/config/docs to a new branch from `agent/genome-clean-start`. Save exact RunPod volume/pod IDs, environment, commands, hashes, storage, target frontier, compiler metrics and resume command. Do not merge your own branch.
