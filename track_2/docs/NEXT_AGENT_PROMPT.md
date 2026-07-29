# Prompt for the RunPod implementation agent

You are taking over GENOME Track 2 from a clean forward-only branch.

Repository:

```text
https://github.com/ambrosecltr/research_projects
```

Active branch:

```text
track_2/pythia-seed9-training
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

Gates 0 to 3 are complete. The fixed compact-target formula passed both
development lives. Gate 4 is applying that formula to every training life.

Continue from the exact state in `docs/RUNPOD_HANDOFF.md`. Do not create a new
volume and do not repeat completed source or evidence work.

Use the existing 100 GB RunPod volume `genome-pythia-v1`, volume ID
`4kwmhcepgj`.

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

1. finish the fixed-formula targets for all training lives;
2. stop if any required target fails its declared gate;
3. construct the 19-record compiler corpus;
4. run a tiny complete compiler smoke and resume test;
5. launch the production compiler run only after every gate passes;
6. freeze the selected compiler using development lives only;
7. compile exactly one hidden 31M seed9 candidate;
8. seal compiler, evidence, program, and Runtime hashes;
9. reveal hidden WT only after sealing;
10. report one-shot results before any repair.

Non-negotiable:

- one learned GENOME Compiler only;
- deterministic Runtime;
- no learned decoder;
- no dense or exact residual;
- no one-value-per-weight representation;
- no hidden WT or endpoint-derived data before sealing;
- no extra model family or size before hidden 31M success;
- stop rather than bypass a failed compact or functional gate.

Commit code, configuration, and documents to the active branch. Save exact
RunPod volume and pod IDs, environment, commands, hashes, storage, target
results, compiler metrics, and resume command. Commit only. Do not push.
