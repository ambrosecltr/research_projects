# RunPod handoff

## Branch

Use only:

```text
agent/genome-clean-start
```

Record the exact commit after pulling. Use this branch only.

## Volume

Create a new 100 GB network volume. Suggested name:

```text
genome-pythia-v1
```

Do not copy any pre-existing GENOME workspace into the new volume.

## CPU setup pod

Use a CPU pod first for source resolution and storage inspection.

```bash
cd /workspace
git clone --branch agent/genome-clean-start \
  https://github.com/ambrosecltr/research_projects.git genome_v1/repo
cd /workspace/genome_v1/repo/track_2
python -m pip install -e '.[dev,evaluation]'
genome init-workspace --root /workspace/genome_v1
python -m compileall -q genome tests
python -m pytest -q
```

Write and resolve the plan:

```bash
genome write-source-plan \
  --output /workspace/genome_v1/control/pythia_v1.requested.json

genome resolve-source-plan \
  --plan /workspace/genome_v1/control/pythia_v1.requested.json \
  --output /workspace/genome_v1/control/pythia_v1.pinned.json
```

Inspect the pinned plan and source metadata before downloading. Hidden 31M seed9 WT must still have no resolved commit.

Materialize approved W0/WT snapshots:

```bash
genome materialize-sources \
  --plan /workspace/genome_v1/control/pythia_v1.pinned.json \
  --workspace /workspace/genome_v1
```

## Required setup outputs

Before switching to GPU, produce:

```text
control/environment.json
control/git.json
control/pythia_v1.pinned.json
source/receipts/materialization.json
control/storage_actual.json
control/source_audit.md
```

Mark `source/hf` read-only after receipts are verified.

## GPU phase

Attach the same new network volume to a GPU pod. Start with a single A40, L40S, RTX 4090, or comparable GPU; Pythia 14M/31M do not justify a premium multi-GPU pod for setup.

The GPU agent must follow `docs/EXPERIMENT_PLAN.md` in order. It must not launch compiler training before both development sizes have accepted compact target programs.

## Production compiler configuration

Use `configs/compiler/pythia_v1.yaml` as the starting point. Benchmark memory and throughput before increasing model width or layer count.

## Hidden reveal

After one-shot hidden compilation:

```bash
genome seal-hidden ...
genome reveal-hidden ...
```

The reveal command rejects a seal that does not match the hidden run and source plan.

## Handoff result

The next agent is done when either:

- a legitimate compiler run is active with accepted compact targets and resumable checkpoints; or
- a pretraining gate fails and the failure is documented without launching invalid training.
