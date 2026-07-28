# Connecting GENOME to `poetry_50m / track_1`

Track 2 now contains a concrete adapter for the exact Track 1 implementation in this repository. Do **not** copy or invent a second model constructor. GENOME imports Track 1's own:

- `DecoderOnlyTransformer` and `ModelConfig`;
- `seed_everything` initialization procedure;
- trainer-checkpoint and weights-only snapshot schemas;
- prepared tokenizer;
- `train`, `validation`, and `test` packed records;
- causal-LM loss and `loss_mask` semantics.

The adapter also knows the declared production shapes:

- GPT R0: **50,343,424** trainable parameters;
- nGPT branch: **54,596,096** parameters.

## 1. Place Track 2 beside Track 1

Expected repository layout:

```text
research_projects/
└── poetry_50m/
    ├── track_1/
    └── track_2/
```

Install both projects into the same virtual environment:

```bash
cd poetry_50m/track_1
uv sync --extra dev
source .venv/bin/activate

cd ../track_2
python -m pip install -e '.[dev,analysis]'
```

## 2. Create the local configuration

```bash
cd poetry_50m/track_2
cp configs/poetry50m_track1.example.yaml configs/poetry50m_track1.yaml
```

The example uses paths relative to the YAML file. Review the actual Track 1 artifact locations, especially:

```yaml
adapter:
  project_root: ../../track_1
  factory: genome.adapters.poetry50m:create_adapter
  kwargs:
    prepared_dir: ../../track_1/artifacts/prepared
    run_dir: ../../track_1/runs/r0
    initial_snapshot: ../../track_1/runs/r0/trajectory/initial.pt
    final_snapshot: ../../track_1/runs/r0/trajectory/final.pt
    run_manifest: ../../track_1/runs/r0/run.manifest.json
    train_receipt: ../../track_1/runs/r0/train.receipt.json

specimen:
  final_checkpoint: ../../track_1/runs/r0/checkpoints/final.pt
  base_checkpoint: ../../track_1/runs/r0/trajectory/initial.pt
```

If the exact initial snapshot was not retained, set `base_checkpoint: null`. The adapter will replay W0 with Track 1's own seed and constructor, and specimen freezing will construct it twice and require identical hashes.

## 3. Preflight while R0 is still training

The code can be installed and tested before R0 finishes. Run:

```bash
genome track1-preflight \
  --config configs/poetry50m_track1.yaml \
  --output artifacts/preflight.json
```

The report checks:

- Track 1 import and source paths;
- confirmation that `poetry50m` was imported from the configured checkout;
- exact production parameter count;
- two independent W0 reconstructions and their hashes;
- prepared metadata, tokenizer, validation packs, and test packs;
- initial trajectory snapshot;
- final trajectory snapshot, run manifest, and train receipt;
- latest available R0 checkpoint;
- `global_step`, `max_steps`, and completion fraction;
- W0/WT model-config, training-config, step, run-ID, and receipt hashes;
- whether all conditions required to freeze R0 are satisfied.

While the full run is active, `ready_to_freeze: false` is expected. GENOME deliberately rejects a partial endpoint when `require_complete_endpoint: true`.

## 4. Freeze only the completed endpoint

After Track 1 has written the full endpoint:

```bash
genome track1-preflight --config configs/poetry50m_track1.yaml

genome freeze --config configs/poetry50m_track1.yaml

genome verify \
  --specimen artifacts/specimens/track1_R0 \
  --config configs/poetry50m_track1.yaml
```

`freeze` accepts either:

- a Track 1 trainer checkpoint (`format_version: 2`, model under `model`); or
- a Track 1 weights-only trajectory snapshot (`poetry50m.weights.v1`, model under `state_dict`).

For the real R0, the endpoint validator requires:

```text
global_step >= max_steps
```

It also requires the trainer checkpoint and `trajectory/final.pt` to contain the
same model state and step, and reconciles both files plus `run.manifest.json`
against `train.receipt.json`. A complete-looking checkpoint with broken lineage
or mismatched hashes is rejected. The completion rule can be disabled only for
an explicitly marked mechanical fixture; never disable it for the research R0.

## 5. What verification must show

Review `artifacts/specimens/track1_R0/verification.json` and require:

- specimen hashes pass;
- canonical WT reproduces the Track 1 packed validation loss;
- W0 is materially worse than WT;
- tensor count, names, shapes, dtypes, and tied groups are exact;
- W0 reconstruction hashes agree;
- endpoint validation records a complete run and matching final artifacts;
- the model has 50,343,424 parameters for the GPT branch.

GENOME's adapter evaluates one packed record at a time using the same
`input_ids = pack.input_ids[:-1]`, `targets = pack.input_ids[1:]`, and
`loss_mask = pack.loss_mask[1:]` path as Track 1's `eval-loss` command. Only a
small deterministic set of anchor logits is retained; full `[B,T,V]` outputs
are not accumulated across the evaluation set.

## 6. First R0 experiments

```bash
genome analyze \
  --specimen artifacts/specimens/track1_R0 \
  --output artifacts/analysis/track1_R0

genome rate-distortion \
  --specimen artifacts/specimens/track1_R0 \
  --config configs/poetry50m_track1.yaml \
  --output artifacts/rate_distortion/track1_R0 \
  --ranks 0,1,2,4,8,16,32,64

genome report artifacts/rate_distortion/track1_R0 \
  --output artifacts/reports/track1_R0_g0
```

The sweep factorizes each unique two-dimensional Delta-T tensor once and reuses
that exact SVD workspace across all SVD and SVD-plus-sparse rank candidates. Its
shared factorization time is recorded in `rate_distortion_context.json` and is
charged once across the frontier rather than once per rank. Candidate fitting
times exclude that shared cost. The full output directory is published
atomically and is never overwritten; an interrupted sweep leaves no apparently
complete frontier.

The first decision is not whether scalar reconstruction error is low. It is
whether a compact genome preserves validation loss, logits, and poetry
behaviour, or reaches them after substantially less repair compute.

## 7. Neural genome only after the transparent frontier

```bash
genome fit-neural \
  --specimen artifacts/specimens/track1_R0 \
  --output artifacts/genomes/track1_R0_neural_v0.mgp \
  --interpreter-output artifacts/interpreters/block_v0 \
  --updates 20000 \
  --batch-size 256 \
  --block-size 16 \
  --hidden-dim 512 \
  --device cuda

genome evaluate \
  --specimen artifacts/specimens/track1_R0 \
  --mgp artifacts/genomes/track1_R0_neural_v0.mgp \
  --interpreter artifacts/interpreters/block_v0 \
  --config configs/poetry50m_track1.yaml
```

The update count is an execution starting point, not a fixed scientific constant. The R0 rate–distortion and role-sensitivity measurements should choose the next architecture.

## 8. Run Track 1's unchanged poetry suite on a phenotype

GENOME can wrap any decoded MGP in an evaluation-only copy of R0's full trainer
checkpoint:

```bash
genome export-track1-checkpoint \
  --specimen artifacts/specimens/track1_R0 \
  --mgp artifacts/genomes/track1_R0_int4.mgp \
  --config configs/poetry50m_track1.yaml \
  --output artifacts/track1_eval_checkpoints/track1_R0_int4.pt
```

Then, from `poetry_50m/track_1`, use the established generation and metrics
commands without reimplementing their prompt or sampling contract:

```bash
uv run poetry50m generate \
  --prepared artifacts/prepared \
  --model-config configs/model/track1_50m.yaml \
  --train-config configs/training/baseline.yaml \
  --run-dir runs/r0 --batch-size 4 \
  --checkpoint ../track_2/artifacts/track1_eval_checkpoints/track1_R0_int4.pt \
  --suite configs/evaluation/prompt_suite.json \
  --evaluation-manifest-id track2-int4-v1 \
  --output ../track_2/artifacts/generations/track1_R0_int4.jsonl

uv run poetry50m metrics \
  --prepared artifacts/prepared \
  --suite configs/evaluation/prompt_suite.json \
  --records ../track_2/artifacts/generations/track1_R0_int4.jsonl \
  --manifest ../track_2/artifacts/generations/track1_R0_int4.manifest.json \
  --output ../track_2/artifacts/generations/track1_R0_int4.metrics.json
```

The exported file retains R0's loader contracts but adds
`genome_evaluation.evaluation_only: true` and `resume_forbidden: true`. It is
for `generate`, `eval-loss`, and reporting only. Never pass it to `train
--resume`.

## 9. Moving from G0 to G1/G2

G1/G2 needs multiple independent model lives or a strictly withheld target. For every run retain:

- W0 and WT;
- exact architecture, tokenizer, corpus, and data-order identities;
- seed and initialization identity;
- allowed trajectory prefix;
- gradient/dataset fingerprint;
- fitted G0 genome code;
- endpoint-hidden status;
- complete compile, candidate-selection, decode, probe, and repair compute.

Checkpoints from one R0 trajectory are correlated observations, not independent model lives. A compiler trained on R0's endpoint must not be described as a transferable accelerator until a sealed R1/R2 gate passes.
