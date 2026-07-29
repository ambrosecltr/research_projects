# GENOME Pythia v1 experiment plan

## Gate 0 — workspace and hidden boundary

Use the existing network volume at `/workspace/genome_v1`.

Pass conditions:

- source, evidence, programs, compiler files, and logs are separate;
- Pythia 31M seed9 WT is absent and unresolved;
- source commits and file hashes are recorded;
- paid compute is stopped when no job is active.

## Gate 1 — whole-life split

Use `configs/sources/pythia_v1.json`.

- Training: 14M seeds0–6,8,9 and 31M seeds0–6,8.
- Development: 14M seed7 and 31M seed7.
- Hidden: 31M seed9.

Seed8 is training and formula-development evidence. Do not call its historical
results development confirmation. Pythia 14M seed9 is a training life.

## Gate 2 — canonical lives and semantic evidence

For each available life:

1. verify native-to-canonical state round trips;
2. export the architecture graph and tensor inventory;
3. record W0 response evidence and corpus evidence;
4. bind the recipe and source-plan IDs.

GENOME targets a functionally good endpoint from the same pinned corpus and recipe.
It does not promise the exact raw WT coordinates or exact training trajectory.

## Gate 3 — three separate data uses

The first immutable sample is already pinned.

- Existing even records: functional refinement.
- Existing odd records: formula tuning.
- Second immutable sample: fresh development verification.

The second sample must use a different shard or non-overlapping byte range, have a
matching receipt and SHA-256, and provide at least 128 evaluation batches.

The created verifier uses shard
`document-00001-of-00020.bin`, has 128 batches, and has token JSONL SHA
`6a9cfd8231943cc0603a7c40c9f6f0bfa02c7032e415ff452258c359a4e8cd99`.

## Gate 4 — freeze one target formula

The formula is `configs/targets/pythia_v1.yaml`. Its immutable ID is:

```text
4f4e6d9d5d9ef7677dd955bb89be81dfedf161ecb010fdfd405475fdce46d155
```

Fixed production rules:

- serialized target fraction is at most 10%;
- endpoint progress gate is at least 80%;
- the candidate beats W0 and is finite;
- no residual is added;
- no direct matrix is allowed;
- `DIRECT_VECTOR` is one-dimensional fp16, at most 4,096 values;
- aggregate direct-vector bytes are reported.

Use only `genome produce-target` for production targets. It performs fit, both
refinement stages, serialization, byte audit, Runtime evaluation, binding, and
acceptance from one formula.

Each evaluation and acceptance report must bind:

- `run_id`;
- `formula_id`;
- `program_id`;
- program manifest SHA;
- payload SHA;
- W0 state ID;
- WT state ID;
- evaluation JSONL SHA;
- source-plan ID;
- full code commit.

## Gate 5 — frozen-formula rerun and training regeneration

Do these steps in order:

1. commit and test the protocol;
2. change the formula status from `formula-development` to `frozen` without
   changing its identity fields;
3. rerun Pythia 14M seed5 once with the exact frozen formula;
4. preserve its rejection and full diagnostics;
5. regenerate all other training targets with the same formula.

Seed5 is training and formula-development evidence. Its known 73.38% historical
result is not an accepted compiler target.

## Gate 6 — fresh seed7 development

Only after Gate 5 is complete:

1. run Pythia 14M seed7 once;
2. run Pythia 31M seed7 once;
3. use only the independent 128-batch verifier;
4. do not adjust the formula after seeing either result.

The code refuses seed7 while the formula is not frozen. It also checks that every
training report uses the same formula and source plan before seed7 can run.

## Gate 7 — compiler corpus

Build the corpus only after both seed7 reports exist.

The expected corpus is:

```text
16 accepted training targets
2 accepted development targets
18 total records
```

Pythia 14M seed5 is kept as rejection evidence and excluded from supervision.
Pythia 31M seed9 is absent. The builder recomputes and verifies every artifact
binding before it writes the corpus.

## Gate 8 — compiler smoke and production training

Do not start this gate before Gates 0–7 pass.

Teacher-forced loss is diagnostic. Checkpoint selection must use free-running
development evaluation:

```text
compiler generation
-> actual MGP serialization
-> byte audit
-> Runtime
-> real Pythia loss for at least 128 batches
-> endpoint progress
```

Select the checkpoint with the best mean generated endpoint progress across the
two seed7 lives. Store each generated program and its full evaluation report.

## Gate 9 — sealed hidden evaluation

For Pythia 31M seed9:

1. load W0 and endpoint-free evidence;
2. generate one primary candidate;
3. serialize and audit it;
4. execute it through the Runtime;
5. seal compiler, evidence, source-plan, MGP, and state hashes;
6. resolve hidden WT only after the seal;
7. evaluate W0, candidate, and WT identically.

Predeclared interpretation:

- `P <= 0`: no signal;
- `0 < P < 0.25`: weak signal only;
- `0.25 <= P < 0.80`: partial result;
- `P >= 0.80`: strong result.

No hidden result is currently claimed.
