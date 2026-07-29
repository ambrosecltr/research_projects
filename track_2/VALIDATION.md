# Current validation

## Local result

Command:

```bash
.venv/bin/python -m pytest -q
```

Result:

```text
50 passed
6 informational PyTorch nested-tensor warnings
0 failures
0 skips
```

The tests cover:

- the corrected 17-training, two-development, one-hidden split;
- Pythia 14M seed9 training inclusion;
- Pythia 31M seed9 hidden exclusion;
- `DIRECT_VECTOR` shape, size, dtype, and aggregate byte reporting;
- immutable target formula validation;
- complete artifact bindings;
- independent verifier separation and 128-batch minimum;
- compiler-corpus binding revalidation;
- 16-training plus two-development accepted-corpus count;
- free-running checkpoint selection interface;
- hidden result tiers;
- compiler checkpoint resume.

## Network-volume checks

- Historical audit bundle copied without weights.
- Bundle size: about 552 KB before the verifier receipt was added.
- Independent verifier: 128 batches.
- Formula sample shard: `document-00000-of-00020.bin`.
- Development verifier shard: `document-00001-of-00020.bin`.
- Verifier JSONL SHA:
  `6a9cfd8231943cc0603a7c40c9f6f0bfa02c7032e415ff452258c359a4e8cd99`.
- Hidden Pythia 31M seed9 WT directory: absent.
- GPU pod: stopped.
- CPU pod: stopped.
- Network volume: retained.

## Important limit

The historical seed8 results are not development validation. They are
formula-development evidence because seed8 influenced the formula.

Fresh development validation has not run. It will use seed7 only after the formula
is frozen and all training targets are regenerated with new bindings.
