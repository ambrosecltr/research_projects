# RunPod handoff

## Resources

```text
Network volume ID: 4kwmhcepgj
Name:              genome-pythia-v1
Region:            EU-RO-1
Size:              100 GB
Mount:             /workspace

CPU pod ID:        ufz8yhwxxmzufs
CPU state:         stopped

GPU pod ID:        g3ou14zh0wb7q7
GPU type:          RTX 4090
GPU state:         stopped
```

Both pods are stopped. The network volume is intact. Do not start the GPU until
the repaired protocol commit is present on the volume and its tests pass there.

Repository and experiment roots:

```text
/workspace/genome_v1/repo
/workspace/genome_v1
```

Branch:

```text
track_2/pythia-seed9-training
```

## Correct split

```text
Training:
  Pythia 14M seeds0–6,8,9
  Pythia 31M seeds0–6,8

Development:
  Pythia 14M seed7
  Pythia 31M seed7

Hidden:
  Pythia 31M seed9
```

Seed8 is training and formula-development evidence. Do not call its results
development confirmation. Pythia 14M seed9 is a training life. Pythia 31M seed9
is the only hidden life and still has no WT directory.

## Data on the volume

Existing even records are refinement data. Existing odd records are formula-tuning
data.

The fresh development verifier is:

```text
/workspace/genome_v1/evidence/corpus/verifier/tokens.jsonl
/workspace/genome_v1/evidence/corpus/verifier/receipt.json
```

It has 128 batches, uses `document-00001-of-00020.bin`, and has SHA:

```text
6a9cfd8231943cc0603a7c40c9f6f0bfa02c7032e415ff452258c359a4e8cd99
```

## Historical results

These results are audit evidence only. They predate the new bindings.

| Life | Progress | Meaning now |
|---|---:|---|
| 14M seed0 | 83.85% | formula development |
| 14M seed1 | 81.88% | formula development |
| 14M seed2 | 81.46% | formula development |
| 14M seed3 | 80.74% | formula development |
| 14M seed4 | 84.80% | formula development |
| 14M seed5 | 73.38% | rejected formula evidence |
| 14M seed8 | 83.20% | formula development, not development |
| 31M seed8 | 80.13% | formula development, not development |

No queue is active. No compiler corpus exists. No compiler training has started.

## Protocol now implemented locally

- Formula ID:
  `4f4e6d9d5d9ef7677dd955bb89be81dfedf161ecb010fdfd405475fdce46d155`.
- Formula status is still `formula-development`.
- Production uses one `genome produce-target` command.
- Evaluation and acceptance reports contain complete immutable bindings.
- Corpus construction checks those bindings again.
- Development requires at least 128 batches from the independent verifier.
- Compiler checkpoints are selected by free-running generated endpoint progress.
- Hidden progress below 25% is weak signal only; 80% or above is strong.

## Exact next order

1. Fetch the final repaired commit into `/workspace/genome_v1/repo`.
2. Start the CPU pod first.
3. Run the full tests on the volume checkout.
4. Confirm hidden WT is still absent.
5. Change only `status: formula-development` to `status: frozen`.
6. Commit and push that status change.
7. Update the volume checkout to the frozen commit.
8. Start the GPU.
9. Rerun Pythia 14M seed5 once with `genome produce-target`.
10. Keep its full rejection diagnostics.
11. Regenerate all other training targets with the same command and formula.
12. Run 14M seed7 once, then 31M seed7 once.
13. Build the 18-record corpus: 16 accepted training and two development.
14. Run compiler smoke.
15. Start compiler training only if all earlier gates pass.

Do not change the 10% byte budget, lower the 80% gate, add a residual, add direct
matrices, or treat teacher-forced loss as checkpoint quality.

Do not resolve or download Pythia 31M seed9 WT before the hidden prediction seal.
