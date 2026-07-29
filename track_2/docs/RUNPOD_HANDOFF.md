# RunPod handoff

## Active resources

Use the existing clean experiment resources:

```text
Network volume: genome-pythia-v1
Volume ID:      4kwmhcepgj
Region:         EU-RO-1
Size:           100 GB
Mount:          /workspace

CPU pod ID:     ufz8yhwxxmzufs
CPU pod state:  stopped

GPU pod:        genome-pythia-v1-gpu
GPU pod ID:     g3ou14zh0wb7q7
GPU:            RTX 4090
```

Repository and experiment roots:

```text
/workspace/genome_v1/repo
/workspace/genome_v1
```

The local branch is:

```text
track_2/pythia-seed9-training
```

The latest synced local implementation commit at this handoff is:

```text
02facf6408146329ad82880bd5ea7197943cda19
```

The matching RunPod implementation commit is:

```text
92e6fa1e4a0d5461c3cb54b5db6d5b6de73ea6a0
```

The hashes differ because the commits were copied with `git am`. The file content
is the same.

## Environment

RunPod uses:

```text
Python 3.11.10
PyTorch 2.6.0+cu124
Transformers 4.57.6
```

The current RunPod test result is:

```text
39 passed
6 informational PyTorch nested-tensor warnings
0 failures
```

## Completed gates

Gates 0 to 3 are complete.

- The source plan has 17 training lives, two development lives, and one hidden life.
- Pythia 14M seed9 is a training life. Its W0 and WT are materialized.
- Pythia 31M seed9 is the only hidden life. Only W0 is materialized.
- Do not resolve, download, or read Pythia 31M seed9 WT before the prediction seal.
- All available lives have canonical states, architecture graphs, recipes, evidence,
  and finalized life manifests.
- The even-record refinement probe and odd-record evaluation probe are separate.

The fixed target formula is in:

```text
configs/targets/pythia_v1.yaml
```

The formula passed both development lives:

| Life | Serialized fraction | Endpoint progress | Result |
|---|---:|---:|---|
| Pythia 14M seed8 | 9.9994% | 83.2048% | accepted |
| Pythia 31M seed8 | 8.4981% | 80.1290% | accepted |

The selected formula uses:

- base-relative row and column matrix scaling;
- rank-balanced low-rank factors;
- one shared vocabulary left factor;
- direct fp16 vectors no larger than 4,096 values;
- 16,384 refinement steps at 0.001;
- 2,048 final refinement steps at 0.0003;
- teacher KL weight 1.0;
- no payload anchor.

The compiler now emits the same active instruction family. It also reuses one
generated vocabulary factor, counts all serialized payload terms, reports
primitive and matrix-rank accuracy, and masks rank loss for tensors that do not
have a rank.

## Work in progress

Gate 4 is applying the fixed formula to every training life.

Accepted training results so far:

| Life | Endpoint progress |
|---|---:|
| Pythia 14M seed0 | 83.8524% |
| Pythia 14M seed1 | 81.8775% |
| Pythia 14M seed2 | 81.4617% |

Pythia 14M seed3 is the active target.

```text
14M queue PID:        21116
14M queue log:        /workspace/genome_v1/logs/target-pythia-14m-remaining-formula-v2.log
31M queue PID:        21144
31M queue log:        /workspace/genome_v1/logs/target-pythia-31m-training-formula-v2.log
Corpus watcher PID:   21242
Corpus watcher log:   /workspace/genome_v1/logs/build-compiler-corpus.log
```

Check the queues with:

```bash
ps -p 21116,21144,21242 -o pid=,etime=,stat=
tail -50 /workspace/genome_v1/logs/target-pythia-14m-remaining-formula-v2.log
```

The 14M queue runs seeds3–7 and seed9 one at a time. The 31M queue cannot start
until every required 14M life is accepted. It then runs training seeds0–7. The
corpus watcher cannot write the corpus until all 19 training and development
programs are accepted. Each queue stops on the first failed command.

If a target does not pass the declared 80% gate, stop the target queue. Do not
train the compiler with a rejected target.

## Next gates

After all 17 training targets and both development targets have accepted programs,
the active corpus watcher will build:

```text
/workspace/genome_v1/compiler/corpus/pythia_v1.json
```

Then:

1. Check that the corpus reports 17 training, two development, and 19 total.
2. Run the tiny compiler smoke and checkpoint-resume test.
3. Start production compiler training only after the smoke gates pass.
4. Select the compiler checkpoint with the two development lives only.
5. Compile one Pythia 31M seed9 candidate.
6. Seal all required hashes.
7. Reveal hidden WT only after the seal.
8. Report the one-shot hidden result before any repair.

Do not push. Commit local changes only.
