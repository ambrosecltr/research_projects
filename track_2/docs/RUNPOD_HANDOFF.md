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

The local repository commit at this handoff is:

```text
93be185af603fb09e284990c60d3f9ef0ee66071
```

The matching RunPod tree has the same changes. Its commit is:

```text
a9b480893579ce8225cd8490986e850d4f05e018
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
38 passed
5 informational PyTorch nested-tensor warnings
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

## Work in progress

Gate 4 is applying the fixed formula to every training life.

At the time of this handoff, Pythia 14M seed1 is running in the background.

```text
PID: 19629
Log: /workspace/genome_v1/logs/target-pythia-14m-seed1-formula-v2.log
```

Check it with:

```bash
ps -p 19629 -o pid=,etime=,stat=
tail -50 /workspace/genome_v1/logs/target-pythia-14m-seed1-formula-v2.log
```

If a target does not pass the declared 80% gate, stop the target queue. Do not
train the compiler with a rejected target.

## Next gates

After all 17 training targets and both development targets have accepted programs:

1. Put each accepted program at
   `/workspace/genome_v1/programs/accepted/<run-id>`.
2. Build the 19-record corpus:

```bash
cd /workspace/genome_v1/repo/track_2
python3 -m genome.cli build-compiler-corpus \
  --plan /workspace/genome_v1/control/pythia_v1.pinned.json \
  --workspace /workspace/genome_v1 \
  --program-root /workspace/genome_v1/programs/accepted \
  --probe-jsonl /workspace/genome_v1/evidence/corpus/probes/refinement.jsonl \
  --output /workspace/genome_v1/compiler/corpus/pythia_v1.json
```

3. Check that the command reports 17 training, two development, and 19 total.
4. Run the tiny compiler smoke and checkpoint-resume test.
5. Start production compiler training only after the smoke gates pass.
6. Select the compiler checkpoint with the two development lives only.
7. Compile one Pythia 31M seed9 candidate.
8. Seal all required hashes.
9. Reveal hidden WT only after the seal.
10. Report the one-shot hidden result before any repair.

Do not push. Commit local changes only.
