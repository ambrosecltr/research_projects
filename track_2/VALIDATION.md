# Current validation

## Local

Environment:

```text
Python 3.11.15
PyTorch 2.4.1
```

Commands:

```bash
.venv/bin/python -m pytest -q
.venv/bin/ruff check \
  genome/compiler/model.py \
  genome/compiler/train.py \
  genome/compiler/data.py \
  tests/test_compiler.py \
  tests/test_training.py \
  --ignore B008
```

Result:

```text
38 passed
5 informational PyTorch nested-tensor warnings
0 failures
0 skips
```

## RunPod

Environment:

```text
Python 3.11.10
PyTorch 2.6.0+cu124
Transformers 4.57.6
RTX 4090
```

Result:

```text
38 passed
5 informational PyTorch nested-tensor warnings
0 failures
```

The suite covers:

- 17 training, two development, and one hidden life;
- Pythia 14M seed9 as training data;
- Pythia 31M seed9 hidden WT exclusion;
- source materialization rules;
- canonical state conversion;
- semantic evidence;
- compact-program fitting, serialization, Runtime, and audit;
- matrix scales, low-rank factors, and direct vectors;
- compiler byte limits;
- complete compiler-corpus construction;
- compiler training and checkpoint resume;
- workspace and command boundaries.

Real development results also passed:

```text
Pythia 14M seed8: 83.2048% endpoint progress
Pythia 31M seed8: 80.1290% endpoint progress
Required:          80.0000%
```
