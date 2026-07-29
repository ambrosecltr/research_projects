# Current local validation

This clean Track 2 tree was validated in the available CPU environment with:

```text
python 3.13.5
torch 2.10.0+cpu
```

Commands:

```bash
python -m compileall -q genome tests
python -m pytest -q
PYTHONPATH=. python -m genome --help
```

Result:

```text
18 passed
4 informational PyTorch Transformer nested-tensor warnings
0 failures
0 skips
```

The suite covers:

- whole-life and hidden split rules;
- undeclared and non-monotonic checkpoints;
- clean 17-training, two-development, one-hidden Pythia source split;
- hidden WT non-materialization;
- semantic fingerprint determinism;
- forbidden MGP primitives;
- compact program fitting, serialization, Runtime and audit;
- hierarchical compiler forward/backward and byte bound;
- compiler training smoke;
- compiler checkpoint resume;
- fresh workspace and CLI boundary.

The local environment did not contain `transformers` or `datasets`, so real Pythia loading, Hugging Face source resolution and RunPod GPU operations are intentionally the next agent's first environment-backed checks.
