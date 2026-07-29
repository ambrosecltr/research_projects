# GENOME recovery validation

This file is superseded by `VALIDATION_REPORT.md`. The recovery branch is validated independently
of the historical PolyPythia claims.

## Required checks

```bash
python -m compileall -q genome tests
python -m pytest -q
```

GitHub Actions are not used for this validation. All required checks run locally with Python 3.11.

The recovery PR must remain draft until:

- the complete historical suite passes;
- every new recovery test passes;
- V4-style residual programs are rejected;
- structured Runtime primitives round-trip;
- the compact target tokenizer has a deterministic inverse;
- the variable compiler performs finite forward/backward/generation tests.

No production training or large source download is permitted merely because unit tests pass. The additional R1–R4 research gates in `RECOVERY.md` still apply.
