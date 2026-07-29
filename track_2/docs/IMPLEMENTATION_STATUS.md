# Implementation status

## Complete

- GPU pod stopped.
- CPU pod stopped after audit export and verifier creation.
- 100 GB network volume retained.
- Correct whole-life split implemented and tested.
- Pythia 14M seed9 included as training.
- Pythia 31M seed9 remains the only hidden life.
- Seed8 reclassified as training and formula-development evidence.
- `DIRECT_VECTOR` policy formalized and aggregate bytes reported.
- One immutable target formula ID implemented.
- One `produce-target` production command implemented.
- Evaluation and acceptance artifact bindings implemented.
- Compiler-corpus binding revalidation implemented.
- Independent 128-batch development verifier created from another shard.
- Free-running generated-model checkpoint selection implemented.
- Hidden result tiers implemented.
- Historical non-weight audit bundle exported.
- 50 local tests pass.

## Historical evidence only

The old formula-v2 results for 14M seeds0–5,8 and 31M seed8 are preserved in
`artifacts/audit/pythia_v1_formula_v2_historical`.

Seed8 is not independent development evidence. Old reports predate the new
artifact bindings and cannot enter the compiler corpus.

The old Pythia 14M seed5 result was rejected at 73.38% endpoint progress. It must
be rerun once after the same formula is formally frozen.

## Not started

- No frozen-formula seed5 rerun.
- No full training-target regeneration under new bindings.
- No seed7 development target.
- No compiler corpus.
- No compiler training.
- No hidden compilation.
- No hidden WT reveal.

## Next safe action

Commit and push this protocol repair. Then update the network-volume checkout to
that commit. Change only the formula status to `frozen`, keep its identity fields
unchanged, and rerun Pythia 14M seed5 once with `genome produce-target`.
