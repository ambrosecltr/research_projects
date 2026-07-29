# GENOME code map

## Active recovery path

| Path | Responsibility |
|---|---|
| `RECOVERY.md` | Recovery decision, architecture and experiment gates. |
| `AGENTS.md` | Mandatory rules for future work. |
| `genome/life_schema.py` | Strict complete/partial/endpoint-only multi-stage model-life records and whole-lineage splits. |
| `genome/semantic_fingerprint.py` | Content-derived corpus statistics plus W0 gradient/activation evidence; no hash-derived semantic vectors. |
| `genome/mgp/` | Deterministic MGP schema, serializer, compact-target policy, validation and Runtime. |
| `genome/mgp/policy.py` | Rejects dense/full residual/per-weight target labels and enforces byte budgets. |
| `genome/compact_targets.py` | Canonical low-rank compact target fitting and global component allocation. |
| `genome/program_tokens.py` | Deterministic compact MGP target tokenization and inverse reconstruction. |
| `genome/program_compiler.py` | Variable-stage/tensor graph compiler that emits MGP tokens and coefficient chunks. |
| `genome/evaluator.py` | Functional Genome Gate and matched-model evaluation. |
| `genome/adapters/` | Reversible source-specific native/canonical mappings. |
| `genome/codecs/` | Transparent G0/rate-distortion baselines; dense codecs are not compiler labels. |
| `tests/test_life_schema_v2.py` | Multi-stage, hidden endpoint and split-leakage contracts. |
| `tests/test_semantic_fingerprint.py` | Semantic corpus and W0-response evidence. |
| `tests/test_structured_mgp.py` | Deterministic Kronecker, DCT, basis and codebook execution. |
| `tests/test_compact_targets.py` | Compact label fitting and rejection of V4/dense disguises. |
| `tests/test_program_tokens.py` | Canonical program sequence round trip. |
| `tests/test_program_compiler.py` | Variable compiler forward/backward/generation smoke. |

## Preserved infrastructure

| Path | Responsibility |
|---|---|
| `genome/adapters/poetry50m.py` | Track 1 deterministic G0 and evaluation boundary. |
| `genome/adapters/gpt_neox.py` | Strict GPT-NeoX native/canonical conversion and model assembly. |
| `genome/polypythia/hub.py` | Immutable Hugging Face source planning, endpoint download and hidden reveal control. |
| `genome/polypythia/lives.py` | Legacy Round One canonical life loader. New corpora use `life_schema.py`. |
| `genome/polypythia/evaluate.py` | Hidden Runtime, Wikitext and pinned LM Evaluation Harness comparisons. |
| `POLYPYTHIA_ROUND1_RESULTS.md` | Immutable failed-experiment evidence and metrics. |
| `configs/lm_eval_round1/` | Pinned Round One task definitions and dataset revisions. |

## Legacy/failed Round One path

The following remain only for reproduction and will be moved under an archive namespace after the recovery foundation is green:

| Path | Status |
|---|---|
| `genome/neural/block_decoder.py` residual mode | V4 stored one residual value per weight; forbidden for active compiler targets. |
| `genome/neural/multilife_decoder.py` | Failed V1–V4 shared-decoder experiment. |
| `genome/neural/blockwise_compiler.py` | Fixed-family blockwise raw endpoint regressor. |
| `genome/neural/predictive_compiler.py` | Failed hidden seed9 compiler path. |
| `configs/polypythia_14m_round1.yaml` | Frozen historical experiment configuration. |
| `POLYPYTHIA_ROUND1.md` | Historical execution runbook, not the next-step plan. |
| `tests/test_polypythia_round1.py` | Reproduction/leakage regression tests for the historical branch. |

No active code should import the legacy decoder/compiler path for new production experiments.
