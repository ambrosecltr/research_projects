# GENOME code map

| Path | Responsibility |
|---|---|
| `genome/adapters/base.py` | Strict generic boundary for any source training project. |
| `genome/adapters/poetry50m.py` | Exact adapter, preflight, lineage/receipt validation, W0, packed evaluation, and evaluation-checkpoint export for this repository's Track 1. |
| `genome/specimen.py` | Freeze, load, hash and verify immutable W0/WT specimens. |
| `genome/tensor_inventory.py` | Canonical ordering, roles, layers and tied groups. |
| `genome/state.py` | Delta-T creation, state validation and statistics. |
| `genome/mgp/` | MGP validation, serialization and deterministic interpretation. |
| `genome/codecs/` | Dense, int8/int4, SVD and low-rank-plus-sparse genomes. |
| `genome/codecs/workspace.py` | One-time Delta-T/full-SVD cache reused across complete rank frontiers. |
| `genome/evaluator.py` | Adapter-driven functional evaluation and Genome Gate reports. |
| `genome/metrics.py` | Parameter, logit, agreement and perplexity metrics. |
| `genome/sensitivity.py` | Spectral and perturbation diagnostics. |
| `genome/bit_accounting.py` | Target-specific, single-model and amortized byte totals. |
| `genome/architecture_graph.py` | Tensor graph and compiler-ready structural features. |
| `genome/fingerprint.py` | CountSketch gradient/data fingerprints. |
| `genome/trajectory.py` | Model-life trajectory features and transparent extrapolation. |
| `genome/model_life.py` | Persistent run-level corpus index. |
| `genome/neural/block_decoder.py` | Shared role-conditioned block interpreter. |
| `genome/neural/autodecoder.py` | G0 fitting of global/layer/tensor genome codes. |
| `genome/neural/compiler.py` | Baseline G1 probabilistic code compiler. |
| `genome/neural/compiler_training.py` | Run-level compiler training and safe artifacts. |
| `genome/repair/` | Latent-only and full-child-weight repair baselines. |
| `genome/rate_distortion.py` | Reproducible transparent frontier orchestration. |
| `genome/reporting.py` | Report discovery and Markdown/CSV/JSON summaries. |
| `genome/experiment.py` | Run IDs, resolved configs, environment and Git state. |
| `genome/cli.py` | Unified command-line interface. |
| `examples/tiny_track1.py` | Executable causal-LM reference fixture. |
| `examples/track1_adapter_template.py` | Template retained for other Track 1 implementations; poetry50m uses the built-in concrete adapter. |
| `tests/` | Scientific-foundation regression tests. |
