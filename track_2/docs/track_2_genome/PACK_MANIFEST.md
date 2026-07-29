# GENOME documentation pack manifest

> **Archived pack manifest.** The recovery files supersede this original design pack. The old
> checksum list has been retired because it was stale and was not enforced by code.

**Pack name:** GENOME Track 2 Research and Implementation Pack  
**Generated:** 27 July 2026  
**Working project expansion:** Generative Endpoint Neural Operator for Model Emission

## Contents

| File | Purpose |
| --- | --- |
| `00_README.md` | Project identity, hypothesis, research levels, and reading order. |
| `AGENTS.md` | Mandatory behavioural rules for implementation agents. |
| `01_THEORY_AND_MATH.md` | Formal model, mathematical objectives, limitations, and central hypotheses. |
| `02_MODEL_GENOME_FORMAT.md` | Model Genome Program representation, opcodes, decoding, and byte accounting. |
| `03_SYSTEM_ARCHITECTURE.md` | Freezer, graph, compiler, interpreter, prober, linker, patcher, and Gate. |
| `04_TRACK1_DATA_CONTRACT.md` | Exact artifacts and data partitions required from Track 1. |
| `05_EXPERIMENT_PLAN.md` | Ordered experimental phases and pass/failure gates. |
| `06_EVALUATION_PROTOCOL.md` | Functional, size, compute, leakage, generation, and repair evaluation. |
| `07_IMPLEMENTATION_BLUEPRINT.md` | Suggested repository modules, APIs, CLIs, tests, and first vertical slice. |
| `08_AGENT_TASKS.md` | Narrow implementation cards from repository orientation through self-hosting. |
| `09_RSI_AND_FUTURE_RESEARCH.md` | Recursive self-improvement criteria and radical follow-on branches. |
| `10_REFERENCES.md` | Primary literature checked through 27 July 2026, with relevance and caveats. |
| `11_DECISION_LOG_TEMPLATE.md` | Required experiment record and decision template. |
| `12_GLOSSARY.md` | Project terminology, research levels, symbols, and metrics. |
| `13_PROJECT_INSTALLATION_AND_FIRST_RUN.md` | Placement instructions and the exact first agent handoff. |
| `reference/track_1_research_map.md` | Bundled snapshot of the supplied Track 1 research map. |

## Recommended entry points

- Human/project owner: `00_README.md`, then `13_PROJECT_INSTALLATION_AND_FIRST_RUN.md`.
- Coding agent: `AGENTS.md`, then the assigned task in `08_AGENT_TASKS.md`.
- Research agent: `01_THEORY_AND_MATH.md`, `05_EXPERIMENT_PLAN.md`, and `10_REFERENCES.md`.
- Evaluation agent: `06_EVALUATION_PROTOCOL.md` and `11_DECISION_LOG_TEMPLATE.md`.

## Integrity

The old `SHA256SUMS.txt` file was retired during recovery validation. It covered a changing source
tree, was stale, and no code enforced it. Immutable experiment artifacts retain their own hashes.
