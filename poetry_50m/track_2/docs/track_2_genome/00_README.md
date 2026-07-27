# GENOME: Track 2 research and implementation pack

**Working project name:** **GENOME — Generative Endpoint Neural Operator for Model Emission**

GENOME is Track 2 of the poetry-model research project. Track 1 asks whether an observed training trajectory can be shortened. Track 2 asks a more radical question:

> Can the learned endpoint of training be represented as a compact executable model genome, and can another model learn to emit that genome directly instead of replaying the entire optimization process?

The first target is deliberately small and controllable: the fully trained Track 1 poetry model. That reference model is called **R0** throughout these documents. R0 gives GENOME a known architecture, known corpus, known training process, known endpoint, and a direct quality baseline.

This is a personal research project. It is allowed to be unconventional, hybrid, messy, and opportunistic. A useful result does not need to prove a universal theorem. A same-run result that materially compresses or accelerates the known Track 1 endpoint is worthwhile, provided it is described honestly.

---

## The core hypothesis

Let ordinary training be a deterministic program once every relevant input and random choice is fixed:

\[
W_T = \mathcal A(D,\mathcal G,s,r,e),
\]

where:

- \(D\) is the training data and exact order;
- \(\mathcal G\) is the model architecture graph;
- \(s\) is initialization and stochastic state;
- \(r\) is the optimizer, schedule, and training recipe;
- \(e\) is the numerical/software execution environment;
- \(W_T\) is the final trained checkpoint.

GENOME attempts to learn a compiler:

\[
p = C_\phi\!\left(\Phi(D),\mathcal G,s,r,\tau_{0:k}\right),
\]

and an interpreter:

\[
\widehat W = \mathcal I_\psi(p,W_0,\mathcal G),
\]

where:

- \(p\) is a compact executable **model genome**;
- \(\Phi(D)\) is a model-native fingerprint of the dataset;
- \(\tau_{0:k}\) is an optional short prefix of normal training;
- \(W_0\) is the reproducible initialization;
- \(\widehat W\) is the generated candidate checkpoint.

The practical success condition is not necessarily \(\widehat W=W_T\). The useful target is:

\[
f_{\widehat W} \approx f_{W_T}
\]

while:

\[
T_{\text{fingerprint}}+T_{\text{compile}}+T_{\text{decode}}+T_{\text{verify}}+T_{\text{repair}}
\ll T_{\text{ordinary training}}.
\]

---

## Naming and fixed terminology

| Term | Meaning |
| --- | --- |
| **GENOME** | The Track 2 project and eventual compiler family. |
| **R0** | The immutable fully trained Track 1 reference endpoint. |
| **W0** | The initial parameters corresponding to R0 before training. |
| **WT** | The final parameters of R0. |
| **Delta-T** | The learned displacement, \(\Delta_T=W_T-W_0\). |
| **Model genome** | A compact program or latent representation that expands into a candidate trained checkpoint. |
| **MGP** | Model Genome Program, the serialized per-model output. |
| **Interpreter** | Shared deterministic code/neural decoder that expands an MGP into weights. |
| **Compiler** | Model that predicts an MGP from architecture, task/data evidence, initialization, and optionally an early trajectory. |
| **Linker** | Iterative component that refines genome codes after probing the generated model. |
| **Patch** | Sparse, low-rank, or residual correction not captured by the main genome. |
| **Genome Gate** | Immutable evaluator that accepts or rejects generated candidates. |
| **Phenotype** | The expanded runnable child model produced by a genome. |

The name is a working name, not a scientific commitment. Do not spend implementation time renaming components unless there is a concrete collision.

---

## Research levels

GENOME must separate these claims. Passing an earlier level is useful even if later levels fail.

| Level | Name | What is allowed to be seen | Claim |
| --- | --- | --- | --- |
| **G0** | Endpoint representation | Full R0 endpoint | R0 has a compact executable representation that preserves useful function. |
| **G1** | Same-run endpoint compilation | R0 trajectory prefix and other R0 training records, but not the final target at inference | A learned compiler can predict the R0 endpoint or a matched-quality alternative from early evidence. |
| **G2** | Held-out run compilation | Other runs from the same architecture/corpus; target run endpoint withheld | The compiler transfers to an unseen seed or data order. |
| **G3** | Structural transfer | Multiple corpora, widths, depths, or objectives | The compiler transfers across a material task or architecture change. |
| **G4** | Self-hosting | Compiler-family archive and hidden evaluator | A generated child compiler beats its parent at producing unseen child models under the same budget. |

Never describe G0 as a general optimizer. Never describe G1 as cross-seed transfer. Never describe a generated but heavily repaired model as zero-training synthesis.

---

## The minimum viable path

The fastest route to useful evidence is:

1. Freeze R0, W0, the architecture manifest, final metrics, and a hidden verifier split.
2. Build a reliable evaluator before building a genome model.
3. Measure the rate–distortion curve of Delta-T with ordinary structured codecs.
4. Fit a shared neural interpreter with learned per-tensor genome codes.
5. Determine how many genome bits are needed to recover R0 function.
6. Add probe-time latent refinement and a sparse exception patch.
7. Only then train a compiler to predict the genome from an early trajectory or dataset fingerprint.
8. Test a genuinely held-out run before making transfer claims.

This ordering matters. It separates three questions that are easy to accidentally conflate:

- **Does a compact representation exist?**
- **Can a model infer that representation?**
- **Can it infer it for a target it has never seen?**

---

## Required documents and reading order

1. [`AGENTS.md`](AGENTS.md) — operating rules for implementation agents.
2. [`01_THEORY_AND_MATH.md`](01_THEORY_AND_MATH.md) — formal problem, limits, and objectives.
3. [`02_MODEL_GENOME_FORMAT.md`](02_MODEL_GENOME_FORMAT.md) — executable genome representation.
4. [`03_SYSTEM_ARCHITECTURE.md`](03_SYSTEM_ARCHITECTURE.md) — complete system design.
5. [`04_TRACK1_DATA_CONTRACT.md`](04_TRACK1_DATA_CONTRACT.md) — exactly what must be extracted from Track 1.
6. [`05_EXPERIMENT_PLAN.md`](05_EXPERIMENT_PLAN.md) — ordered experimental program and gates.
7. [`06_EVALUATION_PROTOCOL.md`](06_EVALUATION_PROTOCOL.md) — metrics, leakage rules, and compute accounting.
8. [`07_IMPLEMENTATION_BLUEPRINT.md`](07_IMPLEMENTATION_BLUEPRINT.md) — suggested modules, APIs, CLIs, tests, and configuration.
9. [`08_AGENT_TASKS.md`](08_AGENT_TASKS.md) — agent-sized implementation tasks with definitions of done.
10. [`09_RSI_AND_FUTURE_RESEARCH.md`](09_RSI_AND_FUTURE_RESEARCH.md) — self-hosting loop and high-risk branches.
11. [`10_REFERENCES.md`](10_REFERENCES.md) — relevant primary research and why it matters.
12. [`11_DECISION_LOG_TEMPLATE.md`](11_DECISION_LOG_TEMPLATE.md) — experiment record template.
13. [`12_GLOSSARY.md`](12_GLOSSARY.md) — concise terminology and symbol reference.
14. [`13_PROJECT_INSTALLATION_AND_FIRST_RUN.md`](13_PROJECT_INSTALLATION_AND_FIRST_RUN.md) — integration instructions and exact first agent handoff.
15. [`PACK_MANIFEST.md`](PACK_MANIFEST.md) — pack contents and integrity notes.
16. [`reference/track_1_research_map.md`](reference/track_1_research_map.md) — bundled Track 1 research map.

---

## Non-negotiable scientific rules

- R0 is immutable. Never overwrite it and never silently change its tokenizer, architecture, or evaluation data.
- Preserve tensor names, shapes, dtypes, tied-weight relationships, and ordering in a machine-readable manifest.
- Evaluate function as well as parameter distance. Weight MSE alone is never a success criterion.
- Count all representation bits. A giant shared decoder cannot be ignored when claiming one-model compression.
- Count all child-generation compute: fingerprinting, compilation, decoding, candidate sampling, verification, and repair.
- Raw cross-seed weight averaging or regression is prohibited until a function-preserving alignment has passed verification.
- The verifier set must not be used to fit genome codes, train the compiler, tune thresholds, or choose candidates.
- Failures are data. Record them with their exact configuration and resulting metrics.
- Begin with the smallest test that can falsify the next assumption. Do not build the RSI loop before proving G0.

---

## What would count as an exciting first result?

Any one of the following would justify continuing:

- Delta-T can be represented at a small fraction of the raw checkpoint size while R0 validation behaviour remains within its terminal training noise.
- A shared interpreter plus small per-tensor codes outperforms SVD, quantization, and low-rank-plus-sparse baselines at the same total byte budget.
- Optimizing only genome codes on a small probe set reaches R0-equivalent quality far faster than optimizing all model weights.
- The first 0.5–2% of R0 training predicts a genome that needs only a small repair budget.
- A compiler trained on cheap sibling runs produces a useful endpoint for a hidden seed.
- The compiler finds a functionally strong endpoint that is far from WT in raw weight coordinates.

A negative result is also useful if it identifies the obstruction: initialization entropy, a particular tensor family, insufficient dataset fingerprinting, unstable alignment, too-large decoder cost, or a need for genome-native architecture co-design.
