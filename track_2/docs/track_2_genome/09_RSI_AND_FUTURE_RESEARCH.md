# GENOME recursive self-improvement and boundary-pushing research

> **Archived future-research note.** This file is outside the active recovery plan and cannot
> authorize learned-interpreter work or paid compute.

## 1. RSI target

The motivating idea is to remove the natural-language-and-code middle layer from model improvement.

A conventional loop is:

\[
\text{language} \rightarrow \text{research proposal} \rightarrow \text{code}
\rightarrow \text{training} \rightarrow \text{new model}.
\]

A GENOME loop aims for:

\[
\text{model/task evidence}
\rightarrow \text{genome transformation}
\rightarrow \text{candidate child model}
\rightarrow \text{hidden functional evaluation}.
\]

The compiler need not explain why the child works in English. It must emit a child that works.

This is not RSI merely because GENOME generates a poetry model. RSI begins when GENOME can generate or modify the **model-producing system itself** and the child system improves under an external, hidden evaluator.

---

## 2. Self-hosting notation

Let \(C_t\) be a compiler at generation \(t\). Let \(\mathcal I\) be the interpreter and \(\mathcal E\) an immutable evaluator over hidden child-model tasks.

The parent emits candidate child-compiler genomes:

\[
p_{t+1}^{(j)}\sim C_t(U_t,\xi_j),
\]

where \(U_t\) contains the compiler architecture, task archive, performance evidence, and allowed parent state.

Decode:

\[
C_{t+1}^{(j)}=\mathcal I(p_{t+1}^{(j)}).
\]

Each child compiler is then asked to create ordinary child models for hidden tasks \(\mathcal H\):

\[
M_{h}^{(j)}=C_{t+1}^{(j)}(h),\qquad h\in\mathcal H.
\]

The evaluator returns a vector:

\[
V(C)=
\left[
Q_{hidden},
-C_{child},
-B_{genome},
-R_{failure},
-G_{transfer}
\right],
\]

where quality, cost, genome size, failure rate, and transfer are kept as a Pareto vector rather than prematurely collapsed to one scalar.

Accept a child only if it improves the predeclared Pareto frontier and passes integrity checks.

---

## 3. Strong self-hosting criterion

A credible result requires all of the following:

1. The parent generated the child compiler’s weights/genome or a material architecture/weight patch.
2. The child was not hand-edited after generation.
3. Parent and child were evaluated on the same hidden task distribution and budget.
4. The child produced better unseen child models or equivalent models at lower cost.
5. Evaluator, hidden tasks, and acceptance rules were fixed before candidate generation.
6. All candidate search and verification cost was counted.
7. The accepted child can itself generate the next compiler generation using the same mechanism.

A weaker but useful precursor is **recursive compiler improvement with repair**, where the parent generates most of the child and a fixed small repair budget is allowed.

---

## 4. Development levels toward RSI

| Level | Capability |
| --- | --- |
| **R0** | GENOME represents an ordinary child endpoint. |
| **R1** | GENOME predicts ordinary child endpoints for hidden runs. |
| **R2** | GENOME predicts model edits: patches, merges, width/depth changes, or optimizer states. |
| **R3** | GENOME represents and reconstructs its own compiler/interpreter. |
| **R4** | Parent generates a functional child compiler. |
| **R5** | Child compiler beats parent on hidden child-generation tasks. |
| **R6** | Accepted child generates a further improved generation under the same protocol. |

Do not jump from R0 to claims of open-ended RSI.

---

## 5. Immutable evaluator design

The evaluator must be outside the generated compiler and difficult to game through memorization.

Hidden task suite should vary:

- initialization seeds;
- data orders;
- small unseen corpora;
- objective mixtures;
- model widths/depths within contract;
- corrupted checkpoints needing repair;
- missing-layer/tensor completion;
- endpoint compilation from differing prefix lengths;
- performance-versus-byte trade-offs.

Rotate a private reserve of tasks only between research generations, never within candidate selection for one declared result.

Score actual executed child models. Do not score the child compiler’s self-reported predictions.

---

## 6. Compiler genome

To close the loop, the compiler must itself be describable by MGP.

Possible staged approach:

1. Make a deliberately small compiler, for example 1M–10M parameters.
2. Fit a G0 genome for that compiler.
3. Train a parent to emit compiler-genome patches rather than complete weights.
4. Expand the patch scope over generations.
5. Eventually emit the complete compiler genome.

A child compiler may contain:

- architecture graph processor;
- dataset/trajectory encoders;
- genome heads;
- linker;
- patch allocator.

The shared interpreter can remain externally fixed initially. Later, interpreter evolution can be included as a separate genome with stricter compatibility tests.

---

## 7. Candidate generation strategies

### 7.1 Direct child compiler genome

Parent emits all child compiler codes.

### 7.2 Parent-to-child delta

\[
p_{child}=p_{parent}+\Delta p.
\]

This is likely easier and supports conservative improvement.

### 7.3 Module replacement

Parent emits only one module at a time:

- trajectory encoder;
- graph planner;
- code decoder;
- linker;
- entropy model;
- patch allocator.

### 7.4 Architecture-plus-weight genome

The parent emits discrete architecture opcodes and compatible weight codes. Begin with a constrained grammar rather than arbitrary source code.

### 7.5 Evolutionary population

Maintain an archive of non-dominated compiler genomes. Apply:

- mutation in latent space;
- structured module crossover;
- patch recombination;
- distillation from an ensemble of parents;
- novelty pressure over failure modes.

Selection remains external and hidden-task based.

---

## 8. Preventing evaluator overfitting

A self-improvement loop will exploit any evaluator weakness.

Use:

- hidden task seeds;
- multiple task families;
- held-out architectures;
- exact resource limits;
- replayable sandboxed evaluation;
- model output checks rather than self-reports;
- a public development evaluator and private result evaluator;
- adversarial corruption and distribution-shift tasks;
- failure penalties for NaNs, invalid genomes, and non-deterministic decode;
- diversity requirements so one narrow benchmark cannot dominate.

If a child improves only the visible development set, record evaluator overfitting rather than compiler improvement.

---

# High-risk research branches

## 9. Time as a coordinate: model-life neural field

Represent an entire trajectory with one field:

\[
W(t,l,m,i,j)=W_0(l,m,i,j)+F_\psi(z_{life},t,l,m,i,j).
\]

Inputs:

- normalized token progress \(t\);
- layer \(l\);
- tensor role \(m\);
- coordinates \(i,j\);
- life code \(z_{life}\).

Uses:

- query any checkpoint time;
- infer \(z_{life}\) from the first few checkpoints;
- query \(t=1\) for endpoint;
- identify layers/tensors that converge on different schedules;
- generate only future residuals;
- interpolate training regimes.

Train with checkpoint reconstruction plus trajectory-gradient consistency:

\[
\partial_tW(t)\approx \operatorname{UpdateRule}(W(t),D,r).
\]

This turns endpoint generation into temporal completion.

---

## 10. Learned boundary-value solver

Instead of imitating WT, solve for a good terminal condition:

\[
\nabla_W L_D(W)\approx0,
\]

subject to generalization and genome-rate constraints.

Iterative latent solver:

\[
p^{k+1}=p^k+F_\omega\left(
R\nabla_WL(\mathcal I(p^k)),
L(\mathcal I(p^k)),
\Phi(D),
\mathcal G
\right).
\]

This learns how to solve families of huge nonlinear systems in genome space. It does not need the original optimizer’s exact endpoint.

---

## 11. Frozen random substrate plus generated modulation

Keep a large deterministic random substrate and generate only structured modulation:

\[
W_l=W_{0,l}(s)+U_l(z)V_l(z)^\top+G_l(z)+S_l.
\]

Generate:

- low-rank updates;
- gates/routing;
- normalization scales;
- spectral envelopes;
- residual adapters;
- small embedding correction;
- output calibration.

This deliberately lowers endpoint degrees of freedom and may create a child architecture whose learned state has a short genome by construction.

---

## 12. Genome-native transformer

Every apparent child matrix is generated from codes during training and inference.

Train:

\[
\min_{z,\psi}L_D\left(W_0+D_\psi(z)\right)+\beta B(z).
\]

Variants:

- fixed shared interpreter, train only codes;
- slowly trained interpreter shared across model lives;
- role-specific interpreters;
- direct storage for embeddings/norms, generated transformer blocks;
- sparse exception parameters released on demand.

Then train a compiler to emit final \(z\). This may be much easier than compactly representing an arbitrary dense checkpoint after the fact.

---

## 13. Holographic depth generation

Generate layer codes from a recurrent dynamical system:

\[
z_{l+1}=F_\theta(z_l,u_l,z_D),
\]

\[
W_l=D_\psi(z_l,role,coordinates).
\]

A deep model is represented by:

- one initial state;
- one shared recurrence;
- compact layer control signals;
- sparse exceptions.

Use global backward/linker passes so early layer codes can respond to late-layer needs.

---

## 14. Influence-directed bit allocation

At a candidate \(W\), estimate functional value of a correction \(c\):

\[
\Delta L(c)\approx\nabla_WL(W)^\top c+\frac{1}{2}c^\top Hc.
\]

Approximate with gradient sketches, Hessian-vector products, Fisher blocks, or finite probes. Spend the next genome bits on:

\[
c^*=\arg\max_c\frac{-\widehat{\Delta L}(c)}{B(c)+\alpha C(c)}.
\]

This makes the genome an adaptive rate–distortion program rather than a uniform tensor compressor.

---

## 15. Model crossover in genome space

Once genomes are canonical and structured, recombine parents:

- global task code from one parent;
- early-layer genes from another;
- attention/MLP modules swapped by role;
- low-rank factors merged;
- sparse patches unioned and re-optimized;
- architecture genes crossed with compatibility checks.

Evaluate child function after every operation. Genome crossover may be less destructive than raw scalar-weight crossover.

---

## 16. Counterfactual architecture surgery

Train GENOME to transform a functioning model into a changed architecture:

- insert identity/residual-neutral layers;
- widen hidden dimensions;
- split heads;
- convert dense MLP to MoE;
- replace positional encoding;
- compress depth;
- change context length;
- add recurrence or shared layers.

Represent the surgery as:

\[
p_{new}=T_\phi(p_{old},\mathcal G_{old},\mathcal G_{new},\text{probe evidence}).
\]

A short repair phase is allowed and measured.

---

## 17. Generate optimizer state

Emit:

\[
(\widehat W,\widehat m,\widehat v,\widehat t)
\]

for Adam-like training, or equivalent state for the Track 1 optimizer.

Even if weights need repair, generated moments/preconditioners may place repair on a much faster path than zero-initialized optimizer state.

Evaluate:

- weight-only candidate + fresh optimizer;
- weight + generated moments;
- early checkpoint + generated future moments;
- genome latent optimizer state.

Count optimizer-state genome bytes.

---

## 18. On-demand weights

Do not materialize a static checkpoint. Decode each layer before execution:

```text
genome + interpreter -> layer weights -> execute layer -> release/cache
```

Potential benefits:

- very small stored child;
- dynamic task-conditioned layers;
- hardware-aware block generation;
- selective high-precision generation;
- mixture of layer genomes.

Costs:

- repeated decode latency;
- deterministic caching complexity;
- interpreter becomes part of every inference;
- harder deployment.

First prove generate-and-cache; on-demand decode is later.

---

## 19. Causal corpus genes

Try to decompose the dataset fingerprint and endpoint genome additively:

\[
z_D\approx\sum_{x\in D}w_x\phi(x),
\]

\[
p_D\approx p_{base}+\sum_k a_k g_k.
\]

This could enable:

- corpus mixture control;
- data attribution;
- unlearning by subtracting a corpus gene;
- adding a new domain without full retraining;
- synthesizing new corpus combinations.

Additivity will not be exact because training is nonlinear. Treat it as a local or learned composition law.

---

## 20. Active loss-landscape interrogation

Let the compiler choose which batches or synthetic probes to evaluate next.

At step \(k\), choose query \(q\) maximizing expected genome uncertainty reduction:

\[
q^*=\arg\max_q I(p_T;E_q\mid E_{1:k}).
\]

Queries may be:

- real corpus batches;
- learned synthetic token sequences;
- adversarial prompts;
- activation probes;
- random gradient directions;
- architecture-local tests.

This turns early training into an active diagnostic protocol rather than a fixed prefix.

---

## 21. Jointly generate a condensed dataset and genome

The compiler may emit both:

\[
(S,p),
\]

where \(S\) is a tiny synthetic corpus or learned token/embedding probe set and \(p\) is an initial genome.

Then perform a few latent or child updates on \(S\). Optimize:

\[
L_{real}(\operatorname{Train}(\mathcal I(p),S,K))
+\beta B(p)+\gamma B(S)+\eta K.
\]

This merges dataset condensation with endpoint compilation and may provide a compact correction mechanism.

---

## 22. Model-to-formula distillation

After a neural compiler works, search for simpler symbolic or programmatic rules that imitate it.

Targets:

- latent update law;
- rank allocation rule;
- per-role code evolution;
- patch-priority formula;
- dataset-fingerprint-to-global-code map.

Use program synthesis/evolutionary search and require the distilled rule to pass unseen-run gates. Interpretability is a later deliverable, not a prerequisite for discovering the mechanism.

---

## 23. Interpreter evolution

Eventually the parent may improve not only genome codes but the shared interpreter.

Compatibility problem:

\[
W=\mathcal I_{old}(p_{old})
\]

may not equal:

\[
W=\mathcal I_{new}(p_{old}).
\]

Solutions:

- version every interpreter and preserve old runtimes;
- learn genome migration \(M(p_{old})=p_{new}\);
- require backward-compatible decoder heads;
- evolve interpreter through residual patches;
- periodically recompile the archive and verify all phenotypes.

Interpreter evolution has high leverage but can invalidate the entire genome archive, so defer it until the fixed-interpreter system is strong.

---

## 24. Self-describing and self-verifying genomes

A future MGP could include compact predicted invariants:

- expected tensor norms/spectra;
- probe-output hashes/quantized signatures;
- expected loss ranges;
- layer activation envelopes;
- repair recommendations;
- uncertainty by tensor.

The runtime can reject malformed or corrupted expansions before full evaluation. Predictions are diagnostics, not a replacement for the external evaluator.

---

## 25. Recommended order for radical branches

1. Time-as-coordinate field over R0 trajectory.
2. Latent boundary-value refinement.
3. Frozen substrate + generated modulation.
4. Genome-native child architecture.
5. Generate optimizer state.
6. Active probes.
7. Architecture surgery.
8. Genome crossover/population search.
9. Compiler self-representation.
10. Parent-generated compiler patches.
11. Full self-hosting generation.
12. Interpreter evolution.

This order keeps every radical branch connected to a measurable capability already established in earlier phases.
