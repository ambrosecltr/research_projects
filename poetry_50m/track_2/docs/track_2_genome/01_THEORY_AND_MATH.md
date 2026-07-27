# GENOME theory and mathematics

## 1. Formal problem

Ordinary training is a map from a complete training specification to a final parameter state:

\[
W_T=\mathcal A(D,\mathcal G,s,r,e),
\]

where \(e\) includes numerical details that are often omitted from informal statements of determinism: framework and kernel versions, precision, distributed reduction order, nondeterministic operators, and hardware-dependent behaviour.

GENOME attempts to replace most of the repeated execution of \(\mathcal A\) with a learned conditional solution operator:

\[
C_\phi:\left(\Phi(D),\mathcal G,s,r,\tau_{0:k}\right)\mapsto p,
\]

\[
\mathcal I_\psi:(p,W_0,\mathcal G)\mapsto \widehat W.
\]

The compiler emits a compact genome \(p\). The interpreter expands it into a runnable checkpoint. Optional probe/refinement stages transform the initial genome into a corrected genome:

\[
p^{(j+1)}=p^{(j)}+\mathcal L_\omega\left(p^{(j)},E^{(j)},\Phi(D),\mathcal G\right),
\]

where \(E^{(j)}\) is evidence gathered by executing the candidate model.

The central research question is not whether the mapping exists. A deterministic training implementation already proves that some mapping exists. The question is whether the target endpoint has a **shorter useful conditional description** than replaying the entire training program.

---

## 2. Determinism is not the same as compressibility

A deterministic function can be computationally irreducible in practice. The shortest exact description of a final checkpoint may be close to:

> initialize the model and run the original optimizer for all original steps.

Therefore, the project must test rather than assume:

\[
K_\epsilon\!\left([W_T]\mid W_0,\mathcal G,\Phi(D),r,\mathcal I_\psi\right)
\ll |W_T|,
\]

where:

- \(K_\epsilon\) is an approximate conditional description length;
- \([W_T]\) is the class of parameter states with sufficiently similar function;
- \(|W_T|\) is the raw checkpoint description length.

There are \(2^{bd}\) possible tensors with \(d\) parameters represented by \(b\) bits. An \(m\)-bit genome can distinguish at most \(2^m\) outputs without help from conditioning information or a shared decoder. A short genome therefore cannot represent every arbitrary checkpoint. GENOME can work only if trained endpoints occupy a highly structured subset of weight space and that structure is shared across targets.

This limitation is useful: it forces exact byte accounting and prevents a decoder from hiding one checkpoint per target.

---

## 3. Predict the learned displacement, not random initialization entropy

Decompose the endpoint as:

\[
W_T=W_0+\Delta_T.
\]

The initialization contains random-looking values whose shortest practical description is normally the initialization algorithm plus random state. The useful learning signal is concentrated in:

\[
\Delta_T=W_T-W_0.
\]

The default GENOME representation is therefore:

\[
\widehat W=W_0+\widehat\Delta.
\]

Benefits:

1. The genome does not waste capacity reproducing pseudorandom initialization values.
2. Same-run coordinates are naturally aligned.
3. Per-tensor delta norms reveal where the corpus and objective changed the model most.
4. Low-rank, spectral, and shared-basis structure is often easier to detect in an update than in a random-plus-update matrix.
5. A generated delta can be gated, interpolated, or partially applied.

Two baseline modes are allowed:

- **Seed replay mode:** regenerate W0 exactly from a pinned initialization manifest.
- **Base checkpoint mode:** store W0 as a shared base artifact and encode only Delta-T.

Base checkpoint mode is acceptable for G0. Seed replay mode is preferred for a strong compression claim. Report which mode was used.

---

## 4. Three endpoint targets

### 4.1 Bitwise endpoint

\[
\widehat W=W_T.
\]

This is the strict lossless-compression target. It is useful for studying literal checkpoint redundancy, but floating-point bit patterns are unforgiving and this target is not required for useful model synthesis.

### 4.2 Functional imitation

Given an immutable anchor/verifier set \(A\):

\[
D_f\left(f_{\widehat W}(A),f_{W_T}(A)\right)\le \epsilon_f.
\]

Possible \(D_f\) terms include logit KL divergence, top-k agreement, hidden-state similarity, attention statistics, and generation behaviour.

### 4.3 Independent matched-quality endpoint

\[
L_D(\widehat W)\le L_D(W_T)+\epsilon_L
\]

without requiring closeness in raw coordinates.

This is the ultimate target. If GENOME discovers a different basin with equal or better held-out behaviour, that is success.

---

## 5. Weight-space symmetries and equivalence classes

Neural networks contain function-preserving reparameterizations. Examples include:

- permutations of MLP hidden channels with the inverse permutation applied to the following projection;
- permutations of attention heads with corresponding permutations in the output projection;
- certain paired basis transforms in query/key or value/output spaces;
- scale transformations in architectures with compensating homogeneous paths;
- factorization ambiguity in low-rank adapters.

Consequently, the Euclidean distance between independently trained checkpoints can be large even when their functions are similar.

GENOME follows four rules:

1. Within one continuous run, raw coordinates may be used.
2. Across independent runs, raw coordinates are not assumed comparable.
3. Alignment must be verified by executing the transformed model and checking that outputs are preserved.
4. If reliable alignment is unavailable, use invariant/equivariant representations, spectra, activation signatures, or function-space targets.

The ideal object is not one checkpoint but an equivalence class:

\[
[W]=\{W': f_{W'}=f_W\}.
\]

A genome compiler should be rewarded for reaching any good member of the class, not punished for avoiding the arbitrary representative produced by the original optimizer.

---

## 6. Hierarchical conditional description

A 50M-parameter model should not be represented as one sequence containing 50M scalar tokens. Use a hierarchy:

\[
z_{\text{model}}
\rightarrow z_{\text{layer}}
\rightarrow z_{\text{tensor}}
\rightarrow z_{\text{block}}
\rightarrow W[i,j].
\]

Let the model contain tensors \(W_1,\ldots,W_M\). The compiler emits:

\[
p=\left(z_g,\{z_l\},\{z_m\},\{q_m\},S\right),
\]

where:

- \(z_g\) is a global task/model code;
- \(z_l\) is a layer code;
- \(z_m\) is a tensor code;
- \(q_m\) selects or parameterizes structured decoder primitives;
- \(S\) is an optional exception patch.

A tensor decoder may operate by coordinates:

\[
\widehat \Delta_m[i,j]
=
D_\psi\!\left(
 z_g,z_{l(m)},z_m,
 \operatorname{role}_m,
 \operatorname{shape}_m,
 \bar i,\bar j,
 W_{0,m}[i,j]
\right),
\]

where \(\bar i,\bar j\) are normalized coordinates.

A block decoder is often more practical:

\[
\widehat \Delta_m[B_{u,v}]
=
D_\psi\!\left(z_g,z_{l(m)},z_m,e_{u,v},\operatorname{role}_m\right),
\]

with \(B_{u,v}\) a fixed block such as \(16\times16\) or \(32\times32\).

The generator attends over hundreds of tensor/block codes, not tens of millions of scalar weights.

---

## 7. Structured tensor formula

A useful hybrid decoder can express each matrix delta as:

\[
\widehat\Delta_m
=
\alpha_m\left(
L_m+K_m+F_m+N_m
\right)+S_m,
\]

where:

### Low-rank term

\[
L_m=U_m\operatorname{diag}(a_m)V_m^\top,
\qquad \operatorname{rank}(L_m)\le r_m.
\]

### Kronecker term

\[
K_m=\sum_{q=1}^{Q_m} c_{m,q}\left(A_{m,q}\otimes B_{m,q}\right).
\]

### Spectral term

\[
F_m=\mathcal T_m^{-1}\!\left(M_m\odot C_m\right),
\]

where \(\mathcal T_m\) may be a DCT, FFT, wavelet, or learned orthogonal transform, \(M_m\) selects modes, and \(C_m\) stores coefficients.

### Neural residual field

\[
N_m[i,j]=D_\psi(z_m,\bar i,\bar j,\operatorname{role}_m).
\]

### Sparse or structured exception

\[
S_m=\operatorname{Scatter}(I_m,V_m)
\quad\text{or}\quad
S_m=A_mB_m^\top.
\]

The interpreter need not enable every term for every tensor. The genome selects a compact mixture based on rate–distortion value.

For vectors such as normalization scales and biases, use one-dimensional analogues: spline/spectral coefficients, shared codebooks, low-order polynomial fields, and sparse exceptions.

---

## 8. Dataset fingerprints: a non-linguistic task description

A new target corpus contains information that cannot be inferred from architecture alone. GENOME therefore requires a compact model-native observation of the data.

At initialization or an early checkpoint, sample probe batches \(B_b\) and calculate per-tensor gradients:

\[
g_{b,m}=\nabla_{W_m}\ell(B_b;W).
\]

Project them with fixed random matrices or structured sketches:

\[
\widetilde g_{b,m}=R_m^\top\operatorname{vec}(g_{b,m})\in\mathbb R^k,
\qquad k\ll \operatorname{numel}(W_m).
\]

Collect inexpensive statistics:

\[
q_{b,m}=
\left[
\widetilde g_{b,m},
\|g_{b,m}\|_2,
\|g_{b,m}\|_\infty,
\operatorname{mean}(g_{b,m}),
\operatorname{var}(g_{b,m}),
\operatorname{spec}_r(G_{b,m}),
\operatorname{actstats}_{b,m}
\right].
\]

Aggregate the unordered batch evidence with a set or attention encoder:

\[
\Phi(D)=\rho\!\left(\{\phi(q_b)\}_{b=1}^{n}\right).
\]

Additional language-model-specific fingerprint channels may include:

- token and byte frequency sketches;
- n-gram Count-Min sketches;
- sequence-length distribution;
- conditional-mode proportions;
- gradient covariance spectra by tensor role;
- activation mean/variance/quantiles;
- attention entropy and residual-stream norms;
- loss distribution and per-token loss histograms;
- a tiny set of learned synthetic probe sequences.

The compiler does not need a prose explanation of the corpus. It needs evidence of how the target architecture responds to the corpus.

---

## 9. Early trajectory as a loss-landscape query

A short normal-training prefix may contain more information than a static fingerprint. Represent it as:

\[
\tau_{0:k}=\left\{
E_W(W_{t_i}-W_0),
E_g(g_{t_i}),
L_{t_i},
A_{t_i},
M_{t_i}
\right\}_{i=0}^{k},
\]

where:

- \(E_W\) is a weight-space encoder;
- \(E_g\) is a gradient sketch encoder;
- \(A_{t_i}\) contains activation summaries;
- \(M_{t_i}\) contains optimizer/state summaries.

The central measurable question is:

> How early can the trajectory be truncated while retaining enough information to predict a useful endpoint genome?

Test prefixes by tokens and compute, not only step count. Suggested first fractions are 0%, 0.1%, 0.25%, 0.5%, 1%, 2%, 5%, 10%, and 20% of the original training budget.

---

## 10. Global planning instead of strict layer autoregression

A final early layer and a final late layer are jointly coupled through end-to-end training. A strict rule of the form

\[
z_l=F(z_{<l})
\]

imposes a causal order that the endpoint itself does not possess.

The default planner should use bidirectional attention or message passing over the complete architecture graph:

\[
Z^{(0)}=\operatorname{Embed}(\mathcal G,\Phi(D),\tau),
\]

\[
Z^{(h+1)}=\operatorname{GraphBlock}_h(Z^{(h)},E_\mathcal G),
\]

\[
p=\operatorname{GenomeHeads}(Z^{(H)}).
\]

Layer-autoregressive generation remains an optional branch for memory efficiency, but it should include backward/global passes or iterative correction.

---

## 11. Compile–probe–link–patch

One-shot generation is unnecessarily restrictive.

### Compile

\[
p^{(0)}=C_\phi(\Phi(D),\mathcal G,s,r,\tau).
\]

### Decode

\[
\widehat W^{(j)}=\mathcal I_\psi(p^{(j)},W_0,\mathcal G).
\]

### Probe

Run on a small allowed probe set \(P\) and collect:

\[
E^{(j)}=\operatorname{Probe}(\widehat W^{(j)},P).
\]

Evidence includes loss, logit residuals, gradient sketches, activation distributions, layer sensitivities, saturation, attention entropy, and output calibration.

### Link

Either use a learned linker:

\[
p^{(j+1)}=p^{(j)}+\mathcal L_\omega(p^{(j)},E^{(j)}),
\]

or directly optimize the low-dimensional genome:

\[
p^{(j+1)}=p^{(j)}-\eta\nabla_p L_P(\mathcal I_\psi(p^{(j)})).
\]

Direct latent optimization should be implemented first. It establishes whether the representation is navigable before the project pays for a learned linker.

### Patch

Allocate remaining bits or compute to the most functionally sensitive residual:

\[
S^*=\arg\min_S
L_P(\widehat W+S)+\beta B(S).
\]

The result is a compiler pipeline, not necessarily a single forward pass.

---

## 12. Training objective

Weight MSE is useful as a diagnostic, not as the sole objective. A full objective is:

\[
\begin{aligned}
\mathcal L_{
\text{GENOME}}={}&
\lambda_w\mathcal L_{\Delta}
+\lambda_t\mathcal L_{\text{task}}
+\lambda_k\mathcal L_{\text{logit}}
+\lambda_h\mathcal L_{\text{hidden}}\\
&+\lambda_g\mathcal L_{\text{grad}}
+\beta R(p,S,\psi)
+\eta C_{\text{repair}}.
\end{aligned}
\]

### Delta reconstruction

Normalize per tensor so large tensors do not completely dominate:

\[
\mathcal L_{\Delta}
=
\frac{1}{M}\sum_{m=1}^M
\frac{\|\widehat\Delta_m-\Delta_m\|_F^2}
{\|\Delta_m\|_F^2+\epsilon}.
\]

### Task loss

\[
\mathcal L_{\text{task}}=L_{\text{NTP}}(D_{\text{fit}};\widehat W).
\]

### Logit imitation

For reference probabilities \(p_T\) and candidate probabilities \(p_G\):

\[
\mathcal L_{\text{logit}}
=
\frac{1}{|A|}\sum_{x\in A}
D_{\mathrm{KL}}(p_T(\cdot|x)\|p_G(\cdot|x)).
\]

### Hidden-state agreement

Use normalized residual-stream cosine, CKA, or covariance agreement rather than raw hidden MSE if bases are not guaranteed aligned.

### Gradient residual

A good candidate should not face a large obvious downhill direction:

\[
\mathcal L_{\text{grad}}
=
\sum_m
\left\|R_m^\top\operatorname{vec}\left(\nabla_{W_m}L_P(\widehat W)\right)\right\|_2^2.
\]

### Rate

\[
R=B(p)+B(S)+B(\text{metadata})+\frac{B(\psi)}{N_{\text{amort}}}.
\]

Always also report the non-amortized rate:

\[
R_{1}=B(p)+B(S)+B(\text{metadata})+B(\psi).
\]

### Repair cost

\[
C_{\text{repair}}=
\text{repair tokens, FLOPs, seconds, and optimizer steps required to pass the gate}.
\]

Do not collapse all metrics into one score in reports. The weighted objective is for optimization; the evaluation table must expose every component.

---

## 13. Distributional generation

There may be many valid endpoints. A deterministic regressor trained against incompatible endpoints can average them into a bad checkpoint. Model a distribution:

\[
p_\phi(p\mid \Phi(D),\mathcal G,s,r,\tau).
\]

Generate \(n\) candidates, but include all candidate generation and verification cost. Selection must use the allowed probe set, never the hidden verifier.

Possible generative mechanisms, in increasing complexity:

1. deterministic regression plus learned variance;
2. mixture-density genome heads;
3. latent-variable VAE;
4. flow matching in genome space;
5. diffusion in genome space;
6. autoregressive discrete MGP tokens.

Start with deterministic or small-mixture outputs. A rich generative model is justified only after the genome representation itself works.

---

## 14. Exact residual coding

For literal recovery, decode in the target dtype and form an exact residual. One option is bitwise XOR:

\[
R_{\text{xor}}
=
\operatorname{bitcast}(W_T)
\oplus
\operatorname{bitcast}(\widehat W).
\]

Entropy-code the residual and report:

\[
B_{\text{exact}}
=B(p)+B(\psi)+B(R_{\text{xor}})+B(\text{metadata}).
\]

Also test arithmetic residuals in higher precision:

\[
R_{\text{arith}}=W_T-\widehat W,
\]

followed by lossless coding of the exact dtype representation. XOR and arithmetic residuals reveal different kinds of structure.

Exact recovery is a compression diagnostic. A functionally equivalent result with an incompressible fp32 residual can still be the more important scientific result.

---

## 15. Rate–distortion view

For budget \(B\), define the best achievable functional distortion:

\[
D^*(B)=\min_{p,S:\,R(p,S)\le B}D_f(\mathcal I(p)+S,W_T).
\]

Plot at minimum:

- normalized validation-loss gap versus total bytes;
- anchor-logit KL versus total bytes;
- repair compute versus total bytes;
- decode time versus total bytes;
- parameter error by tensor family versus total bytes.

The important object is the complete Pareto frontier, not one hand-selected checkpoint.

---

## 16. Information leakage and conditional description

The genome may condition on legitimate shared information:

- architecture code;
- tokenizer and vocabulary;
- initialization seed or base checkpoint;
- training recipe;
- dataset fingerprint;
- early trajectory prefix;
- shared interpreter learned across targets.

It may not condition on hidden target information at inference:

- WT or a transform that can trivially recover WT;
- final anchor logits if endpoint prediction is the claim;
- target-specific decoder weights not counted as genome bits;
- a cache keyed by target ID that stores target tensors;
- verifier results used for candidate selection.

For every result, state the conditioning set explicitly. The scientific claim is always conditional on what the compiler was allowed to observe.

---

## 17. What a negative result would mean

A failed compact representation does not prove direct model generation is impossible. It distinguishes possibilities:

1. R0 may have high conditional description length under the tested decoder family.
2. The useful structure may be functional but not scalar-coordinate structure.
3. Embeddings or output heads may dominate the irreducible information.
4. A conventional dense architecture may not be genome-friendly.
5. The shared decoder may need a population of models before structure becomes visible.
6. The endpoint may be representable, but the dataset fingerprint may be insufficient to infer it.
7. A short training prefix may be necessary.
8. The best system may be compile-and-polish rather than zero-step generation.

Each case leads to a different next experiment. Do not treat all failures as one result.
