# GENOME theory and mathematics

## 1. Problem

A fully specified training process defines an endpoint:

\[
W_T=\mathcal A(D,\mathcal G,W_0,r,e),
\]

where:

- \(D\) is the dataset and order;
- \(\mathcal G\) is the architecture graph;
- \(W_0\) is the true random initialization;
- \(r\) is the complete staged training recipe;
- \(e\) is the numerical execution environment.

GENOME learns an amortized compiler:

\[
p=C_\phi\!\left(\Phi(D,W_0),\mathcal G,W_0,r\right),
\]

and a deterministic Runtime executes:

\[
\widehat W=\operatorname{Runtime}(p,W_0,\mathcal G).
\]

The hidden target \(W_T\) is absent from compiler input.

The question is not whether a deterministic mapping exists. Ordinary training already supplies one. The question is whether the useful endpoint has a substantially shorter conditional description than replaying training:

\[
K_\epsilon([W_T]\mid W_0,\mathcal G,\Phi(D,W_0),r,\operatorname{Runtime})
\ll |W_T|.
\]

The equivalence class \([W_T]\) matters because a functionally good endpoint need not use the exact hidden-unit coordinates produced by the reference optimizer.

## 2. Learned displacement

GENOME predicts a program for the learned displacement:

\[
\Delta_T=W_T-W_0,
\qquad
\widehat W=W_0+\widehat\Delta.
\]

W0 is provided directly. The compiler does not waste capacity reproducing pseudorandom initialization values.

Each model life keeps its own W0 coordinate system. Raw deltas are never averaged across independent seeds as though neuron indices were universal.

## 3. Model Genome Program

A v1 tensor program is:

\[
\widehat\Delta_m=H_m+L_m+V_m+S_m,
\]

where the permitted terms are:

### Low-rank matrix

\[
L_m=U_mV_m^\top,
\qquad
U_m\in\mathbb R^{d_o\times r},
\quad
V_m\in\mathbb R^{d_i\times r}.
\]

Its fp16 payload cost is approximately:

\[
B_{LR}=2r(d_o+d_i)\text{ bytes}.
\]

It is rejected when its payload is not smaller than the dense matrix.

### Base-relative row/column scaling

For a matrix with base value \(W_{0,m}\):

\[
H_m=W_{0,m}\odot(a_m\mathbf 1^\top+\mathbf 1b_m^\top).
\]

This stores one fp16 value per row and column. Its payload cost is:

\[
B_H=2(d_o+d_i)\text{ bytes}.
\]

It is not a one-value-per-weight representation. The Runtime expands it
deterministically from W0.

### Shared vocabulary factor

When an embedding matrix and language-model head use the same vocabulary row
coordinate, they may share one left factor:

\[
L_{\text{in}}=A_{\text{vocab}}B_{\text{in}}^\top,\qquad
L_{\text{out}}=A_{\text{vocab}}B_{\text{out}}^\top.
\]

The shared payload is serialized once. Each tensor keeps its own right factor.
This sharing is allowed only when the architecture graph proves that the row
coordinates are compatible.

### Quantized vector

For one-dimensional normalization and bias tensors:

\[
V_m=s_m q_m,
\qquad q_m\in\{-127,\ldots,127\}^{d}.
\]

Direct vector storage is bounded. It is never permitted for a matrix.

A bounded vector may instead store its values directly in fp16 when every
vector value must remain trainable during functional refinement. The same
4,096-value limit applies. Direct vector storage is never permitted for a
matrix.

### Sparse patch

\[
S_m=\operatorname{Scatter}(I_m,a_m),
\]

with a global cap of 0.1% of model values. It is an exception table, not a dense escape hatch.

### Tied tensor

A tied alias copies its declared owner. It does not duplicate payload bytes.

The schema has no dense, full residual, exact residual, or neural decoder primitive.

## 4. Rate gate

Let:

\[
B_{\Delta16}=2\sum_m \operatorname{numel}(W_m)
\]

be the direct fp16 Delta-T baseline. The primary target gate is:

\[
\frac{B_{MGP}}{B_{\Delta16}}\leq0.10.
\]

\(B_{MGP}\) is measured from the serialized manifest and Safetensors payload. It includes factor values, vector values, scales, indices and container overhead. Shared assets are counted separately and reported both unamortized and amortized.

A differentiable compiler rate proxy must depend on predicted primitive probabilities and predicted ranks:

\[
\widehat B_\phi=\sum_m
P_\phi(\text{low-rank}\mid m)
\,2(d_o+d_i)\,\mathbb E_\phi[r_m]
+
P_\phi(\text{vector}\mid m)(d_m+4).
\]

A constant teacher sequence length is not a rate loss.

## 5. Functional gate

Model quality is the primary distortion measure.

For validation loss \(L\), endpoint progress is:

\[
P=\frac{L(W_0)-L(\widehat W)}{L(W_0)-L(W_T)}.
\]

Interpretation:

- \(P\le0\): no improvement over W0;
- \(0<P<1\): part of the endpoint gap closed;
- \(P=1\): reference endpoint loss matched;
- \(P>1\): reference endpoint loss exceeded on that evaluation.

A development program is production supervision only when:

\[
P\ge0.80,
\qquad
B_{MGP}/B_{\Delta16}\le0.10,
\]

and the model loads, produces finite logits, and passes the fixed evaluation suite.

Weight error remains diagnostic:

\[
D_W=\frac1M\sum_m
\frac{\|\widehat\Delta_m-\Delta_m\|_F^2}
{\|\Delta_m\|_F^2+\epsilon}.
\]

It does not decide acceptance.

## 6. Fitting known target programs

Training and development WT may be used offline to fit compact programs. That fitting creates labels; it is not hidden inference.

The first candidate is globally budgeted truncated SVD. Singular components are ranked by energy per encoded byte. Small vectors are quantized. No residual is added.

The compact parameters can then be optimized through the Runtime and real model:

\[
\min_p
L_{task}(\operatorname{Runtime}(p,W_0))
+\lambda_{KL}D_{KL}(f_{W_T}\Vert f_{\widehat W})
+\lambda_A\frac{\|p-p_{\mathrm{fit}}\|_2^2}
{\|p_{\mathrm{fit}}\|_2^2+\epsilon}
+\beta B(p).
\]

Only program coefficients are trainable. Child weights are materialized outputs, not free optimization variables.

This function-through-Runtime objective also avoids requiring one arbitrary SVD factorization to be a unique label.

## 7. Compiler representation

A flat program token stream is unsuitable because even a 5% Pythia target can contain tens of thousands of coefficient chunks.

GENOME therefore uses one hierarchical compiler:

1. Construct one token per logical tensor plus one global task token.
2. Apply graph message passing over tensor connectivity.
3. Apply bidirectional attention over all tensors.
4. Predict primitive and rank distributions.
5. Generate bounded coefficient packets with shared coordinate-conditioned heads.
6. Enforce the byte budget while allocating ranks.
7. Serialize the resulting deterministic MGP.

For a matrix with scale flag \(h\in\{0,1\}\), the output count is:

\[
(r+h)(d_o+d_i),
\]

rather than \(d_od_i\). No fixed block output table exists.

## 8. Compiler objective

For accepted target programs, teacher-forced compiler training uses:

\[
\begin{aligned}
\mathcal L={}&
\lambda_s\mathcal L_{primitive}
+\lambda_r\mathcal L_{rank}
+\lambda_\Delta\mathcal L_{decoded\;delta}\\
&+\lambda_f\mathcal L_{task\;through\;Runtime}
+\beta\widehat B_\phi.
\end{aligned}
\]

Decoded-delta loss compares the function represented by factors, not individual U/V labels. Functional loss periodically executes the predicted compact displacement in the real Pythia architecture.

Development lives select the compiler checkpoint. Hidden WT is not available.

## 9. Dataset and W0 evidence

The compiler cannot infer a corpus from a filename or SHA digest. Its semantic input contains:

\[
\Phi(D,W_0)=
[
\text{token sketches},
\text{bigram sketches},
\text{byte frequencies},
\text{length distribution},
\text{W0 loss},
\text{gradient sketches},
\text{activation moments},
\text{recipe}
].
\]

Gradient CountSketches are calculated per tensor role at W0. Activation summaries include per-layer mean, variance and quantiles. Source hashes remain in receipts only.

## 10. Hidden protocol

For the fresh hidden life:

1. W0, architecture, semantic evidence and recipe are available.
2. WT and later checkpoints remain unresolved and absent.
3. The frozen compiler emits the predeclared number of candidates.
4. Selection uses only the declared endpoint-free rule.
5. The MGP, compiler hash, evidence ID and Runtime state are sealed.
6. Only then is hidden WT resolved and downloaded in a separate revealed directory.
7. W0, candidate and WT are evaluated identically.
8. One-shot results are published before repair.

The first hidden criterion is simply \(P>0\). Values \(P\ge0.25\) and \(P\ge0.50\) are reported as useful and strong transfer, but do not replace the raw metrics.

## 11. Scope

Pythia 14M and 31M are the only initial families. Cross-size, dataset, recipe and architecture transfer are deferred until fresh hidden 31M seed9 improves over W0.
