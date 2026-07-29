# GENOME active theory and mathematics

This file uses the recovery design: one learned compiler and one deterministic Runtime.

## 1. Conditional endpoint compilation

Ordinary training maps a full model-life specification to an endpoint:

\[
W_T=\mathcal A(D,\mathcal G,W_0,r,e).
\]

GENOME seeks a shorter conditional program:

\[
p=C_\phi(\Phi(D,W_0),\mathcal G,W_0,r),
\]

\[
\widehat W=\operatorname{Runtime}(p,W_0,\mathcal G).
\]

The claim is useful only when the complete target-specific program is small and the decoded model
passes functional tests.

## 2. Determinism does not prove compressibility

A deterministic training run can still have a long endpoint description. The tested condition is:

\[
K_\epsilon([W_T]\mid W_0,\mathcal G,\Phi(D,W_0),r,\operatorname{Runtime})
\ll |W_T|.
\]

The project must measure this condition. It must not hide endpoint bytes in a learned decoder,
dense residual, shared asset, patch, index, or scale.

## 3. Predict Delta-T from true W0

\[
W_T=W_0+\Delta_T.
\]

W0 contains random initialization entropy. The MGP describes a compact update relative to the
exact W0 of the same life. Cross-run coordinates are not assumed to be aligned.

## 4. Functional equivalence

Parameter error is not the final objective. A candidate can succeed when:

\[
L_D(\widehat W)\le L_D(W_T)+\epsilon_L
\]

and its logit, task, and generation metrics pass the declared gate.

For hidden transfer, report:

\[
\frac{L(W_0)-L(\widehat W)}{L(W_0)-L(W_T)}.
\]

The first hidden result must be better than W0.

## 5. Deterministic primitive language

For a matrix update, the Runtime can combine transparent terms:

\[
\widehat\Delta_m =
L_m + K_m + F_m + B_m + P_m.
\]

Examples are:

\[
L_m=U_m\operatorname{diag}(s_m)V_m^\top
\]

for low rank,

\[
K_m=\sum_q c_q(A_q\otimes B_q)
\]

for Kronecker terms, and

\[
F_m=\operatorname{DCT}^{-1}(I_m,c_m)
\]

for sparse spectral modes.

Shared bases and codebooks are permitted only when their bytes are counted. Sparse and low-rank
patches must remain bounded. A dense or exact residual is not permitted for compiler supervision.

## 6. Canonical identity and SVD ambiguity

Target labels must have stable serialized identity:

```text
same W0 + WT + config + code -> same target bytes and hash
```

Wall-clock metadata is outside the canonical label identity.

A sign rule removes the simple SVD sign ambiguity. It does not define a unique basis inside a
repeated or nearly repeated singular-value subspace. Production coefficient training must
therefore use deterministic Runtime or function loss, or a stronger verified subspace rule. The
current sign-only fitter is a transparent baseline. It is not a universal canonical label.

Cross-hardware SVD kernels can also select different valid bases or rounding. A target corpus must
record its hardware and numerical environment.

## 7. Semantic evidence

Compiler evidence comes from corpus content and the response of W0. It can contain:

- token and byte frequency sketches;
- n-gram sketches;
- sequence-length and supervision statistics;
- W0 losses;
- per-role W0 gradient sketches and moments;
- W0 activation moments and quantiles;
- explicit numeric recipe values.

SHA-256 values, repository revisions, file paths, WT, fitted programs, and later hidden
checkpoints do not enter the semantic tensor order.

## 8. Rate and serialized bytes

For one candidate:

\[
R=B(p)+B(\text{manifest})+B(\text{containers})+B(\text{patches})
+B(\text{indices})+B(\text{scales})+B(\text{shared assets}).
\]

Report shared and target-specific bytes separately. Audit the real serialized files. An in-memory
logical byte count cannot approve training supervision.

A rate term in compiler training must depend on the prediction. Teacher target length is constant
and has zero gradient. The current compiler has no misleading teacher-length rate penalty.

## 9. Program sequence scalability

The current flat tokenizer emits one autoregressive numeric token for each small coefficient
chunk. With chunk width 16, a 5% fp16-Delta payload is already about 44,268 tokens for Pythia 14M
and 95,602 tokens for Pythia 31M. The 10% cases are about 88,230 and 190,898 tokens.

These values exceed `max_program_tokens=4096`. Increasing the flat context to these sizes is not
the solution. Production work needs a hierarchical form:

- an autoregressive skeleton for primitives and discrete arguments;
- bounded per-tensor coefficient packets;
- shared primitive-specific coefficient heads;
- no silent coefficient truncation.

## 10. Target acceptance

A fitted program is a target candidate only. It becomes compiler supervision after all these
checks:

1. deterministic serialization;
2. post-serialization byte audit;
3. deterministic Runtime decode;
4. real-model load;
5. finite execution;
6. improvement over W0;
7. predeclared functional quality;
8. no hidden endpoint leakage.

If a life fails, label it not representable by the current grammar. Do not add a dense residual.
