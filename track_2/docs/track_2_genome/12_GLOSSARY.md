# GENOME glossary and symbol reference

## Project terms

| Term | Definition |
| --- | --- |
| **GENOME** | Generative Endpoint Neural Operator for Model Emission; Track 2 project. |
| **R0** | Immutable fully trained Track 1 poetry reference model. |
| **W0** | R0 initialization or declared base checkpoint. |
| **WT** | R0 final trained checkpoint. |
| **Delta-T** | Learned displacement \(W_T-W_0\). |
| **Model genome** | Compact target-specific program/codes that expand into a model. |
| **MGP** | Model Genome Program, the serialized genome format. |
| **Phenotype** | Expanded runnable model. |
| **Compiler** | Model mapping task/architecture/trajectory evidence to a genome. |
| **Interpreter** | Shared decoder/runtime mapping a genome and base state to weights. |
| **Auto-decoder** | Shared interpreter trained with free per-target latent codes; proves representation, not prediction. |
| **Linker** | Iterative genome-code correction component using probe evidence. |
| **Patch** | Sparse, low-rank, or exact exception correction. |
| **Genome Gate** | Hidden immutable evaluator. |
| **Model life** | Architecture, initialization, data, trajectory, metrics, and endpoint for one training run. |
| **Dataset fingerprint** | Compact model-native description of data/task based on statistics, gradients, activations, or learned set encoding. |
| **Trajectory prefix** | Early part of ordinary training supplied to the compiler. |
| **Conditional description length** | Genome length given declared shared information such as W0, architecture, and interpreter. |
| **Function-space target** | Behavioural target such as logits/hidden states rather than exact scalar weights. |
| **Canonicalization/re-basing** | Function-preserving transformation intended to align equivalent weight coordinates. |
| **Genome-native architecture** | Child model whose apparent weights are generated from compact codes by construction. |
| **Compile-and-polish** | Generate most of an endpoint, then perform a small latent/full-weight repair phase. |
| **Self-hosting** | GENOME represents/generates the compiler itself. |

## Research levels

| Label | Meaning |
| --- | --- |
| **G0** | Full endpoint visible; test compact representation. |
| **G1-Pure** | Same known lineage; compile from permitted early/task evidence without final outputs at inference. |
| **G1-Distill** | Same known lineage; final reference behaviour on declared probes is available at inference. |
| **G2** | Endpoint of an independent run is withheld. |
| **G3** | Transfer across material task/architecture change. |
| **G4** | Parent-generated child compiler improves hidden child-generation performance. |

## Core symbols

| Symbol | Meaning |
| --- | --- |
| \(D\) | Training dataset/corpus, including exact order when relevant. |
| \(\mathcal G\) | Architecture graph. |
| \(s\) | Initialization/random state. |
| \(r\) | Training recipe: optimizer, schedule, objective, precision, etc. |
| \(e\) | Execution environment/numerical implementation details. |
| \(\mathcal A\) | Ordinary training algorithm/program. |
| \(W_t\) | Model weights at progress/time \(t\). |
| \(W_0\) | Initial/base weights. |
| \(W_T\) | final reference weights. |
| \(\Delta_T\) | \(W_T-W_0\). |
| \(\widehat W\) | GENOME-generated candidate weights. |
| \(p\) | Model genome/program or its code collection. |
| \(C_\phi\) | Genome compiler with parameters \(\phi\). |
| \(\mathcal I_\psi\) | Genome interpreter with shared parameters \(\psi\). |
| \(\Phi(D)\) | Dataset fingerprint. |
| \(\tau_{0:k}\) | Early trajectory evidence. |
| \(A\) | Anchor examples. |
| \(P\) | Allowed probe set. |
| \([W]\) | Equivalence class of weights implementing the same/similar function. |
| \(K_\epsilon\) | Approximate conditional description length. |
| \(B(x)\) | Encoded bit/byte cost of object \(x\). |
| \(D_f\) | Functional distortion. |
| \(C_{repair}\) | Compute required to repair a candidate. |
| \(z_g,z_l,z_m\) | Global, layer, and tensor genome codes. |
| \(S\) | Sparse/low-rank exception patch. |

## Metrics

| Metric | Meaning |
| --- | --- |
| **NRMSE-W** | normalized error in complete weights. |
| **NRMSE-Delta** | normalized error in learned displacement. |
| **Anchor KL** | KL divergence between R0 and candidate token distributions on fixed anchors. |
| **Top-k agreement** | overlap/agreement of most likely tokens. |
| **Terminal loss z-score** | candidate loss gap measured in units of R0 final-window variability. |
| **Payload ratio** | target-specific MGP bytes divided by raw WT bytes. |
| **Single-model ratio** | payload plus complete shared decoder/base cost divided by WT bytes. |
| **Amortized ratio** | payload plus shared cost divided across N generated models. |
| **Repair-to-quality** | tokens/FLOPs/time required to enter the frozen quality band. |
| **Rate–distortion curve** | functional error as a function of total description size. |

## Formula components

| Component | Meaning |
| --- | --- |
| **Low rank** | \(U\Sigma V^\top\) approximation. |
| **Kronecker sum** | sum of \(A\otimes B\) structured factors. |
| **Tensor train** | high-order tensor decomposition into compact cores. |
| **Spectral code** | selected transform coefficients, such as DCT/FFT. |
| **Block codebook** | each matrix block represented by a shared dictionary index. |
| **Coordinate field** | decoder maps tensor coordinates plus latent codes to scalar weights. |
| **Block field** | decoder maps block coordinates plus latent codes to a weight block. |
| **Sparse patch** | explicit corrections at selected coordinates/blocks. |
| **Low-rank patch** | small residual \(AB^\top\). |

## Claim language

Use these phrases accurately:

- **Endpoint representation:** target was visible while fitting the genome.
- **Endpoint distillation:** target behaviour or weights supervised the system.
- **Same-run compilation:** target lineage is known; endpoint unavailable at candidate inference.
- **Held-out-run compilation:** target endpoint and later trajectory are withheld by run.
- **Functional equivalence:** behaviour passes declared metrics; weights need not match.
- **Exact reconstruction:** output dtype bits match WT.
- **Acceleration:** total child-generation cost to matched quality is lower than baseline.
- **Compression:** complete declared description is smaller, with shared costs reported.
- **RSI:** generated child compiler improves model-generation ability under hidden external evaluation.
