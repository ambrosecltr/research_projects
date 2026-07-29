# GENOME research references

> **Pre-recovery reference list.** This list is background only. It does not define the active
> architecture or experiment order.

**Checked through:** 27 July 2026

This list prioritizes primary papers and official project pages. A cited result is evidence only for the setting actually studied. Vision, small-network, adapter, implicit-neural-representation, and controlled-program results are useful mechanism evidence, not proof that a 50M poetry language-model endpoint can already be generated.

---

# A. Direct weight-generation and checkpoint-modeling work

## Position: Weight Space Should Be a First-Class Generative AI Modality

- **Wang, Wang, Wang (2026)**
- <https://arxiv.org/abs/2605.18632>

Argues that checkpoints should be treated as a first-class generative modality and organizes the field around a weight-generation pipeline. It explicitly distinguishes rapidly advancing adapter/conditional generation from still-open unrestricted frontier-scale checkpoint synthesis.

**GENOME relevance:** strongest current framing for treating model weights as data and for AI systems creating other AI systems.

**Caveat:** a position/survey paper is not experimental proof of GENOME’s target.

---

## DeepWeightFlow: Re-Basined Flow Matching for Generating Neural Network Weights

- **Gupta et al. (2026)**
- <https://arxiv.org/abs/2601.05052>

Uses flow matching over canonicalized/re-based weight spaces to generate complete high-performing networks across its tested architectures and modalities. It emphasizes permutation symmetry and efficient generation.

**GENOME relevance:** supports complete-weight generation, canonicalization, and flow models as a later genome-space branch.

**Caveat:** tested architectures/settings are not the Track 1 50M decoder-only poetry pretraining endpoint.

---

## Structure-Aware Graph Hypernetworks for Neural Program Synthesis

- **Li, Xu, Khalil, Sanner (ICLR 2026)**
- <https://openreview.net/forum?id=x7zOzUwtR7>

Builds a structure-aware graph hypernetwork that generates full target-network weights while respecting neuron-permutation structure. In controlled program families, it improves unseen-parameter generalization; its AddMod analysis finds a compact closed-form mapping into Transformer weights.

**GENOME relevance:** directly supports the view that weights are a continuous program modality and that target architecture structure should be part of the compiler.

**Caveat:** controlled program synthesis is much simpler than language-model pretraining.

---

## NNiT: Width-Agnostic Neural Network Generation with Structurally Aligned Weight Spaces

- **Kim et al. (2026)**
- <https://arxiv.org/abs/2603.00180>

Tokenizes weight matrices into local patches and jointly models architecture tokens and continuous weight patches. Uses Graph HyperNetworks to create structural alignment and demonstrates generation across unseen MLP topologies in its tasks.

**GENOME relevance:** supports patch tokenization, architecture-plus-weight sequences, and width-agnostic generation.

**Caveat:** focuses on MLP-centered settings rather than decoder-only language models.

---

## Learning to Learn with Generative Models of Neural Network Checkpoints — G.pt

- **Peebles et al. (2022)**
- <https://arxiv.org/abs/2209.12892>
- Official code/project: <https://github.com/wpeebles/G.pt>

Trains a conditional diffusion Transformer directly on a very large checkpoint corpus. Given starting parameters and a desired metric, G.pt generates updated parameters and demonstrates one-update optimization of unseen initializations in small vision/RL settings.

**GENOME relevance:** direct evidence that checkpoint distributions can support learned optimization and that intermediate checkpoints add useful training signal.

**Caveat:** G.pt required millions of checkpoints from many small runs and did not generate a pretrained language model endpoint.

---

## Parameter Prediction for Unseen Deep Architectures — Graph HyperNetworks / GHN-2

- **Knyazev et al. (2021)**
- <https://arxiv.org/abs/2110.13100>

Predicts parameters for unseen architecture graphs in one forward pass and demonstrates large complete parameter generation in vision settings.

**GENOME relevance:** architecture graph encoder, shared tensor decoders, and single-pass parameter prediction.

**Caveat:** generated models were not equivalent to fully optimized frontier endpoints, and the domain was vision architecture prediction.

---

## Graph HyperNetworks for Neural Architecture Search

- **Zhang, Ren, Urtasun (2018)**
- <https://arxiv.org/abs/1810.05749>

Introduces graph hypernetworks that map architecture graphs to weights and performance signals.

**GENOME relevance:** foundational architecture-as-graph formulation.

---

## HyperNetworks

- **Ha, Dai, Le (2016)**
- <https://arxiv.org/abs/1609.09106>

Introduces networks that generate the weights of other networks and explicitly uses the genotype/phenotype analogy.

**GENOME relevance:** foundational hypernetwork and model-genome concept.

---

## Neural Metamorphosis

- **Yang, Wang (2024)**
- <https://arxiv.org/abs/2410.11878>

Learns coordinate-based implicit functions that generate weights for differently sized neural networks from a continuous weight manifold.

**GENOME relevance:** coordinate decoders, model-space coordinates, and size-varying weight generation.

**Caveat:** demonstrations are in vision/image-generation settings.

---

## HyperNet Fields: Efficiently Training Hypernetworks without Ground Truth by Learning Weight Trajectories

- **Hedlin et al. (2024)**
- <https://arxiv.org/abs/2412.17040>

Models the entire convergence trajectory as a hypernetwork field and constrains predicted states through task-gradient consistency, reducing dependence on separately optimized endpoint weights.

**GENOME relevance:** time-as-coordinate model life, gradient residual objective, and endpoint completion from trajectories.

---

## Towards Scalable and Versatile Weight Space Learning — SANE

- **Schürholt, Mahoney, Borth (ICML 2024)**
- <https://arxiv.org/abs/2406.09997>

Processes sequential subsets of network weights as tokens, learns task-agnostic weight representations, and can sequentially generate models.

**GENOME relevance:** scalable patch/subset tokenization rather than all-scalar contexts; basis for a literal weight language model.

---

## HyperTinyPW: Once-for-All Channel Mixers / Generative Compression for TinyML

- **Shaalan (MLSys 2026)**
- <https://openreview.net/forum?id=NrDa5Fu10D>

Uses a shared micro-generator plus tiny per-layer codes to synthesize large pointwise-convolution weights once at load time, with careful total-byte accounting.

**GENOME relevance:** generate-and-cache deployment, shared decoder plus per-layer codes, and honest packed-byte accounting.

**Caveat:** TinyML convolutional mixers are far smaller and more regular than a language-model checkpoint.

---

# B. Weight-space symmetry, alignment, and representations

## Equivariant Neural Functional Networks for Transformers

- **Tran et al. (2024)**
- <https://arxiv.org/abs/2410.04209>

Derives transformer weight-space symmetries, designs an equivariant Transformer-NFN, and releases a large transformer-checkpoint benchmark.

**GENOME relevance:** the most directly relevant reference for processing Transformer weights without pretending head/neuron ordering is universal.

---

## Permutation Equivariant Neural Functionals

- **Zhou et al. (2023)**
- <https://arxiv.org/abs/2302.14040>

Develops networks that process the weights/gradients of other networks while encoding hidden-unit permutation symmetry.

**GENOME relevance:** neural functional architecture and symmetry-aware learned operators.

---

## Git Re-Basin: Merging Models modulo Permutation Symmetries

- **Ainsworth, Hayase, Srinivasa (2022)**
- <https://arxiv.org/abs/2209.04836>

Aligns hidden units by permutations before merging independently trained networks and demonstrates zero-barrier connectivity in studied vision settings.

**GENOME relevance:** practical canonicalization/alignment baseline and reminder that raw cross-run coordinates are not universal.

**Caveat:** the paper also discusses limits/counterexamples; it does not prove every transformer can be globally re-based.

---

## W2T: LoRA Weights Already Know What They Can Do

- **Han et al. (2026)**
- <https://arxiv.org/abs/2603.15990>

Canonicalizes equivalent LoRA factorizations with QR/SVD, tokenizes the resulting components, and predicts adapter attributes/performance directly from weights.

**GENOME relevance:** factorization ambiguity, canonical weight tokens, and behaviour encoded in weights.

**Caveat:** LoRA updates are much smaller and lower-rank than full pretrained checkpoints.

---

## Structure Is Not Enough: Leveraging Behavior for Neural Network Weight Reconstruction

- **Meynent et al. (2025 workshop)**
- <https://openreview.net/forum?id=APsHrpqO3W>

Reports that adding behavioural/probing losses improves weight-space autoencoder reconstruction/generation.

**GENOME relevance:** supports combining weight reconstruction with executed functional loss rather than relying on MSE alone.

---

## How Training Window Length Shapes Neural Language Model Weights

- **Bugaud (ICML 2026 workshop)**
- <https://openreview.net/forum?id=BmjGywA2Ip>

Studies language-model weight geometry under context-window changes and reports that some weight-displacement directions are initialization dependent even when magnitudes are reproducible.

**GENOME relevance:** current caution against treating cross-seed raw directions as universal; supports function-space validation and base-state conditioning.

**Caveat:** workshop-scale evidence with a specific intervention; use as motivation, not a general law.

---

# C. Checkpoint trajectories and nowcasting

## Accelerating Training with Neuron Interaction and Nowcasting Networks — NiNo

- **Knyazev et al. (2024/ICLR 2025)**
- <https://arxiv.org/abs/2409.04434>

Uses graph representations of neuron interactions to predict future parameters periodically during ordinary optimization and reports training acceleration in vision and language tasks.

**GENOME relevance:** architecture-aware trajectory encoding, near-future weight prediction, and graph-based transformer connectivity.

---

## Pythia: A Suite for Analyzing Large Language Models Across Training and Scaling

- **Biderman et al. (2023)**
- <https://arxiv.org/abs/2304.01373>

Releases 16 language models from 70M to 12B with 154 checkpoints each, consistent training data order, and tools for training-dynamics research.

**GENOME relevance:** public decoder-only checkpoint trajectories for weight-encoder/trajectory pretraining.

---

## PolyPythias: Stability and Outliers across Fifty Language Model Pre-Training Runs

- **van der Wal et al. (ICLR 2025)**
- <https://arxiv.org/abs/2503.09543>

Adds 45 runs across five Pythia sizes, with nine additional seeds per size and roughly 7,000 checkpoints, to study seed/data-order stability.

**GENOME relevance:** public run-level splits, cross-seed dynamics, and hidden-run training data for early weight-space experiments.

---

# D. Dataset/task fingerprints and condensed training evidence

## Task2Vec: Task Embedding for Meta-Learning

- **Achille et al. (2019)**
- <https://arxiv.org/abs/1902.03545>

Builds fixed-dimensional task embeddings from Fisher-information estimates of a probe network without needing semantic label interpretation.

**GENOME relevance:** precedent for model-native task fingerprints based on how a network responds to data.

---

## Dataset2Vec: Learning Dataset Meta-Features

- **Jomaa, Schmidt-Thieme, Grabocka (2019)**
- <https://arxiv.org/abs/1905.11063>

Learns representations of datasets as hierarchical sets for meta-learning.

**GENOME relevance:** set encoders for corpus/task fingerprints.

---

## Dataset Condensation with Gradient Matching

- **Zhao, Mopuri, Bilen (ICLR 2021)**
- <https://arxiv.org/abs/2006.05929>

Learns small synthetic datasets by matching training gradients from real data.

**GENOME relevance:** gradient sketches as corpus information and the future joint “condensed data + genome” correction branch.

---

# E. Description length and structured compression

## The Description Length of Deep Learning Models

- **Blier, Ollivier (2018)**
- <https://arxiv.org/abs/1802.07044>

Studies neural networks through minimum description length and includes model-parameter encoding costs.

**GENOME relevance:** formal motivation for counting the shared decoder, per-model genome, residual, and data fit together.

---

## Compressibility Measures Complexity: Minimum Description Length Meets Singular Learning Theory

- **Urdshals et al. (2025)**
- <https://arxiv.org/abs/2510.12077>

Connects neural-network compressibility measurements with singular-learning-theory complexity estimates in experiments including Pythia.

**GENOME relevance:** possible later theoretical diagnostic for which checkpoints/regions should be genome-compressible.

---

## Low-Rank+Sparse Tensor Compression for Neural Networks

- **Hawkins et al. (2021)**
- <https://arxiv.org/abs/2111.01697>

Combines coarse low-rank tensor structure with sparse corrections.

**GENOME relevance:** direct basis for the structured-plus-exception representation.

---

# F. Adjacent theoretical/representation work

## HyperINR: Ensuring Semantics in Weights with Implicit Function Theorem

- **Qiu, Sonis, Shen (2026)**
- <https://openreview.net/forum?id=EQPv3DOC1x>

Uses a low-dimensional latent-to-weight hypernetwork for implicit neural representations and provides an implicit-function-theorem framing of semantic weight latents.

**GENOME relevance:** theoretical support for low-dimensional latent-to-weight mappings in a restricted setting.

---

## Self-Supervised Representation Learning on Neural Network Weights for Model Characteristic Prediction

- **Schürholt, Kostadinov, Borth (NeurIPS 2021)**
- <https://openreview.net/forum?id=F1D8buayXQT>

Learns self-supervised representations over model populations using weight-space augmentations.

**GENOME relevance:** pretraining tasks, model-zoo representations, and augmentation controls.

---

# How to use these references

Use the papers to import mechanisms, not conclusions:

- **G.pt / DeepWeightFlow / NNiT:** weight generation is possible in controlled settings.
- **GHNs / structure-aware neural program synthesis:** architecture structure should condition weight generation.
- **Transformer-NFN / Git Re-Basin / W2T:** symmetries and canonicalization are central.
- **SANE:** tokenize subsets/blocks rather than all scalar weights.
- **HyperNet Fields / NiNo / Pythia:** model trajectories, not only endpoints.
- **Task2Vec / gradient matching:** represent a task through model-native responses to its data.
- **MDL / low-rank+sparse:** count the complete code and permit a compact structured program plus exceptions.

The direct GENOME claim remains speculative until it passes the project’s own R0 and hidden-run gates.
