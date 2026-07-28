# GENOME experiment decision log

Copy this file into every experiment run directory as `decision_log.md` and complete it before looking at hidden Gate results.

---

## Identity

```text
run_id:
date_utc:
research_level: G0 / G1-Pure / G1-Distill / G2 / G3 / G4
owner/agent:
parent_run_id:
code_commit:
working_tree_patch_hash:
resolved_config_hash:
environment_hash:
R0_specimen_hash:
interpreter_hash:
compiler_hash:
```

## Question

What single question does this experiment answer?

> 

## Hypothesis

State the expected measurable result.

> 

## Why this experiment now?

Which previous observation justifies it, and what alternative explanation will it separate?

> 

## Allowed conditioning information

Check every item available to the method at candidate inference:

- [ ] architecture graph/config
- [ ] tokenizer/vocabulary
- [ ] W0 seed replay
- [ ] stored W0 shared base
- [ ] training recipe
- [ ] static dataset fingerprint
- [ ] initial gradient sketches
- [ ] early trajectory prefix
- [ ] R0/target final weights
- [ ] final reference logits on fit set
- [ ] final reference logits on probe set
- [ ] target ID
- [ ] external checkpoint data
- [ ] other: 

Explain why the selected research-level label is correct:

> 

## Data split contract

```text
training run IDs:
development run IDs:
hidden run IDs:
D_genome_fit hash:
D_fingerprint hash:
D_probe hash:
D_verifier_hidden hash:
P_generation hash:
```

Leakage checks completed:

- [ ] checkpoint slices inherit their run split
- [ ] hidden endpoint absent from training artifacts
- [ ] hidden verifier unavailable to training/refinement
- [ ] final probe outputs are labelled as distillation if used

## Method

### Representation

```text
base mode:
formula/opcodes:
block size:
global code:
layer code:
tensor code:
code dtype:
patch type/budget:
interpreter architecture:
```

### Training/fitting

```text
optimizer:
updates/tokens:
loss terms and weights:
candidate sample count:
selection rule:
latent refinement:
repair protocol:
```

### Baselines

- [ ] raw/canonical WT
- [ ] W0
- [ ] quantized WT
- [ ] quantized Delta-T
- [ ] SVD
- [ ] low-rank+sparse
- [ ] ordinary continuation
- [ ] Track 1 transport
- [ ] LoRA repair
- [ ] full-weight repair
- [ ] mean/nearest genome
- [ ] linear latent extrapolation
- [ ] shuffled/random control
- [ ] other: 

## Predeclared acceptance rule

Write the exact quality, byte, and compute gates before hidden evaluation.

```text
loss gate:
logit gate:
generation-health gate:
maximum payload/single-model ratio:
maximum child compute:
maximum repair budget:
```

## Expected failure signatures

What result would support each alternative explanation?

| Observation | Interpretation | Next branch |
| --- | --- | --- |
|  |  |  |
|  |  |  |
|  |  |  |

## Commands run

```bash
# exact commands
```

## Artifacts

| Artifact | Path | SHA-256 | Bytes |
| --- | --- | --- | ---: |
|  |  |  |  |

## Development results

Record all candidates, not only the best.

| Candidate | Probe loss | Bytes | Decode s | Refinement/repair | Selected? |
| --- | ---: | ---: | ---: | --- | --- |
|  |  |  |  |  |  |

Selection rationale:

> 

## Hidden Genome Gate result

```text
candidate MGP hash:
validity:
hidden validation loss:
loss z-score:
perplexity:
anchor KL:
top-k agreement:
generation health:
payload bytes:
single-model bytes:
amortized bytes at declared N:
decode seconds:
child-generation compute:
repair compute:
pass/fail:
failure codes:
```

## Full result interpretation

What was learned? Separate demonstrated result from inference.

### Demonstrated

> 

### Plausible interpretation

> 

### Not demonstrated

> 

## Unexpected findings and failures

> 

## Integrity/leakage audit

- [ ] target endpoint not available at inference beyond declared contract
- [ ] hidden verifier not used for selection/stopping
- [ ] decoder/interpreter bytes counted
- [ ] best-of-N cost counted
- [ ] patch bytes counted
- [ ] fingerprint/compile/probe/repair compute counted
- [ ] cross-run alignment verified where applicable
- [ ] all failed candidates retained in metrics

## Decision

Choose one:

- [ ] continue same branch with one specified change
- [ ] promote to next phase
- [ ] branch to representation redesign
- [ ] branch to function-space target
- [ ] branch to genome-native architecture
- [ ] stop this method

Next experiment and exact reason:

> 
