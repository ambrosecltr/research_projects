# Pythia v1 source matrix

## Accepted source family

GENOME v1 uses the standard non-deduplicated Pythia/PolyPythia GPT-NeoX checkpoints.

Requested revisions:

```text
W0: step0
WT: step143000
```

The RunPod source-resolution command converts these refs into immutable commit SHAs before downloading bytes. Hidden WT remains unresolved until the prediction seal exists.

## Whole-life split

| Size | Seeds | Count | Role | WT before hidden seal |
|---|---:|---:|---|---|
| 14M | 0–7 | 8 | training | available |
| 14M | 8 | 1 | development | available |
| 14M | 9 | 0 | excluded | never used |
| 31M | 0–7 | 8 | training | available |
| 31M | 8 | 1 | development | available |
| 31M | 9 | 1 | hidden | unavailable |

Total initial lives: 19.

## Initial download set

Download only:

- W0 and WT for 16 training lives;
- W0 and WT for two development lives;
- W0 only for one hidden life;
- tokenizer/config files required to load each snapshot.

Do not download 154 checkpoints per life. Intermediate checkpoints are not part of the first compiler dataset.

## Dataset evidence

Pythia was trained on the Pile, with preshuffled index files used by the seed runs. For v1:

- preserve the exact published dataset/order identities in provenance;
- pin and stream a deterministic accessible Pile sample for semantic evidence;
- use the same content sample across lives;
- compute W0-specific gradient/activation evidence separately per life;
- never transform an order seed or SHA digest directly into a semantic vector.

If the exact historical example stream cannot be reconstructed from accessible public data, record that limitation. It does not permit fabricated evidence.

## Storage estimate

Use actual receipts as the source of truth. Before materialization, use these conservative planning bands:

| Area | Planning allowance |
|---|---:|
| Model source W0/WT snapshots | 8–20 GB |
| Hugging Face cache/duplicate download headroom | 15–30 GB |
| Canonical states and evidence | 10–20 GB |
| Candidate/accepted programs and evaluations | 5–10 GB |
| Compiler checkpoints, logs and temporary files | 15–25 GB |

A new **100 GB RunPod network volume** is recommended for the first run. Increase before download if resolved file metadata exceeds the plan.

## Licences and provenance

The source plan records the model-card licence, requested ref, resolved commit, every downloaded file, actual bytes and SHA-256. Source files become immutable after receipts are written.

The source plan is configuration, not proof. `resolve-source-plan` and the materialization receipts establish the actual immutable identities used by the experiment.
