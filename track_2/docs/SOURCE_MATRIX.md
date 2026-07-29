# Pythia v1 source matrix

Requested revisions are `step0` for W0 and `step143000` for WT. Materialized refs
are pinned to immutable commits and file hashes.

| Size | Seeds | Count | Role | WT available now |
|---|---:|---:|---|---|
| 14M | 0–6, 8, 9 | 9 | training and formula development | yes |
| 14M | 7 | 1 | fresh development | yes |
| 31M | 0–6, 8 | 8 | training and formula development | yes |
| 31M | 7 | 1 | fresh development | yes |
| 31M | 9 | 1 | hidden | no |

Total: 17 training lives, two development lives, and one hidden life.

Seed8 is not independent development evidence because it affected formula design.
Pythia 14M seed9 is a training life and may provide an endpoint-free compiler input
and a fitted target. Pythia 31M seed9 is the only hidden evaluation.

The source volume contains W0 and WT for all training and development lives. It
contains only W0 for hidden Pythia 31M seed9. Intermediate checkpoints are not part
of v1.

The first semantic sample is from
`EleutherAI/pile-standard-pythia-preshuffled` at commit
`bac79b6820adb34e451f9a02cc1dc7cd920febf0`.

- Even records are refinement data.
- Odd records are formula-tuning data.
- Fresh development verification uses 128 batches from
  `document-00001-of-00020.bin`.
- The formula data uses `document-00000-of-00020.bin`.

The project explicitly targets a functionally good same-corpus endpoint. It does
not claim exact reconstruction of raw WT coordinates when the exact historical
example order cannot be reconstructed.
