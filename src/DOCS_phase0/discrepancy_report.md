# Phase 0 — Discrepancy Report (diagram vs READMEs)

**Rule applied (build plan §Cross-cutting #1):** the `.md` READMEs are the spec; the
diagram is secondary. Where they disagree, the README wins and the conflict is recorded
here. The review/implementation brief file is not present in `DOCS/`, but the build plan
(`glacier_ssl_build_plan.md`) already encodes its three headline resolutions plus the
cross-cutting checklist; those are treated as the brief's content.

| # | Area | Diagram / naive reading | README-mandated resolution | Source |
|---|------|-------------------------|----------------------------|--------|
| 1 | Head 1 objective | Explicit positive/negative (InfoNCE) contrast | **DINO self-distillation** (teacher/student, EMA, prototypes, cross-entropy). No explicit negatives. | `head1_archi.md` §5, §1 |
| 2 | Head 2 objective | Gap-class classification ("1-year gap", "5-year gap") | **Continuous elapsed-time regression** (SmoothL1 on `log1p(years)`). Classification is an *optional* experiment only. | `head2_archi.md` §4, §6 |
| 3 | Head 3 clustering | Change Encoder → clustering → pseudo-labels as a mandatory per-step loss | **Staged experiment.** Baseline = two-view instance InfoNCE; prototype/cluster objective is a separate, later stage with pseudo-labels confined to train and refreshed on a schedule. | `head3_archi.md` §9, §17 |
| 4 | HPE merge stages | Could be read as 3 reductions (32→16→8→4) | **Exactly two** merge stages (32→16→8). A third stage would drop below the required 64 regional tokens. | `hpe.md` §4 |
| 5 | Sequence length `T` | 36 fixed (4 seasons × 9 years) | **Variable `T`**, always. 36 is one archive pattern, never a model requirement. Sampled windows of varying length. | `VIT_aarchi.md` §2, `Temptrans_aarchi.md` §1, build plan #3 |
| 6 | Temporal quantities | Could use array indices / assumed annual cadence | **Computed from actual acquisition dates only** (elapsed years = elapsed days / 365.2425). Irregular gaps stay visible. | `Temptrans_aarchi.md` §1, `head2_archi.md` §3 |
| 7 | TET attention | Full attention over all `T×64` tokens | **Factorized** temporal-then-spatial attention + MLP per block. Never flatten `T×64` into one sequence. | `Temptrans_aarchi.md` §2 |
| 8 | TET sequence output | Pool timestamp embeddings to get the sequence embedding | **Final context-token hidden state** is `sequence_out`. Do NOT derive it by pooling timestamp embeddings. | `Temptrans_aarchi.md` §4C |
| 9 | Head 1 input | Could crop the regional embeddings after the backbone | **Views built on images before the backbone**; Head 1 consumes `timestamp_out` per view. Latent-region cropping is a deferred, separate research note. | `head1_archi.md` §1, §2, §10 |
| 10 | Head 1 crop geometry | One identical crop for all views, or independent per-timestamp crops | **Same crop geometry across all timestamps within a view; different geometry across views.** | `head1_archi.md` §2, §3 |
| 11 | Head 2 predictor input | Feed `delta_years` / date diff / index diff | **Never.** Only symmetric visual pair features `concat(abs(u_i-u_j), u_i*u_j)`. | `head2_archi.md` §3, §11 |
| 12 | Head 3 encoder input | Pass the time gap to the Change Encoder | **Never.** Gap is used for sampling/eval only; the encoder gets visual features only. | `head3_archi.md` §4, §15 |
| 13 | Head 3 direction | Force `c(i→j) == c(j→i)` | **Preserve direction.** Reverse interval is a directed transformation, not an automatic positive. | `head3_archi.md` §6 |
| 14 | SCL band | 7th model input channel | **Quality information only**, used for validity masking; never a model channel. | `VIT_aarchi.md` §2 |

**Confirmed no further drift:** the ViT (no CLS/pool/head), HPE (two-stage, width-preserving,
no cross-timestamp mixing), and TET (3 named outputs, 8 blocks) specs are internally
consistent across the backbone files and the head files' backbone-contract sections.

**Context-token communication (the one flagged open item in `Temptrans_aarchi.md` §2/§187):**
resolved concretely in Phase 7 — see the decision ledger entry `tet.context_mode`.
