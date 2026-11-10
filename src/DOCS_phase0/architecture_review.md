# Architecture review after the first full pretraining run (Nov 2026)

**Scope.** End-to-end review of the glacier SSL system against what the project must
deliver, using the first completed 20-epoch run (step 6720) and a 40-held-out-glacier
evaluation. Records what was wrong, the evidence, the theory, what changed, and what was
deliberately left alone. Decision-ledger rows: "Architecture review" section.

## 1. What the system has to deliver

From the problem statement: learn **transferable representations of glacier evolution**
(retreat, boundary shrinkage, fragmentation, long-term change) from Sentinel-2 time series
**without labels**, and generalise to **unseen regions**. The intended downstream task is
sequence-level event classification (stable / retreat / advance / fragmentation /
disappearance). The novelty rests on the application and on the explicit three-way
decomposition: **state** (Head 1, DINO), **interval** (Head 2, elapsed time), **change**
(Head 3, directed change). See `research/glacier_ssl_novelty_report.md`.

Glacier evolution plays out over **years**. At 10 m pixels, terminus retreat of a few
metres per year is sub-pixel within a single year, while the seasonal snow cycle changes
the whole scene every three months. Any objective that only ever sees short spans will
learn seasonality, not evolution.

## 2. Verdict

The first run trained stably (no collapse; final epoch-mean DINO 3.65), but the
representation is weak, and the eval shows why. **Four root causes**, all with a concrete
mechanism, and none requires removing a component.

| Evidence (40 held-out glaciers unless noted) | Value |
|---|---|
| Head 2 MAE, trained head | **2.64 yr** |
| Predicting the training-set median gap (no model) | 1.70 yr |
| Linear fit on index difference + length + season (no images) | **0.087 yr**, Spearman 0.998 |
| Ridge probe of the gap on **untrained** post-TET features | 0.22 yr |
| Ridge probe on pre-TET (appearance-only) features: trained vs untrained | 1.43 vs 1.46 yr |
| Head 2 bias by gap: <0.5 / 0.5-1 / 1-2 / 2-4 / >=4 yr | +0.02 / -0.43 / -1.15 / -2.54 / -5.13 |
| Training: Head 2 loss (median) | 0.015 (smooth-L1, log1p) |
| Training: Head 3 InfoNCE, median as a fraction of chance log(n) | 18% (<10% in 32% of steps), ~7 candidates |
| Effective rank (RankMe, of 768): trained vs untrained | 22.4 vs 50.8 |
| Season / year linear probe: trained vs untrained | 57.5% / 63.2% vs 99.7% / 97.9% |
| Identity retrieval P@1: trained / untrained / raw 32x32 pixels | 0.53 / 0.28 / 0.99 |

The untrained model "winning" the season/year probes is not image understanding: the TET
adds the season and elapsed-time encodings to every token, and a random network passes
them through almost linearly. Those probes read the injected metadata. The comparison that
measures learned visual content is the pre-TET one, where trained is approximately equal to untrained.

## 3. Root causes and the theory behind them

### A. The temporal targets are inputs (Heads 2 and 3) -- shortcut learning

The TET adds `ElapsedTimeEncoding(elapsed)` and `SeasonEmbedding(season)` to every token.

* **Head 2** predicts `|elapsed_j - elapsed_i|`. Both elapsed values sit, encoded, in its
  own input. When a target is a deterministic function of an input feature, gradient
  descent finds that function long before any harder visual cue (shortcut learning;
  Geirhos et al. 2020). The ViT receives almost no pressure to encode temporal
  appearance. Evidence: training loss of about 0.015; untrained post-TET features already give
  0.22 yr; the trained head's error explodes outside the training gap range, because it
  learned to decode an encoding rather than to look.
* **Head 3** sends each interval's two endpoints through the TET with `elapsed = [0, gap]`
  and both seasons. Its two InfoNCE views share that gap and season pair exactly, while
  the in-batch negatives differ. InfoNCE can therefore match a temporal fingerprint
  instead of the change. This violates `head3_archi.md` principle 4 ("the Change
  Encoder receives visual features only") in spirit: the gap is not a direct input, but
  it is encoded in the regional features. Evidence: loss at 18% of chance with ~7
  candidates.

`head2_archi.md` section 8 anticipated this risk and prescribed exactly this control
("re-run a controlled TET variant without explicit elapsed-time embeddings for the Head 2
path"). The baseline implementation never applied it.

### B. A sensor artefact gives an absolute clock -- Sentinel-2 processing baseline 04.00

From baseline 04.00 (late 2021 / 25 Jan 2022, plus later reprocessing) L2A reflectance
carries a +1000 DN offset. The SSL download pipeline never removed it (the event-labelling
pipeline does, `glacier_gt/measure.py`). **12,432 of 37,476 observations (33%)** carry it.
In normalised units, every band steps up at 2022: **+1.2 sd in SWIR**, +0.2 sd in the visible bands.
This is a second leakage route, and it survives the firewall: "brighter SWIR = after 2022" is
visible in the pixels themselves. It also inflated the SWIR normalisation std by about 25% and
confounds every multi-year comparison, including the downstream retreat labels.

The acquisition date is only a proxy (late-December-2021 scenes are already on 04.00; 4
of 265 sampled scenes disagree), so the per-scene `s2:processing_baseline` was fetched
from the Planetary Computer STAC API for all 37,476 observations. After harmonisation the
visible-band step is within normal year-to-year variation and SWIR drops from +1.23 to
+0.25 sd. The residue is concentrated in 2022, the record Alpine melt year (less snow
gives brighter SWIR), so it is plausibly real signal; this cannot be proven from this data.

### C. The training timescale cannot contain evolution

The Kaggle run sampled **contiguous** windows of at most 8 seasonal observations: under
1.75 years. Head 2 and Head 3 intervals are drawn inside the window, so neither head ever saw a
multi-year interval, and within 1.75 years the dominant change is the seasonal cycle.
Evidence: Head 2's predictions saturate at the training range (bias of -5.1 yr for >=4-year
gaps). Window length is capped by GPU memory, but **time span does not need to be**.

### D. A ViT-B trained from scratch on ~24k images at batch size 2

ViTs lack convolutional inductive bias and need very large data or a pretrained
start. Here: 86M ViT parameters, 671 training glaciers (about 24k images), 2 glaciers per step,
6,720 steps. `VIT_aarchi.md` section 7 prefers a remote-sensing-pretrained checkpoint, and
`vit.pretrained_ckpt` existed, but **`load_pretrained()` was never called anywhere**, so
the setting was inert. Evidence: effective rank 22 of 768; pre-TET features no better
than random for the gap; raw pixels beat the model at identity retrieval.

### Secondary

* Small-batch DINO with K = 8192 prototypes for about 24k images: low effective rank is the
  classic partial-dimensional-collapse signature. Not changed yet (see section 7).
* Head 3 InfoNCE has only ~7 candidates per step. The firewall makes the same-glacier
  negatives (other intervals of the same glacier) genuinely hard, which partly
  compensates.
* Head 2 paired timestamps regardless of validity (head2 section 11.5): about 5% are mostly
  cloud or nodata. Fixed.

## 4. What changed

All changes are config flags. Default = new behaviour; the original is one flag away.
Checkpoints trained before the review resolve to the original behaviour automatically
(absent keys mean off; the old normalisation file is kept), so their evaluations stay valid.

| Change | Fixes | Key (default) |
|---|---|---|
| **Temporal firewall**: time-blind TET pass for Head 2 and Head 3; DINO keeps the time-aware context | A | `head2.time_blind` / `head3.time_blind` (true), `head1.time_blind` (false) |
| **Baseline harmonisation**: per-scene `max(DN - 1000, 0)` for baseline >= 04.00, after nodata detection; stats refitted | B | `data.harmonize_baseline` (true), `data.norm_stats_path` |
| Per-scene processing baseline recorded in the manifest | B | `scripts/fetch_processing_baselines.py` |
| **Sparse multi-year windows**: half the windows are a sorted random subset of the whole record (same T, years of span) | C | `sampling.window.sparse_prob` (0.5) |
| **Pretrained ViT wired in**: loader handles real checkpoint layouts (prefixes, 3-D tubelet kernel, cls + multi-frame positional table, grid resampling, band selection); 10x lower LR for pretrained weights | D | `vit.pretrained_ckpt`, `vit.pretrained_lr_scale` (0.1) |
| Head 2 pairs only valid timestamps | secondary | — |
| Eval: Head 2 scored on the path it trained on; new `pretrained` baseline (Prithvi ViT + untrained HPE/TET) | — | `evaluate.py --baselines` |

**Why time-blind is safe for the decomposition.** Without the encodings, temporal
attention is permutation-equivariant (unit-tested): a firewalled head sees an unordered
set of observations, so any temporal signal it uses must come from appearance. Weights
are shared with the time-aware path, so the TET still learns to use dates for the
state/sequence representations, and Heads 2 and 3 now push temporal cues into the
ViT/HPE features: the transferable part.

**Prithvi-EO-1.0-100M** matches this ViT exactly (ViT-B/16, 768-d, 12 blocks) and its six
HLS bands are this model's band order (B8A stands in for B08). Verified on the real file:
149/149 tensors load, and a full real-data training step runs with every change active.

## 5. What was deliberately not changed, and why

* **No component removed.** All three heads, the ViT, HPE, TET and context token stay.
  Removing Head 2 or Head 3 would collapse the design into DINO-on-satellite-time
  (S3-TSS / SatDINO territory) and erase the state/interval/change decomposition, which is
  the second novelty axis.
* **DINO keeps explicit time.** Head 1 learns the *state* of an observation in its
  sequence context, which is what the TET's temporal conditioning is for, and its target
  is not a temporal quantity. `head1.time_blind` exists only as an ablation.
* **The elapsed-time and season encodings remain** in the backbone; they are needed by the
  sequence/state representations and by downstream use.
* **K, temperatures and loss weights unchanged.** The run above did not isolate them;
  change one variable at a time.

## 6. Effect on novelty

This strengthens the paper. Two design principles come out of the review, both specific to
SSL on dated satellite time series and both testable by ablation:

1. **Separate temporal conditioning from temporal supervision.** A backbone that is given
   dates must not be asked to predict quantities derived from those dates through the same
   path. Time2Agri's time-difference task and SatMAE's temporal encodings coexist without
   addressing this leak. The firewall turns Head 2 into what `head2_archi.md` wanted:
   *visual* temporal distance.
2. **Decouple temporal span from sequence length.** Mixed contiguous/sparse windows give
   multi-year supervision at a fixed memory budget.

Also report the processing-baseline artefact: it affects any multi-year Sentinel-2 study
built on L2A data from 2022 onward, and documenting it is a contribution in itself.
Present all of these as design choices validated by ablation, not as new architectures.

## 7. Ablation grid this enables (one flag each)

| Row | Firewall H2/H3 | Harmonise | Sparse windows | Pretrained ViT |
|---|---|---|---|---|
| Original run | off | off | 0 | none |
| Full (new default) | on | on | 0.5 | Prithvi |
| - firewall | **off** | on | 0.5 | Prithvi |
| - sparse windows | on | on | **0** | Prithvi |
| - pretrained ViT | on | on | 0.5 | **none** |
| Prithvi only, no glacier SSL | eval baseline `pretrained` | | | |

Plus the head ablations from the novelty report. Harmonisation is a data fix and should not
be ablated.

## 8. What to watch in the next run

Retrain from scratch; do not resume an older checkpoint. The normalisation, the ViT's initial
weights and the optimizer's parameter groups all differ. The step-by-step Kaggle commands,
including the Prithvi download, are in `src/README.md` under "Kaggle flow".

* **Head 2 loss will be much higher** (0.08 at the first real-data step vs ~0.015
  before). That is the intended consequence: the answer is no longer in the input. The
  question is whether it falls below the metadata baselines in the eval.
* **Head 3 loss** should no longer fall to a few percent of `log(n_intervals)` early on.
* DINO collapse guard as before (`dino` pinned at 9.01, or the `COLLAPSE:` warning).
* Eval: compare trained vs `pretrained` vs `random`; Head 2 time-blind vs the metadata rows.

## 9. Open risks

* Visual elapsed-time prediction may simply be hard for glaciers. If time-blind Head 2
  stays near the median-gap baseline, report it honestly; Head 2 still regularises the
  backbone, and its shortcut-free score is what a reviewer will accept.
* The seasonal cycle may still dominate the representation (season eta-squared ~0.2).
  Sparse windows help Heads 2 and 3; DINO may need season-matched positive pairs in a
  later experiment.
* Domain gap: Prithvi was pretrained on HLS over the contiguous US, not on glaciers; its
  normalisation differs (fine-tuning at 0.1x LR is meant to absorb this).
* Small-batch DINO (2 glaciers per step) with K = 8192 is a likely cause of low rank;
  K = 2048 is the first thing to try if rank stays low.
