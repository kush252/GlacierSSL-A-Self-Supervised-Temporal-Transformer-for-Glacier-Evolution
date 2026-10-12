# Glacier SSL — Phased Build Plan

**What this document is:** a build roadmap for an AI coding agent. It does **not** restate the architecture specs. Each phase names the exact `.md` file(s) to open and treat as the source of truth for that part, plus the process rules, connection points, and cautions to keep in mind while building it.

**Source-of-truth files (read, never override):**
- `VIT_aarchi.md` — Multispectral Spatial ViT
- `hpe.md` — Hierarchical Patch Embedding (HPE)
- `Temptrans_aarchi.md` — Temporal Evolution Transformer (TET)
- `head1_archi.md` — Head 1, Temporal DINO
- `head2_archi.md` — Head 2, Temporal Distance Prediction
- `head3_archi.md` — Head 3, Temporal Change Representation
- The **review/implementation brief** (the architecture-review document) — diagram-vs-spec contradictions and the 37-item implementation checklist

**How to use this plan:** build strictly in phase order. Do not skip ahead to a head before the backbone it depends on exists and passes its shape checks. When a phase says "read X.md," open that file and implement to its contract exactly — including its own "implementation checks" / "validation checks" section, which is the acceptance test for that phase.

---

## Cross-cutting rules (apply in every phase)

Keep these in mind continuously; they are the most common ways this build goes wrong.

1. **READMEs are the spec; the diagram is secondary.** Where the diagram and a `.md` disagree, follow the `.md` and record the discrepancy. The review brief already lists the known conflicts (Head 1 is DINO, not explicit negatives; Head 2 is regression, not gap-classification; Head 3 clustering is a staged experiment, not a mandatory per-step loss). Do not implement the diagram's version of these.
2. **Do not invent architecture.** No new modules, objectives, pooling tricks, or interface changes beyond what the files specify. Unresolved choices are marked "experiment" in the files — expose them as config, do not silently pick and bury them.
3. **Variable length is a property of the whole pipeline, not just the model.** `T` is never fixed (not 36, not 4/year, not 9 years). Every layer — dataset, collate, sampler, backbone, pooling, loss — must preserve this. Test it at each level, not just once.
4. **Dates are the only temporal truth.** All temporal quantities (TET encodings, sequence span, Head 2 targets, Head 3 intervals) are computed from actual acquisition dates, never array indices or an assumed annual cadence.
5. **Padding/validity masks must be inert.** Padded timestamps, invalid observations, and invalid regions must never affect attention, pooling, span, pair/interval sampling, or any loss.
6. **No leakage.** Split by glacier identity (and region where feasible) *before* generating any pair/interval. Fit normalization on the train split only. Keep pseudo-labels/clusters confined to train.
7. **Separate "fixed" from "experimental."** Hyperparameters labelled initial/baseline in the files go into config with the file's default; they are starting points, not proven optima. Never hardcode them in module logic.
8. **Reproducibility is a requirement, not a nicety.** Persist dates, original indices, crop coordinates, validity masks, glacier IDs, seeds, and sampler config with every generated example.

---

## Phase 0 — Orientation and discrepancy report

**Goal:** before writing any module, produce the alignment artifact that governs the whole build.

**Read:** all seven files, with the review/implementation brief last.

**Do:**
- Produce a written **discrepancy report**: diagram vs each README, listing each contradiction and the resolution (follow the README). Start from the conflicts the review brief already identified and confirm nothing else drifts.
- Produce a **decision ledger** that tags every non-trivial choice as *fixed architecture* or *experiment*, pulling the "experiments" tables from `head2_archi.md`, `head3_archi.md`, and the "status/experiment" notes in the backbone files.
- Produce a one-page **tensor-contract map** listing the agreed shapes at every boundary (ViT out → HPE out → TET's three outputs → each head's input). Take the numbers from the files; this map is the reference the integration phases check against.

**Exit criteria:** discrepancy report, decision ledger, and tensor-contract map exist and are internally consistent. Do not start coding modules until these are done — the review brief explicitly recommends generating the discrepancy report and plan *first*.

---

## Phase 1 — Repository scaffolding, config, logging, reproducibility

**Goal:** the skeleton everything else plugs into.

**Read:** no new spec; use the decision ledger from Phase 0 to decide what must be configurable.

**Do:**
- Lay out modules: `data/`, `backbone/` (vit, hpe, tet), `heads/` (head1, head2, head3), `train/`, `eval/`, `tests/`, `configs/`.
- Build a single config system. Every value the files call "initial/baseline/experiment" is a config field with the file's default, never a literal in module code.
- Central logging for per-loss values, gradient norms where practical, valid-sample counts, and collapse diagnostics (these are demanded by the head files' evaluation sections).
- Global seeding and a recorded run-metadata block (config snapshot + git state + seeds).

**Exit criteria:** an empty end-to-end run boots, loads a config, and logs. No model yet.

---

## Phase 2 — Data manifest and sequence construction

**Goal:** turn the raw archive into per-glacier chronological sequences with full metadata, and define the splits.

**Read:** `VIT_aarchi.md` §2 (input contract) for what one timestamp is; the review brief's dataset/sequence sections (manifest fields, variable-length, glacier-identity split).

**Do:**
- Build a manifest row per observation: glacier ID, observation ID, acquisition date, band paths, quality metadata, spatial reference.
- Assemble each glacier's **complete valid chronological observation list**. Do **not** pre-bake fixed 1/4/9-year sequences — windows are sampled later (Phase 4). The review brief flags this fixed-sequence idea as the specific thing to avoid.
- Preserve missing observations and irregular gaps as real gaps in the timeline.
- **Split train/val/test by glacier identity here, before any pairing**, so later pair/interval sampling cannot leak.

**Connections:** feeds the preprocessing (Phase 3) and the window/view sampler (Phase 4).

**Exit criteria:** can enumerate, for any glacier, an ordered dated observation list; splits are disjoint by glacier.

---

## Phase 3 — Preprocessing and quality control

**Goal:** deterministic, leakage-free image preprocessing matching the ViT's expectations.

**Read:** `VIT_aarchi.md` §2 "Preprocessing" (band order, SCL handling, normalization policy, band-alignment checks).

**Do:**
- Enforce the documented six-band order and the rule that SCL is quality information, not a model channel.
- Fit band-wise normalization statistics **on the training split only** and reuse everywhere; do not per-image normalize.
- Implement validity masking from SCL-derived quality; define, for each head, how invalid observations/regions are excluded (the heads reference these masks in their losses).
- Verify band alignment, extent, nodata, reflectance scaling, pixel dimensions before stacking.

**Connections:** consumes Phase 2 manifest; produces normalized, masked, stackable per-timestamp tensors for the ViT.

**Exit criteria:** a timestamp loads as the ViT's expected per-timestamp tensor with a matching validity mask; normalization stats are train-only and persisted.

---

## Phase 4 — Window sampling, global/local views, augmentation, metadata tracking

**Goal:** produce training examples *before* the backbone, with time-consistent crops and tracked validity. This phase is where the crop policy that all three heads depend on is enforced.

**Read:** `head1_archi.md` §2–§3 (view-before-backbone rule, multi-crop config, per-view time-consistent crop geometry); `head3_archi.md` §8 (two-view, same directed interval, aligned endpoints); the review brief's §4 (three-way split of crop geometry vs spatial vs radiometric augmentation).

**Do:**
- Implement a **variable-length window sampler** over each glacier's observation list; expose min/max window length and short/medium/long sampling frequency as config (the review brief lists this as an explicit decision not to leave to the agent).
- Implement view construction **on images, not on backbone embeddings** (for the primary DINO objective).
- Enforce the core crop rule: within one view, the **same crop geometry across all timestamps**; **different geometries across views**; never one identical crop for all views, never independent per-timestamp crops within a view.
- Separate and track three distinct things: crop geometry (where), spatial augmentation (flips/rotations), radiometric augmentation (reflectance/noise). Keep augmentations physically plausible; do not erase snowline/terminus/fragmentation evidence.
- Track per-view, per-timestamp **crop validity**; exclude invalid view/timestamp pairs downstream.
- Persist dates, original indices, crop coordinates, validity masks, glacier ID, and seeds with every generated example.

**Connections:** this is the shared front end feeding Phase 8's backbone; its outputs are consumed differently by Head 1 (Phase 9), Head 2 (Phase 10), Head 3 (Phase 11).

**Exit criteria:** a view is reproducible from stored metadata; crop geometry is provably constant through time within a view and different across views; invalid pairs are flagged.

---

## Phase 5 — Multispectral Spatial ViT

**Goal:** the per-timestamp spatial encoder.

**Read:** `VIT_aarchi.md` in full — especially §3–§6 (patch embedding, positional encoding, encoder blocks, output contract) and §7 (pretrained-checkpoint handling).

**Do:**
- Implement exactly the patch projection, positional encoding, and encoder stack the file specifies. Do not add CLS, pooling, or any prediction head.
- Process each timestamp independently with shared weights; no temporal operation here.
- Implement the checkpoint-compatibility checks from §7 as explicit guards (patch size, dim, depth, bands, order, normalization, positional format); adapt/reinitialize deliberately if mismatched.

**Connections:** input from Phase 3/4 per-timestamp tensors; output is the ViT contract consumed by HPE.

**Exit criteria:** `VIT_aarchi.md` §10 implementation checks all pass.

---

## Phase 6 — Hierarchical Patch Embedding (HPE)

**Goal:** compress each timestamp's spatial grid to the regional grid.

**Read:** `hpe.md` in full — especially the input/output contracts (§2–§3), the two-stage merge (§4–§5), regional positional encoding (§6), and the forward path (§10).

**Do:**
- Implement the reshape and the two learned patch-merge stages exactly as specified (fixed neighborhood order, LayerNorm + linear projection, width preserved). Two stages, not three.
- Add the learned regional positional embedding at the output grid; do not add season/time/year embeddings here (those belong to TET).
- Treat the timestamp dimension as an independent batch — **never merge across timestamps.**

**Connections:** consumes the ViT output contract; emits the regional contract to TET.

**Exit criteria:** `hpe.md` §11 implementation checks pass, including variable `T` (down to a single timestamp) and "no cross-timestamp mixing."

---

## Phase 7 — Temporal Evolution Transformer (TET)

**Goal:** the temporal model and the three representations the heads consume.

**Read:** `Temptrans_aarchi.md` in full — input/temporal embeddings (§1), factorized temporal/spatial attention (§2), hyperparameters (§3), outputs (§4), and the "implementation item that must be resolved explicitly" (context-token path).

**Do:**
- Implement token-level temporal embeddings (season + elapsed-time) and the sequence-level context token per §1, computing all temporal scalars from dates. Keep the `ElapsedTimeEncoding(scalar) -> (768,)` interface swappable (config), as the file requires.
- Implement **factorized** temporal-then-spatial attention + MLP per block; do **not** flatten all `T×64` tokens into one full-attention sequence.
- **Resolve the context-token communication path as its own sub-task**: choose one concrete mechanism, implement it, and unit-test it before any full training. The review brief and the TET file both single this out as the main remaining ambiguity.
- Exclude the context token from spatial attention; expose the three outputs (regional, timestamp via masked pooling, sequence = final context token) exactly as the file defines them. Do not derive the sequence output by pooling timestamp embeddings.

**Connections:** consumes HPE output; emits `regional_out`, `timestamp_out`, `sequence_out` to the heads.

**Exit criteria:** the five "minimum tests" at the end of `Temptrans_aarchi.md` pass, especially context-token update and padded-batch correctness.

---

## Phase 8 — Backbone integration (ViT → HPE → TET)

**Goal:** one wired backbone that respects variable length and masks end to end.

**Read:** the §12 "interface" sections of `VIT_aarchi.md`, `hpe.md`, and `Temptrans_aarchi.md`; cross-check against the Phase 0 tensor-contract map.

**Do:**
- Connect the three modules using the exact documented tensor contracts; no boundary projections beyond what's specified.
- Implement padded variable-length batching and propagate timestamp/pixel/region validity masks correctly through every stage.
- Verify a batch mixing short/medium/long sequences produces correct per-sequence outputs with padding provably inert.

**Connections:** this backbone is frozen-interface shared infrastructure for all three heads.

**Exit criteria:** end-to-end shapes match the contract map for several `T` values; masks verified inert against attention/pooling/span.

---

## Phase 9 — Head 1: Temporal DINO

**Goal:** state-representation objective on timestamp embeddings.

**Read:** `head1_archi.md` in full — backbone interface (§2), multi-crop (§3), projector/prototype (§4), objective (§5), invariance rules (§6).

**Do:**
- Consume **timestamp embeddings** per view from the shared backbone (not regional, not sequence).
- Implement teacher/student with EMA, projector, and weight-normalized prototype layer per §4–§5; exclude same-view pairs, invalid timestamps, and invalid crop/timestamp pairs from the loss.
- Honor §6: encourage invariance to nuisance views, **not** to genuine evolution/identity change. Do not add explicit negatives (that's the diagram's version).

**Connections:** uses Phase 4 views and Phase 8 backbone; contributes `L_DINO`.

**Exit criteria:** `head1_archi.md` §9 validation checks pass (view shapes, prototype logit shapes, EMA-not-gradient, collapse monitoring).

---

## Phase 10 — Head 2: Temporal Distance Prediction

**Goal:** symmetric elapsed-time regression on timestamp-embedding pairs.

**Read:** `head2_archi.md` in full — backbone contract (§2), pair construction/targets (§3), pair features and MLP (§4), target transform and loss (§5), shortcut checks (§8), final baseline (§13).

**Do:**
- Sample same-glacier valid timestamp pairs with the gap-stratified baseline; compute targets from dates in years.
- Build the symmetric pair feature and regression MLP exactly as the file's baseline specifies; predict in transformed space and convert back for metrics.
- **Never** pass `delta_years`, date differences, or index differences into the predictor.
- Implement the shortcut controls (visual-feature-only, metadata-only) and the pair-order-invariance unit test — these are required before claiming the head works, because TET exposes explicit elapsed-time encodings.

**Connections:** uses Phase 8 `timestamp_out`; contributes `L_distance`.

**Exit criteria:** `head2_archi.md` §12 implementation checks pass, including pair symmetry and the two control experiments running.

---

## Phase 11 — Head 3: Temporal Change Representation

**Goal:** directed regional change representations with staged clustering.

**Read:** `head3_archi.md` in full — regional interface and alignment requirement (§3), interval sampling (§4), region pair features (§5), Change Encoder (§6), projection (§7), two-view objective (§8), staged clustering (§9), final baseline (§16–§17).

**Do:**
- Consume **`regional_out`**; first **verify the 64 regions correspond across dates** (the file says this must hold or an alignment step is required before training this head).
- Build directed interval features, the shared region-wise Change Encoder, and validity-aware pooling exactly per the baseline; preserve direction (do not force reverse = forward).
- **Never** pass the time gap into the Change Encoder.
- Implement the **two-view instance contrastive baseline first**; treat prototype/cluster objectives as a separate, staged experiment with pseudo-labels confined to train and refreshed on a schedule — not a mandatory per-step loss. This resolves the diagram ambiguity the review brief flagged.

**Connections:** uses Phase 8 `regional_out`; contributes `L_change`.

**Exit criteria:** `head3_archi.md` §15 implementation checks pass, including region-correspondence, gap-not-in-encoder, direction preservation, and no cluster collapse.

---

## Phase 12 — Combined SSL training loop

**Goal:** joint three-head training over the shared backbone.

**Read:** the combined-loss sections of `head1_archi.md` §8, `head2_archi.md` §7, `head3_archi.md` §11 (same initial coefficients in all three).

**Do:**
- Implement the combined objective with the documented initial coefficients, but keep them in config and **log each loss separately plus gradient norms** — the files warn that a coefficient does not equal influence.
- Run each head's distinct sampler (window sampler, Head 2 pair sampler, Head 3 interval sampler) off the shared data pipeline.
- Ensure all three student-side objectives backpropagate into the intended shared-backbone parameters; teacher is EMA-only.
- Implement checkpointing that saves student backbone, EMA teacher, all heads, optimizer state, config, and metadata for reliable resumption.

**Exit criteria:** a multi-head step runs with finite losses, verified gradient flow to the backbone, and per-loss logging.

---

## Phase 13 — Test suite

**Goal:** the acceptance gate before any scaled run.

**Read:** the "implementation checks" / "validation checks" sections of every file, plus the review brief's checklist (items on variable length, masks, crop consistency, symmetry/direction).

**Do:**
- Unit tests for every documented intermediate and output shape (contract map from Phase 0).
- Variable-length and missing-observation tests at dataset, collate, backbone, and loss levels.
- Date-derived target tests; Head 2 pair-swap symmetry; Head 3 interval-direction preservation.
- Crop-consistency-through-time and crop-validity tests.
- Mask-inertness tests (padding changes nothing in attention/pooling/span/loss).
- Context-token communication test (from Phase 7).

**Exit criteria:** all module check-lists green; a tiny controlled batch overfits without instability (per Head 1 §9 and Head 3 §15).

---

## Phase 14 — End-to-end sanity run, evaluation, and ablations

**Goal:** prove the system trains and set up honest evaluation — without claiming scientific validity from "the code runs."

**Read:** evaluation sections — `head2_archi.md` §9–§11, `head3_archi.md` §12–§14, plus the leakage safeguards in both.

**Do:**
- Small end-to-end run with finite losses and verified gradients across all three heads.
- Evaluate on **held-out glaciers** (and regions where feasible); run the per-head ablations and the Head 2 shortcut diagnostics.
- Report representation diagnostics and training-health metrics, keeping test data untouched until architecture/hyperparameters are fixed.
- Produce the final **unresolved-decisions list**, separating fixed architecture requirements from experimental choices (ties back to the Phase 0 ledger).

**Exit criteria:** sanity run reproducible from stored config/seeds; evaluation and implementation-check results reported separately, with no validity claims beyond what the ablations support.

---

## Recommended execution order (summary)

Discrepancy report and contracts (Phase 0) → scaffold/config (1) → data manifest and splits (2) → preprocessing (3) → windows/views (4) → **backbone bottom-up: ViT (5) → HPE (6) → TET (7) → integrate (8)** → **heads: DINO (9) → Distance (10) → Change (11)** → combined loop (12) → tests (13) → sanity/eval (14).

Build and verify the data pipeline and backbone interfaces **before** integrating the heads, so temporal-sampling, crop-alignment, and tensor-contract errors are caught before they contaminate the joint objective.
