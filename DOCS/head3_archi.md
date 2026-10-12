# Head 3 Architecture — Temporal Change Representation

**Status:** Proposed baseline with explicit experiments for unresolved design choices.  
**Project:** Glacier evolution self-supervised learning (SSL).  
**Backbone:** Multispectral ViT → Hierarchical Patch Embedding (HPE) → Temporal Evolution Transformer (TET).  
**Primary purpose:** Learn transferable representations of *changes between glacier observations*, including local structural change and longer-term evolution patterns, without manually labelled change events.

---

## 1. Purpose and boundaries

Head 3 should answer:

> Given two observations of the same glacier at different dates, what changed between them, and how is that change related to changes observed in other glaciers or time intervals?

Target phenomena include retreat, advance, boundary shrinkage, fragmentation, disappearance, snowline migration, and other persistent changes. The head must not assume that a larger time gap implies greater physical change.

The three SSL heads have distinct responsibilities:

| Component | Main learning target |
|---|---|
| Head 1 — Temporal DINO | Consistent representations of glacier states across valid views |
| Head 2 — Temporal Distance | Elapsed time between two observations of the same glacier |
| **Head 3 — Temporal Change Representation** | Represent the visual/structural change between two observations and group similar change patterns |
| Shared ViT + HPE + TET | Learn features useful to all three objectives |

Head 3 is not a supervised retreat/advance classifier. It does not receive event labels. It also does not directly estimate physical area loss or displacement. Those claims would require suitable geometric measurements or downstream evaluation.

## 2. Design principles

1. **Use regional TET outputs.** Change can be localized. The timestamp-pooled embedding may hide where a change occurred.
2. **Preserve temporal direction.** The change from date \(i\) to date \(j\) is ordered: \(i \rightarrow j\). The reverse interval is not interchangeable.
3. **Separate nuisance invariance from real change.** Mild radiometric variation and valid augmentation should not dominate the representation, but aggressive augmentation must not erase snowline, terminus, or fragmentation evidence.
4. **Do not use the time gap as an input to the change predictor.** The model should not infer change representations from the answer to Head 2. Actual dates can be used for sampling and evaluation, but the baseline Change Encoder receives visual features only.
5. **Do not treat every other interval as a negative forever.** Different glaciers can undergo similar changes. Pure instance-level InfoNCE can push genuinely similar change events apart; clustering-based or prototype-based objectives must be evaluated.
6. **Avoid relying on noisy pseudo-labels as ground truth.** Clusters are latent groups, not verified physical event classes.

## 3. Backbone interface

For a glacier with chronologically ordered, valid observations:

```text
Input sequence:
    X                         (T, 6, 512, 512)

Multispectral ViT:
    spatial tokens            (T, 1024, 768)

HPE:
    regional tokens           (T, 64, 768)

TET:
    regional_out              (T, 64, 768)
    timestamp_out             (T, 768)
    sequence_out              (768,)
```

**Head 3 uses `regional_out: (T, 64, 768)`.** It does not directly use `timestamp_out` or `sequence_out` in the baseline.

The 64 regions must have a consistent spatial correspondence across timestamps. The existing HPE/TET design is expected to maintain a common regional grid; verify this property in implementation. If regions are not geographically aligned, the pairwise region difference described below is invalid and a spatial alignment strategy must be added before training Head 3.

Padded timestamps and invalid observations must be excluded from interval construction. Invalid/cloudy regions must be handled with the data pipeline's validity masks. Do not silently treat masked or missing regions as real zero-valued features.

## 4. Interval sampling and targets

Choose two valid observations \(i\) and \(j\) of the **same glacier**, with \(d_i < d_j\). Construct a directed interval \(i \rightarrow j\).

- Use actual acquisition dates, not timestamp array indices.
- Sample short, medium, and long intervals where the data supports them.
- Sample both adjacent and non-adjacent observations.
- Do not make every training interval consecutive: that would underrepresent long-term evolution.
- Do not force a fixed number of observations per year.
- The baseline does not require a manually labelled event class or a numeric change-magnitude target.

Initial interval sampling experiment:

| Setting | Candidate values |
|---|---|
| Pair gap | Adjacent-only; mixed gaps; gap-stratified mixed sampling |
| Maximum interval gap | No cap; 3 years; 5 years; dataset-supported alternative |
| Pair weighting | Uniform over valid pairs; balanced across short/medium/long gap bins |
| Glacier sampling | Uniform over glaciers, then sample intervals within each glacier; compare with uniform-over-interval sampling |

**Recommended baseline:** sample glaciers uniformly, then sample intervals with a gap-stratified policy. This reduces domination by glaciers with many observations and helps cover a range of timescales. Record the actual date gap for diagnostics, but do not pass it into the Change Encoder.

Split data by glacier identity (and, where feasible, geographic region) **before** generating training, validation, and test intervals. Do not let intervals from the same held-out glacier leak across splits.

## 5. Change feature construction

Let \(R_i \in \mathbb{R}^{64 \times 768}\) and \(R_j \in \mathbb{R}^{64 \times 768}\) be the TET regional embeddings at the two dates. Let \(r_i^k\) and \(r_j^k\) denote the embeddings for corresponding region \(k\).

For each corresponding region, construct a directed pair feature:

\[
q_{i\rightarrow j}^{k} =
\left[
\operatorname{LN}(r_i^k)
\;\Vert\;
\operatorname{LN}(r_j^k)
\;\Vert\;
\left(\operatorname{LN}(r_j^k)-\operatorname{LN}(r_i^k)\right)
\;\Vert\;
\left|\operatorname{LN}(r_j^k)-\operatorname{LN}(r_i^k)\right|
\;\Vert\;
\operatorname{LN}(r_i^k)\odot\operatorname{LN}(r_j^k)
\right].
\]

Here, `LN` is LayerNorm, `||` is concatenation, and `⊙` is elementwise multiplication.

- Signed difference preserves the direction of representation change.
- Absolute difference captures the size of the feature discrepancy independently of sign.
- The start/end features provide context for interpreting the difference.
- The product provides a simple interaction feature.

The pair feature dimension is `5 × 768 = 3840` per region.

**Initial design:** use all five feature groups. This is a hypothesis, not a guaranteed optimum. Compare against `[r_j-r_i]`, `[abs(r_j-r_i)]`, and `[r_i, r_j, r_j-r_i]` in ablation experiments.

### Important interpretation caveat

A signed latent difference is not automatically a physical direction such as retreat or advance. Its axes are learned, and the representation may rotate or reorganize during training. Physical interpretation must be tested using held-out glacier observations and independent geometric or expert-derived evaluation.

## 6. Change Encoder

Apply the same Change Encoder to each region's pair feature. The baseline is a shared region-wise MLP:

```text
Input per region:       3840
LayerNorm:              3840
Linear:                 3840 → 1536
GELU
Dropout:                 0.1
Linear:                 1536 → 768
GELU
LayerNorm:               768
Output change token:     768
```

For an interval:

```text
64 region pair features       (64, 3840)
            |
            v
Shared Change Encoder
            |
            v
Regional change tokens        (64, 768)
            |
            v
Validity-aware regional pooling
            |
            v
Interval change embedding     (768,)
```

Pool only over valid regions. The baseline pooling is a validity-aware mean. Do not use an unmasked mean if some regions are invalid.

### Pooling experiments

Compare:
- Validity-aware mean pooling (baseline).
- Validity-aware mean plus max pooling followed by a learned projection.
- Learned attention pooling over the 64 change tokens.

Mean pooling is the simplest stable baseline. Attention pooling may help emphasize a small retreating terminus or fragmented region, but it can also overfit to cloud artifacts or local noise.

### Directionality check

The baseline is directed. Reversing an interval swaps the start/end features and changes the signed difference. Do not force \(c_{i\rightarrow j}=c_{j\rightarrow i}\).

Run a diagnostic with both directions. If reverse intervals are included as training examples, treat them as directed transformations rather than automatically as positive pairs. A reversed interval can be a useful control, but it is not semantically identical to the forward interval.

## 7. Projection head and normalized change representation

Use a projection MLP for the SSL objective, while retaining the 768-dimensional interval embedding as the primary Head 3 representation.

```text
Interval change embedding: 768
LayerNorm:                  768
Linear:                     768 → 512
GELU
Linear:                     512 → 256
L2 normalization:           256
```

- `change_embedding`: `(768,)` — output of the Change Encoder plus regional pooling; retain for downstream use.
- `contrastive_embedding`: `(256,)` — normalized projection used for contrastive/prototype losses.
- Do not use the 256-dimensional projection as the only saved downstream representation by default. Projection heads often specialize to the pretraining objective.

The two representations should be named separately in code and checkpoints.

## 8. Self-supervision: two-view change learning

Create two valid augmented views of the **same directed interval**. For each view, apply a physically plausible augmentation consistently to both dates within that view, then pass both dates through the shared backbone and Change Encoder.

The two views must preserve the same interval endpoints and temporal direction. Use consistent spatial crop geometry between the two dates within a view so the model does not mistake misregistration introduced by augmentation for glacier change. The two views can use different crop geometries from one another.

Suitable candidates for experiments:
- Moderate crop/resize with valid glacier coverage.
- Horizontal/vertical flips or small rotations only where geographically and scientifically appropriate.
- Mild sensor noise and radiometric perturbations.
- Cloud/invalid-region masking using valid masks.

Avoid:
- Independent random crop coordinates for the two dates of one interval.
- Strong color jitter that destroys multispectral relationships.
- Treating different seasons as automatically equivalent.
- Treating the reverse interval as an identical positive.
- Crop pairs where most of the glacier is absent or invalid.

The positive pair for the base objective is the same directed interval under two valid views. This gives a stable way to start without manually labelled change types. It alone does **not** guarantee that semantically similar changes from different glaciers will cluster together; Section 9 addresses that goal.

## 9. Learning similar-change groups

The diagram proposes a Change Encoder followed by clustering, pseudo-labels, and a contrastive loss. This is a reasonable research direction, but the cluster assignments are inferred targets and can be wrong. Use a staged experiment rather than assuming pseudo-labels are reliable from the first optimization step.

### Baseline objective

Start with a two-view instance-level contrastive loss on `contrastive_embedding` for the same directed interval. A symmetric InfoNCE loss is a straightforward baseline:

\[
\mathcal{L}_{\mathrm{inst}} =
\frac{1}{2}
\left[
\operatorname{InfoNCE}(z_a,z_b)
+
\operatorname{InfoNCE}(z_b,z_a)
\right].
\]

Here \(z_a,z_b\) are normalized projection embeddings of two views of the same interval. All invalid views are excluded. This objective learns augmentation invariance, but other intervals used as negatives can include genuinely similar changes. Therefore, it is a baseline, not the final answer to semantic grouping.

### Primary experiment: online prototypes / clustering

Compare the instance-level baseline with an online clustering-based objective such as SwAV-style prototype assignments, or a DeepCluster-style iterative procedure:

1. Compute interval embeddings from the training split only.
2. Assign embeddings to `K` learned prototypes or cluster centroids.
3. Use balanced or otherwise collapse-resistant assignments where supported by the selected method.
4. Train the model to predict the assignments for a second augmented view of the same interval.
5. Refresh assignments periodically; do not update pseudo-labels from validation/test data.
6. Monitor cluster occupancy and stability. Discard runs with collapsed or highly unstable assignments.

This encourages similar intervals to share a latent group without requiring hand-labelled retreat/advance categories. A cluster is **not** automatically a glacier event class. Cluster meaning must be assessed after training.

### Optional combined objective

If the individual objectives are stable, test:

\[
\mathcal{L}_{\mathrm{change}} =
\mathcal{L}_{\mathrm{view}} +
\lambda_{\mathrm{cluster}}\mathcal{L}_{\mathrm{prototype}}.
\]

- \(\mathcal{L}_{\mathrm{view}}\): same-interval two-view contrastive objective.
- \(\mathcal{L}_{\mathrm{prototype}}\): cross-view prototype-assignment objective.
- \(\lambda_{\mathrm{cluster}}\): tune experimentally; start with `0.5` after checking the individual loss scales, and compare `0.25`, `0.5`, and `1.0`.

Do not assume that adding losses improves representations. Compare each objective alone and in combination, with identical data splits and evaluation.

### Avoid a false-negative conflict

Do not simply add pseudo-label positives while retaining a standard InfoNCE denominator that treats every other interval—including same-cluster intervals—as a negative. That would directly oppose the goal of grouping similar changes. Use a prototype/assignment objective or a carefully designed supervised-contrastive loss where pseudo-labels define which pairs are positives and which should be excluded from negatives.

## 10. Temporal composition / evolution consistency — optional experiment

A possible additional objective is interval composition. For three dates \(i<j<k\), encode:

- \(c_{i\rightarrow j}\)
- \(c_{j\rightarrow k}\)
- \(c_{i\rightarrow k}\)

Test whether the long-interval representation can be predicted from the two shorter-interval representations:

\[
\hat c_{i\rightarrow k}
= g(c_{i\rightarrow j},c_{j\rightarrow k},\Delta t_{ij},\Delta t_{jk}),
\]

\[
\mathcal{L}_{\mathrm{comp}}
= \left\|\hat c_{i\rightarrow k}
- \operatorname{stopgrad}(c_{i\rightarrow k})\right\|_2^2.
\]

This is an **optional experiment**, not part of the minimum baseline. Glacier evolution is not necessarily additive in latent space: melting, snowfall, seasonal variation, and abrupt fragmentation can be nonlinear. The composition predictor must be learned rather than assuming that `c_ik = c_ij + c_jk`.

Compare this experiment only after the simpler contrastive/prototype objective is stable. Include a no-composition ablation. If the objective encourages collapse or merely predicts the time gap, remove it.

## 11. Loss and gradient flow

Initial three-head objective:

\[
\mathcal{L}_{\mathrm{total}}
= 1.0\,\mathcal{L}_{\mathrm{DINO}}
+ 0.5\,\mathcal{L}_{\mathrm{distance}}
+ 0.5\,\mathcal{L}_{\mathrm{change}}.
\]

These are starting coefficients from the existing training plan, not established optima. Log individual loss values and, if feasible, gradient norms reaching the shared ViT, HPE, and TET. Adjust weights only after confirming that each objective is implemented correctly and is learning.

Gradient flow for Head 3:
- Gradients flow through the Change Encoder and projection/prototype layers.
- Gradients flow through the TET regional outputs into HPE and the Multispectral ViT.
- If a teacher/stop-gradient branch is used by a chosen SSL method, follow that method's update rules.
- Pseudo-label generation itself should not backpropagate through discrete assignments.

For the first implementation, train one Head 3 objective at a time before combining objectives. Verify that the loss changes when the backbone is trainable and that gradients reach the expected modules.

## 12. Experiments to run

Only choices that cannot be justified from the existing interface should be treated as experiments. Keep the backbone output contracts fixed.

| ID | Question | Values to compare | Baseline |
|---|---|---|---|
| E1 | Which pair features best encode change? | Signed difference only; absolute difference only; `[start, end, signed difference]`; five-group feature | Five-group feature |
| E2 | What regional aggregation is useful? | Valid mean; mean + max; learned attention pooling | Valid mean |
| E3 | How should similar changes be grouped? | Instance InfoNCE; prototype assignment (SwAV-style); iterative DeepCluster; combined view + prototype loss | Run InfoNCE first, then prototype objective |
| E4 | How many prototypes/clusters? | `K ∈ {32, 64, 128, 256}` | `K=64` for the first clustering trial |
| E5 | Change Encoder capacity | `3840→768`; `3840→1536→768`; `3840→2048→1024→768` | `3840→1536→768` |
| E6 | Projection dimension | `128`, `256`, `512` | `256` |
| E7 | Pair-gap sampling | Adjacent-only; uniform valid pairs; gap-stratified pairs | Gap-stratified |
| E8 | Max interval gap | None; 3 years; 5 years; data-supported alternative | No cap initially, with gap-stratified reporting |
| E9 | Cluster loss weight | `λ_cluster ∈ {0.25, 0.5, 1.0}` | `0.5` after separate-loss checks |
| E10 | Temporal composition | Off; on with learned composition predictor | Off |
| E11 | Change loss coefficient | `λ_change ∈ {0.25, 0.5, 1.0}` | `0.5` in total loss |
| E12 | Augmentation strength | Mild; moderate | Mild |

Do not run the full Cartesian product. First establish a stable baseline, then vary one major design choice at a time. Use fixed validation splits and a fixed validation interval set for fair comparisons.

## 13. Evaluation and diagnostics

Because there are no manual event labels for SSL pretraining, evaluate both representation quality and training health.

### Training health
- Change loss and each component loss over time.
- Embedding variance and pairwise cosine-similarity distribution.
- Prototype/cluster occupancy, entropy, and stability across refreshes.
- Fraction of invalid intervals/views excluded.
- Gradient norms and NaN/Inf checks.
- Sensitivity to time-gap distribution.

### Representation evaluation
- Cluster interval embeddings and visualize them with UMAP/PCA, but do not use visual separation as proof of semantic validity.
- Inspect representative image pairs from each cluster, including short and long intervals.
- If expert annotations or independently derived change measurements become available, evaluate against them only after the representation is trained; keep the evaluation labels out of SSL training.
- Test linear probes or small downstream classifiers for event categories if trustworthy labels are available.
- Test cross-glacier and, where feasible, cross-region transfer.
- Compare with simple baselines: raw spectral-difference features, pooled absolute embedding differences, and a model trained without Head 3.
- Report results by interval-gap range and season, not only as a single aggregate.

A useful representation should organize intervals by visual/structural change patterns rather than merely by glacier identity, season, cloud coverage, or time-gap length.

## 14. Data leakage and scientific safeguards

1. Split by glacier identity before constructing interval pairs.
2. For a stronger zero-shot test, hold out entire geographic regions.
3. Fit normalization using training data only.
4. Do not provide the elapsed-time target, interval index, or pair ordering metadata as Change Encoder inputs.
5. Keep pseudo-label generation confined to the training split.
6. Exclude invalid/padded timestamps and regions from feature pooling and losses.
7. Do not call learned clusters “retreat”, “advance”, or “fragmentation” without independent validation.
8. Do not infer physical change magnitude directly from embedding distance unless calibrated and validated against a physical measurement.
9. Keep validation/test labels and observations out of hyperparameter selection where required by the experimental protocol.
10. Evaluate whether Head 3 improves the shared backbone using a full-model ablation.

## 15. Implementation checks

Before large-scale pretraining, verify:

1. TET `regional_out` has shape `(T, 64, 768)`.
2. The 64 regions correspond consistently across dates.
3. Interval endpoints are valid, ordered by acquisition date, and from the same glacier.
4. The region-wise pair feature has shape `(64, 3840)` for the baseline.
5. The Change Encoder returns `(64, 768)` and pooling returns `(768,)`.
6. The projection head returns a normalized `(256,)` vector.
7. Two augmented views preserve the same directed interval.
8. Invalid regions and padded timestamps never contribute to losses.
9. The interval gap is not passed to the Change Encoder.
10. Backpropagation reaches the expected shared-backbone parameters.
11. The instance-contrastive baseline can overfit a tiny controlled batch without numerical instability.
12. Prototype assignments do not collapse to one or a few clusters.
13. Validation and test glaciers never participate in training pseudo-label generation.
14. Reverse-interval diagnostics confirm that direction is preserved.

## 16. Final baseline specification

```text
Inputs:
    regional_i, regional_j: (64, 768)
    same glacier, valid dates, date_i < date_j

Per-region features:
    LN(r_i)
    LN(r_j)
    LN(r_j) - LN(r_i)
    abs(LN(r_j) - LN(r_i))
    LN(r_i) * LN(r_j)

Concatenate:
    (64, 3840)

Shared Change Encoder:
    LayerNorm(3840)
    Linear(3840, 1536)
    GELU
    Dropout(0.1)
    Linear(1536, 768)
    GELU
    LayerNorm(768)

Regional change tokens:
    (64, 768)

Validity-aware mean pooling:
    change_embedding: (768,)

Projection head:
    LayerNorm(768)
    Linear(768, 512)
    GELU
    Linear(512, 256)
    L2 normalize

contrastive_embedding:
    (256,)

Training:
    First establish two-view instance InfoNCE baseline.
    Then compare prototype/cluster-based objectives.
    Optional composition consistency only after baseline stability.

Initial loss coefficient:
    lambda_change = 0.5 in the combined three-head loss.
```

## 17. Final architectural decision

**Fix for the initial implementation:** use `regional_out`, directed region-pair features, a shared Change Encoder, validity-aware mean pooling, a separate projection embedding, and two-view learning for the same interval.

**Treat as experiments:** pair-feature ablations, pooling strategy, prototype count, clustering algorithm, contrastive/prototype loss combination, pair-gap sampling details, and optional temporal composition consistency.

This design follows the diagram's Change Encoder → contrastive learning / pseudo-label clustering idea while keeping the backbone and the responsibilities of Heads 1 and 2 intact. The architecture is a research hypothesis; its ability to discover meaningful glacier-evolution patterns must be demonstrated through ablations, held-out-glacier/region evaluation, and independent interpretation of the learned clusters.
