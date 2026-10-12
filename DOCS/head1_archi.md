# Head 1 Architecture — Temporal DINO

## 1. Purpose and scope

Head 1 learns robust representations of **glacier states** from multispectral temporal sequences. It should suppress nuisance variation while preserving glacier identity, morphology, spectral characteristics, glacier condition, and meaningful long-term evolution.

It is not the primary head for predicting elapsed time or explicitly clustering change events. Those responsibilities belong to Head 2 (Temporal Distance Prediction) and Head 3 (Temporal Change Representation).

**Training principle:** create different spatial views of the input sequence *before* the backbone. Each view passes through the complete spatial and temporal backbone. Do not crop the already-computed regional embeddings for the primary DINO objective.

## 2. Backbone interface and input contract

For one glacier sequence, let the input be:

- `X`: `(T, C, H, W)` — `T` timestamps, `C` multispectral bands, spatial dimensions `H × W`.
- Each timestamp has a real acquisition date.
- Missing/invalid pixels must be represented using the data pipeline's validity mask; padded timestamps, if any, must be excluded from the loss.

Each DINO view is a spatial crop/resize of the **whole temporal sequence**. Apply the same crop geometry to every timestamp within a view so that the crop refers to the same geographic area over time. Teacher and student views use different crop geometries. Do not independently crop every timestamp within one view: that would introduce artificial spatial misalignment.

Every view passes through the existing backbone:

`multispectral ViT → hierarchical patch embedding → Temporal Evolution Transformer`

The relevant transformer output for Head 1 remains the **timestamp embeddings**:

- Per view: `(T, 768)`
- DINO loss is computed on corresponding valid timestamp embeddings from teacher and student views, then averaged over valid timestamps and valid view pairs.
- The regional output `(T, 64, 768)` is not the direct input to this head.
- The sequence evolution embedding `(768)` is not the direct input to this head.

The temporal transformer may use the full sequence as context, so a timestamp embedding can contain contextual information from other observations. That is intentional. The teacher and student must receive views of the same underlying glacier sequence, with matching timestamp identities and dates.

## 3. Multi-crop view construction

Initial design:

| Network | Views | Crop scale | Purpose |
|---|---:|---|---|
| Teacher | 2 global views | Large fraction of the scene | Stable target with broad glacier/context information |
| Student | 2 global + 2 moderate local views | Global and glacier-centered local regions | Learn agreement across views and scales |

Crop sizes are resized to the spatial size required by the ViT. Local crops should retain meaningful glacier content; avoid crops consisting mostly of surrounding terrain. If GLIMS boundaries are used to guide crop sampling, use them only to guide view construction, not as a supervised target.

For each view, keep its crop coordinates consistent through time. Global teacher views and student views should be independently sampled, but each view must remain geographically aligned across its own temporal sequence.

**Important:** local views can omit parts of the glacier and may not be valid for every timestamp. Use a validity rule for crop/timestamp pairs and exclude invalid pairs from the loss rather than treating them as positives. The exact minimum glacier coverage and crop-scale range should be validated experimentally.

## 4. Projection architecture

Use the same projection architecture for student and teacher; their parameters differ.

### Timestamp embedding → projector

`768 → Linear(2048) → GELU → Linear(2048) → GELU → Linear(1024) → bottleneck`

- Input dimension: `768`
- Hidden dimension: `2048`
- Projector bottleneck dimension: `1024`
- Activation: `GELU`
- Student and teacher projectors have identical architecture.
- Teacher parameters are updated by EMA, not by gradient descent.

### DINO prototype/logit layer

A standard DINO head also maps the projector bottleneck to a distribution over learned prototypes. Treat the `1024` dimension above as the **bottleneck**, not as the final number of DINO classes/logits.

`1024 → weight-normalized linear layer → K prototype logits`

- Use weight normalization on the final prototype layer.
- `K` is the number of prototypes. Start with `K = 8192` as a practical initial setting; compare against a larger value only if training evidence warrants it.
- Apply softmax to prototype logits using the relevant student/teacher temperature.
- Do not L2-normalize the prototype logits. If the chosen DINO implementation normalizes the bottleneck before the prototype layer, follow that implementation consistently.

The projection/prototype layers are pretraining components. They are normally discarded after SSL pretraining; the backbone is retained for downstream use.

## 5. DINO objective

For each valid timestamp and view, let `z_s` and `z_t` be student and teacher prototype logits.

Teacher target distribution:

`p_t = softmax((z_t - c) / τ_t)`

Student distribution:

`p_s = softmax(z_s / τ_s)`

Here `c` is the running teacher-logit center, `τ_t` is teacher temperature, and `τ_s` is student temperature.

Cross-entropy for a teacher/student view pair:

`L_DINO = - Σ_k p_t[k] log(p_s[k])`

Use the standard DINO cross-view objective:
- Teacher processes the two global views.
- Student processes all four views.
- Compare each teacher global-view target against student views **excluding the matching same-view pair**.
- Average over valid view pairs and valid timestamps.
- Use stop-gradient for teacher targets.
- Update the teacher with EMA; do not backpropagate gradients through the teacher.

This objective teaches consistency between different views of the same underlying observations. It does not use explicit negative samples in the InfoNCE sense.

## 6. What should and should not be invariant?

Encourage invariance to:
- Moderate spatial crop/scale differences that retain the same glacier state.
- Flips and small rotations where scientifically appropriate.
- Mild radiometric/illumination variation and realistic sensor noise.
- Minor appearance changes that do not indicate meaningful glacier evolution.

Do **not** force invariance to:
- Different glacier identities.
- Major retreat/advance, fragmentation, disappearance, or substantial morphology changes.
- Large temporal gaps as though they necessarily represented the same state.

Seasonality needs care: seasonal snow and illumination can be nuisance variation, but snowline/ice changes can also be physically meaningful. Do not apply aggressive augmentations that erase spectral or seasonal evidence. The temporal heads provide additional learning signals, but they do not guarantee that DINO will preserve every meaningful change; this must be tested with ablations and downstream probes.

## 7. Initial hyperparameters

| Parameter | Initial value | Status |
|---|---:|---|
| Input to Head 1 | Timestamp embeddings `(T, 768)` per view | Fixed interface |
| Projector | 3-layer MLP | Initial fixed choice |
| MLP hidden dimension | `2048` | Initial fixed choice |
| Projector bottleneck | `1024` | Initial fixed choice |
| Activation | GELU | Fixed initial choice |
| Prototype count `K` | `8192` | Tune only if needed |
| Student temperature `τ_s` | `0.1` | Initial value |
| Teacher temperature `τ_t` | `0.04` | Initial value; monitor stability |
| Teacher centering | EMA running center | Enabled |
| Teacher update | EMA | Enabled |
| EMA momentum | Cosine schedule `0.996 → 0.9999` | Initial schedule |
| Teacher/student architecture | Identical | Fixed |
| Views | 2 global teacher; 2 global + 2 moderate local student | Initial design |
| Batch size | Determined by GPU memory | Experiment |
| Learning rate | Tune with effective batch size and backbone optimizer setup | Experiment |
| Weight decay | AdamW starting point `0.04`, scheduled if appropriate | Initial value; validate |
| Head loss coefficient | `λ_DINO = 1.0` | Initial combined-loss weight |

The teacher temperature, prototype count, crop scales, and learning rate are sensible starting values, not guaranteed optima for glacier sequences. Keep a small, controlled tuning budget rather than changing many variables simultaneously.

## 8. Combined objective and gradient flow

Initial combined loss:

`L_total = 1.0 × L_DINO + 0.5 × L_distance + 0.5 × L_change`

The coefficients for the other heads are initial values and should be checked for scale/gradient imbalance. Log each loss and, where practical, gradient norms reaching the shared backbone. Do not assume that a coefficient of `1.0` means DINO contributes most strongly; raw loss magnitudes can differ.

Student DINO loss backpropagates through the student projector and the shared backbone. The teacher receives no gradients and is updated from the student backbone/projector by EMA. Other SSL heads train their respective paths according to their own definitions.

## 9. Validation checks before full training

1. Verify that each view has shape `(T, C, H, W)` and crop geometry is consistent across its timestamps.
2. Verify backbone output for each view has timestamp embeddings `(T, 768)`.
3. Verify student and teacher prototype logits have shape `(T, K)` for each view.
4. Confirm teacher parameters receive EMA updates but no gradients.
5. Confirm same-view teacher/student pairs are excluded from the cross-view DINO loss.
6. Exclude padded timestamps and invalid crop/timestamp pairs from loss reduction.
7. Monitor teacher entropy, prototype usage, loss curves, embedding variance, and NaNs to detect collapse or instability.
8. Run a small overfit/sanity run before scaling to the full glacier dataset.

## 10. Research note: latent-region cropping

Cropping the transformer's regional embeddings after the backbone is **not** the primary DINO design in this specification. It may be investigated later as a separate latent-region consistency objective and compared against crop-first DINO. Do not mix the two approaches in the first implementation, because that would make the effect of each objective difficult to interpret.
