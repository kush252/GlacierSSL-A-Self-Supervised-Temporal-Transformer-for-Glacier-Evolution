# Head 2 Architecture — Temporal Distance Prediction

**Status:** Proposed baseline architecture with explicitly defined experiments for unresolved choices.  
**Role:** Learn a representation that preserves information about the elapsed time between two observations of the same glacier.  
**Backbone:** Multispectral ViT → Hierarchical Patch Embedding (HPE) → Temporal Evolution Transformer (TET).  
**Training:** Self-supervised; timestamp acquisition dates provide automatically generated targets, not manual glacier-evolution labels.

---

## 1. Purpose and boundaries

Head 2 asks:

> Given two observations of the same glacier, can the learned representations indicate how much real time separates them?

Examples:
- Two observations 3 months apart.
- Two observations 1 year apart.
- Two observations 5 years apart.

The target is **elapsed time between observations**, not geographic distance between glaciers. Glacier identity is held constant for the primary objective. Comparing different glaciers would mix temporal distance with glacier identity and morphology, so cross-glacier pairs are not part of the baseline.

The head should encourage the shared backbone to encode temporal scale and the visual consequences that may accumulate over time. It must not be interpreted as a direct physical measurement of retreat, ice loss, or change magnitude. Two dates far apart can show little visible change; two dates close together can show substantial change. Head 3 is responsible for learning temporal-change representations.

### Division of responsibilities

| Component | Primary role |
|---|---|
| Head 1 — Temporal DINO | Consistent representations across valid views of glacier observations, while retaining meaningful state information |
| **Head 2 — Temporal Distance** | Predict elapsed time between two observations of the same glacier |
| Head 3 — Temporal Change Representation | Represent changes between observations and learn relationships among change intervals |
| Shared ViT + HPE + TET | Learn features used by all three objectives |

Head 2 must not duplicate Head 3 by predicting a change-event class or a retreat/advance magnitude. It predicts the time interval only.

---

## 2. Backbone contract

For one glacier, observations are chronologically ordered and have actual acquisition dates.

```text
Input at each timestamp
(T, 6, 512, 512)
        |
        v
Multispectral ViT
(T, 1024, 768)
        |
        v
Hierarchical Patch Embedding
(T, 64, 768)
        |
        v
Temporal Evolution Transformer
        |
        +--> regional_out:  (T, 64, 768)
        +--> timestamp_out: (T, 768)
        +--> sequence_out:  (768,)
```

Head 2 uses **`timestamp_out`**, not the global sequence embedding and not a pooled representation of the entire glacier sequence.

- `T` is variable; 36 observations is a possible archive pattern, not a fixed model requirement.
- Timestamps must remain chronologically sorted.
- Actual dates, not array indices, determine elapsed time.
- Padded or invalid timestamps must be excluded from pair sampling and loss calculation.
- The current TET contract defines `timestamp_out[t]` as the masked mean of the 64 regional output tokens at timestamp `t`.

### Important limitation

TET timestamp embeddings are temporally contextualized and receive explicit elapsed-time encodings. A distance predictor might therefore infer the target partly from temporal metadata rather than from visual representations. This is a potential shortcut, not something to ignore. Section 10 defines tests for it.

---

## 3. Pair construction and target generation

For timestamp indices `i` and `j` belonging to the same glacier, with acquisition dates `d_i` and `d_j`:

```text
delta_years(i, j) = abs(d_j - d_i).days / 365.2425
```

Use `i != j`. The target is calculated from real dates, including irregular observation gaps. Do not calculate it as `abs(i - j)`, because observations may be missing or irregularly spaced.

### Pair-sampling rules

1. Sample two valid timestamps from the **same glacier sequence**.
2. Sample a mix of short, medium, and long temporal gaps.
3. Avoid allowing the abundant short-gap pairs to dominate training.
4. Exclude duplicate dates unless duplicate acquisitions are deliberately being handled as a separate data-quality case.
5. Exclude pairs where either timestamp fails the pipeline's image-quality validity rule.
6. Keep the two timestamps' original dates and ordering metadata for target creation and auditing.
7. Split validation by glacier identity, not by randomly splitting pairs from the same glacier across train and validation. Otherwise, the same glacier may leak into both splits.

A pair is one training example, but the two timestamp embeddings are produced by the shared backbone. The pair sampler must not treat two different glaciers as a positive temporal-distance pair in the baseline.

### Sampling strategy to compare

- **Experiment A — Uniform timestamp-pair sampling:** sample valid pairs uniformly from available pairs. Simple baseline, but likely dominated by the distribution of available short gaps.
- **Experiment B — Gap-stratified sampling (recommended baseline):** sample a target gap range first, then sample a pair from that range. This gives the model exposure to both short and long intervals.
- **Experiment C — Weighted sampling:** sample pairs naturally, but weight the loss inversely by the frequency of the target gap region.

Start with B. Keep the validation and test pair distribution representative of the real dataset; do not rebalance the test set without also reporting results on the natural distribution.

---

## 4. Recommended prediction formulation

Use **continuous elapsed-time regression as the primary objective**. It preserves the distinction between, for example, a 1-year gap and a 1.5-year gap without requiring arbitrary class boundaries.

Let the TET timestamp embeddings be:

```text
z_i = timestamp_out[i]  # (768,)
z_j = timestamp_out[j]  # (768,)
```

Normalize each embedding with LayerNorm before constructing pair features. Do not L2-normalize automatically in the baseline; compare that choice in an experiment.

### Pair-feature construction

Use a symmetric pair representation so swapping the two timestamps does not change the predicted absolute interval:

```text
u_i = LayerNorm(z_i)
u_j = LayerNorm(z_j)

pair_features = concat(
    abs(u_i - u_j),   # 768
    u_i * u_j         # 768
)
```

Result:

```text
pair_features: (1536,)
```

The absolute difference captures feature discrepancy; the elementwise product captures shared feature activation. Symmetry is appropriate because the target is an absolute time gap. Do not concatenate the raw acquisition dates or `delta_years` into the predictor input: that would expose the answer directly.

### Regression MLP

```text
Input: (1536,)
    |
LayerNorm(1536)
    |
Linear(1536 -> 512)
    |
GELU
    |
Dropout(p = 0.1)
    |
Linear(512 -> 128)
    |
GELU
    |
Linear(128 -> 1)
    |
Softplus
    |
Predicted elapsed time in years: (1,)
```

Softplus keeps the prediction non-negative. It does not impose an upper bound; predictions should be evaluated for calibration and extreme-gap errors.

**Initial fixed baseline choices**
- Pair representation: concatenation of absolute difference and elementwise product.
- MLP widths: `1536 → 512 → 128 → 1`.
- Activation: GELU.
- Output activation: Softplus.
- Dropout: `0.1`.
- Primary target: elapsed time in years.
- Primary loss: Smooth L1 / Huber loss on a transformed target, as specified below.

These are engineering starting points, not experimentally established optimal values.

---

## 5. Target transform and loss

Temporal gaps are likely to be skewed: many pairs may be separated by months, while fewer pairs span several years. Regressing directly in years can make long gaps dominate the error scale.

Use the transformed target:

```text
y = log(1 + delta_years)
y_hat = log(1 + predicted_delta_years)
```

For implementation simplicity, a cleaner equivalent is to predict a non-negative gap in years and compare transformed values:

```text
L_distance = SmoothL1(
    log1p(predicted_delta_years),
    log1p(target_delta_years)
)
```

The prediction must be clamped to a small non-negative minimum only if required for numerical safety; `Softplus` already guarantees non-negativity. The log transform reduces the influence of very long gaps and emphasizes relative error across the range.

**Implementation note:** verify gradients and units carefully. The loss is computed in transformed space, while reported predictions and metrics must be converted back to years using `expm1`.

### Loss alternatives to test

- **Huber/Smooth L1 on `log1p(gap)` — recommended baseline:** robust to outliers and balances short/long gaps.
- **MAE on `log1p(gap)`:** more robust to extreme errors, but less smooth at zero error.
- **MSE on `log1p(gap)`:** smooth, but can be more sensitive to residual outliers.
- **MSE or Huber directly in years:** interpretable units, but can overemphasize large gaps.

Do not tune the regression loss and pair-sampling strategy simultaneously in the first experiment. Change one major variable at a time.

---

## 6. Optional alternative: gap-bin classification

The diagram's examples (“1 year gap”, “5 year gap”) could be implemented as classes, but classification loses exact temporal information. Keep it as an alternative experiment rather than the primary architecture.

Possible bins for a first test:

| Class | Actual elapsed gap |
|---|---|
| 0 | `0 < gap ≤ 0.5` years |
| 1 | `0.5 < gap ≤ 1` year |
| 2 | `1 < gap ≤ 2` years |
| 3 | `2 < gap ≤ 4` years |
| 4 | `gap > 4` years |

These thresholds are provisional. Inspect the real gap distribution before finalizing them. If a class has too few examples, revise the bins using training-set quantiles or combine classes. Freeze the bin boundaries before validation/test evaluation.

For classification, use the same pair features and replace the last layer with `Linear(128 → 5)`, followed by cross-entropy. Consider class-balanced sampling or weights if the training distribution is imbalanced. Report macro-F1 and balanced accuracy in addition to accuracy.

Do not train both regression and classification losses by default. First compare them as separate controlled experiments. A joint regression-plus-classification auxiliary loss is a later experiment only if evidence supports it.

---

## 7. How Head 2 trains the shared backbone

```text
Timestamp i image ──> shared ViT + HPE + TET ──> z_i (768)
                                                     \
                                                      > Pair features
                                                     /      |
Timestamp j image ──> shared ViT + HPE + TET ──> z_j       v
                                                        Regression MLP
                                                             |
                                                     predicted time gap
                                                             |
                                                      distance loss
```

- Both timestamps use the same encoder weights.
- The distance loss backpropagates through the pair MLP and both timestamp embeddings into the shared backbone.
- The head-specific MLP is discarded after SSL pretraining unless a downstream use specifically needs it.
- The pretrained ViT, HPE, and TET are retained.
- The three SSL losses jointly train the shared backbone, with separate task-specific head parameters.

Initial combined objective:

```text
L_total = 1.0 * L_DINO
        + 0.5 * L_distance
        + 0.5 * L_change
```

The coefficients `0.5` for Head 2 and Head 3 are initial values. Check each loss magnitude and, where practical, the gradient norms reaching the shared backbone. A coefficient alone does not establish how much influence a head has.

---

## 8. Critical shortcut checks

The TET intentionally receives actual elapsed-time encodings. Since Head 2 predicts elapsed time, the model could solve the task by reading the explicit temporal encoding rather than learning useful temporal features from imagery. The model might also exploit sequence length or pair-sampling artifacts.

Run these checks before claiming that Head 2 learns visual temporal-distance representations.

### Experiment D — Standard TET features

Use the normal `timestamp_out` embeddings. This is the straightforward end-to-end baseline and allows the head to exploit all learned contextual information. Interpret its results cautiously until the following controls are run.

### Experiment E — Visual-feature-only control

Run a controlled variant in which the distance head receives representations that do not expose explicit date/gap information to its prediction path. Possible implementations include:
- Pool the HPE/ViT features per timestamp before adding TET temporal encodings, then use a small shared projection to obtain 768-dimensional timestamp features; or
- Re-run a controlled TET variant without explicit elapsed-time embeddings for the Head 2 path.

The first option more cleanly tests whether spatial-spectral appearance alone carries elapsed-gap signal; the second tests how much TET's explicit temporal encoding contributes. These are diagnostic variants, not replacements for the agreed shared-backbone baseline without further evidence.

### Experiment F — Metadata-only baseline

Predict elapsed time using only metadata that might accidentally correlate with the target, such as timestamp index difference, number of missing observations, or sequence length. Do not provide actual `delta_years` as a feature. This baseline identifies whether pair construction makes the task artificially easy.

### Experiment G — Pair-order invariance

Swap `i` and `j`. Because the target is an absolute interval and the pair features are symmetric, the prediction should remain numerically identical (up to floating-point tolerance). Add a unit test for this property.

**Interpretation:** a strong standard score but weak visual-feature-only score suggests the task is being solved primarily through temporal metadata. That result can still validate implementation, but it is not sufficient evidence that the backbone learned visual temporal distance.

---

## 9. Experiments and values to compare

Do not search a large grid initially. Establish a working baseline and then run controlled ablations.

| ID | Question | Values |
|---|---|---|
| E1 | Pair sampling | Uniform pairs; gap-stratified pairs; frequency-weighted loss |
| E2 | Prediction target | Regression in years; regression on `log1p(gap)`; gap-bin classification |
| E3 | Pair features | `abs(u_i-u_j) + u_i*u_j`; absolute difference only; concatenation `[u_i, u_j, abs difference]` |
| E4 | MLP capacity | `1536→512→128→1`; `1536→256→64→1`; `1536→1024→256→1` |
| E5 | Embedding normalization | LayerNorm; L2 normalization; LayerNorm plus L2 normalization |
| E6 | Loss | Smooth L1 on transformed target; MAE on transformed target; MSE on transformed target |
| E7 | Temporal shortcut | Standard TET features; visual-feature-only control; metadata-only baseline |
| E8 | Gap range | All valid gaps; capped maximum gap; separate short/medium/long evaluation |
| E9 | Head loss coefficient | `λ_distance ∈ {0.25, 0.5, 1.0}` after establishing loss scales |
| E10 | Cross-glacier evaluation | Hold out entire glaciers; optionally hold out geographic regions for stronger transfer testing |

Recommended order:
1. Implement the baseline and pair-symmetry tests.
2. Verify data leakage controls and split by glacier identity.
3. Compare pair sampling and regression/classification formulation.
4. Run shortcut diagnostics.
5. Tune MLP capacity and loss coefficient only after the objective is demonstrably useful.

Use the same fixed validation pairs across architecture experiments where possible, and never tune on test results.

---

## 10. Evaluation

Report both the primary regression metrics and diagnostics by gap range.

### Regression metrics

- **MAE (years):** mean absolute error after converting predictions back to years.
- **RMSE (years):** penalizes large errors.
- **Spearman correlation:** tests whether predicted and true intervals are ranked consistently.
- **Calibration plot:** predicted versus actual gap, grouped into bins.
- **Per-gap-range MAE:** short, medium, and long intervals.

### Classification metrics (only if classification is tested)

- Accuracy.
- Balanced accuracy.
- Macro-F1.
- Confusion matrix.

### Representation and learning diagnostics

- Training and validation loss.
- Predicted-gap distribution versus target-gap distribution.
- Error versus actual gap.
- Performance on held-out glacier identities and, if feasible, held-out geographic regions.
- Comparison against a simple metadata-only baseline.
- Comparison against visual-feature-only features.

A low regression error alone does not establish scientific usefulness. Evaluate whether the objective improves representations for downstream glacier-evolution tasks, and ablate Head 2 from the full three-head system.

---

## 11. Data leakage and scientific safeguards

1. Split by glacier identity before generating train/validation/test pairs. Never split pairs randomly after generating them from all glaciers.
2. Fit normalization statistics using the training split only.
3. Do not pass `delta_years`, date differences, or pair indices directly into the prediction MLP.
4. Do not treat pair order as meaningful for absolute-distance regression.
5. Exclude invalid/padded timestamps from pair creation and losses.
6. Do not assume longer time means greater glacier change.
7. Do not claim Head 2 measures retreat, advance, fragmentation, or ice loss.
8. Report performance separately for different gap ranges; a model can look good on frequent short gaps while failing on long gaps.
9. Keep the test set untouched until architecture and hyperparameters are fixed.
10. Compare with a metadata-only baseline to quantify the effect of temporal sampling shortcuts.

---

## 12. Implementation checks

Before full-scale pretraining, verify:

1. `timestamp_out` has shape `(T, 768)`.
2. Every pair uses two valid timestamps from the same glacier.
3. The target is computed from actual acquisition dates in years.
4. Pair features have shape `(1536,)`.
5. The MLP returns one non-negative scalar per pair.
6. Swapping pair order leaves the prediction unchanged.
7. Padded timestamps never enter a pair or loss.
8. Backpropagation from `L_distance` reaches the intended shared-backbone parameters.
9. No raw date gap or timestamp-index difference enters the predictor.
10. Validation is grouped by glacier identity.
11. Regression predictions are converted back from `log1p` space before reporting year-based metrics.
12. The visual-feature-only and metadata-only controls run successfully.

---

## 13. Final baseline specification

```text
Input:
    z_i, z_j = TET timestamp embeddings, each (768,)
    i and j are different valid timestamps of the same glacier

Pair features:
    u_i = LayerNorm(z_i)
    u_j = LayerNorm(z_j)
    p = concat(abs(u_i - u_j), u_i * u_j)
    p shape = (1536,)

Prediction MLP:
    LayerNorm(1536)
    Linear(1536, 512)
    GELU
    Dropout(0.1)
    Linear(512, 128)
    GELU
    Linear(128, 1)
    Softplus

Target:
    delta_years = abs(date_j - date_i).days / 365.2425

Loss:
    SmoothL1(log1p(predicted_delta_years), log1p(delta_years))

Sampling:
    Gap-stratified same-glacier pairs

Validation:
    Entire glacier identities held out

Experiments:
    Pair sampling, target formulation, pair features, normalization,
    MLP size, loss, temporal shortcut controls, loss coefficient
```

This is the recommended starting architecture. Its core interface and pair-symmetric feature design can be fixed for implementation; the target formulation, pair sampling, and temporal shortcut behavior must be validated empirically.
