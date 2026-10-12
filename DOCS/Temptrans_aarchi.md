# Temporal Evolution Transformer (TET) — Architecture Specification

**Status:** Core architecture frozen; continuous-time encoding implementation remains configurable.  
**Scope:** TET input contract, temporal metadata, factorized attention, hyperparameters, and outputs. SSL-head internals and upstream ViT/HPE internals are out of scope.

## 1. Input representation and temporal embeddings

### Input contract

```text
Multispectral ViT output: (T, 1024, 768)
HPE output:               (T, 64, 768)
TET input:                (T, 64, 768)
```

- `T` is variable, not fixed at 36. The archive may contain 36 seasonal observations, but training should sample chronological windows of varying lengths.
- Each timestamp has 64 regional tokens, each of dimension 768.
- Timestamps must be sorted chronologically. Never shuffle observations within a sequence.
- Each observation must have acquisition-date metadata, or a consistently defined representative date for a seasonal composite. Compute temporal metadata from dates, not list positions.
- Variable-length architecture does not guarantee reliable predictions from arbitrarily short sequences; downstream reliability depends on the training distribution and task.

### Token-level temporal embedding

For timestamp `t` and region `r`:

```text
x[t, r] = HPE[t, r]
          + SeasonEmbedding(season[t])
          + ElapsedTimeEncoding(elapsed_years[t])
```

Output shape remains `(T, 64, 768)`.

**Season embedding**
- Four IDs: `DJF`, `MAM`, `JJA`, `SON`.
- Month mapping: DJF = December–February; MAM = March–May; JJA = June–August; SON = September–November.
- Use a learnable embedding table with 4 entries of dimension 768.
- Season is per observation. The model must not require all four seasons to be present every year.

**Elapsed-time encoding**
- Scalar input: actual elapsed time in years from the first observation in the current input window.
- Compute from acquisition dates using a consistent convention (e.g. elapsed days / 365.2425). For seasonal composites, use the documented representative date consistently.
- Do not use absolute calendar-year embeddings or fixed 36-position embeddings.
- This value encodes order and irregular gaps. Example: `0.00, 0.25, 0.50, 0.75, 2.00, 2.25`; the gap after `0.75` must remain visible.
- The scalar-to-768 encoding method is intentionally configurable for later ablation: continuous sinusoidal encoding, a small learnable MLP, or linear projection. Keep a common interface `ElapsedTimeEncoding(scalar) -> (768,)`.

### Sequence-level context token

Initialize one token per input window:

```text
sequence_token_0 = LearnableSequenceToken + SequenceSpanEncoding(span_years)
```

- `LearnableSequenceToken` is a trainable 768-dimensional vector.
- `span_years` is the elapsed time between the first and last observations. For a single observation, span is zero, though one observation may not be useful for an evolution task.
- `SequenceSpanEncoding` maps the scalar span to 768 dimensions. Initially use the same *encoding family* as elapsed-time encoding for consistency; weights/parameters may be separate.
- The token is updated through TET blocks and its final hidden state becomes the **Sequence Evolution Embedding**.
- The context token is global, not a physical region; it must not participate in spatial attention.

**Do not confuse:**
- Elapsed time = one value per timestamp, measured from the first timestamp in the current window.
- Sequence span = one value for the whole window.
- Neither uses absolute calendar year.

## 2. Factorized temporal-spatial attention design

Use factorized attention, not full attention across all `T × 64` regional tokens. Each block performs temporal attention, spatial attention, then an MLP.

### Temporal attention

Purpose: model each region's evolution through time and update the global sequence context.

- Regional input: `(T, 64, 768)`.
- Rearrange to `(64, T, 768)`, so each of the 64 regions forms a temporal sequence.
- Apply one shared Temporal Multi-Head Self-Attention module to all 64 regional sequences. Do not create 64 separate transformers.
- The sequence context token participates in temporal reasoning, but it cannot be naively appended to `(64, T, 768)`: it is global and is not associated with one region. Its communication path must be explicit in the implementation.
- Possible implementation: broadcast the context token into each region's temporal sequence, then update it from temporal outputs using a documented aggregation (e.g. masked mean); alternatively, use a dedicated context-token attention path that explicitly communicates with regional temporal attention. Choose and unit-test one approach before full training. Do not silently flatten all `T × 64` regional tokens into one full-attention sequence.
- Apply Pre-LayerNorm before temporal attention and a residual connection after it.
- Reshape regional outputs back to `(T, 64, 768)`.

### Spatial attention

Purpose: model relationships among the 64 regions at each timestamp.

- Input: `(T, 64, 768)`.
- Apply self-attention independently to each timestamp over its 64 region tokens.
- The context token does **not** participate in spatial attention.
- Apply Pre-LayerNorm before spatial attention and a residual connection after it.

### Feed-forward network (MLP)

Applied after spatial attention to regional tokens, and to the context token after its temporal update:

```text
LayerNorm
Linear(768 -> 3072)
GELU
Dropout
Linear(3072 -> 768)
Dropout
Residual connection
```

Weights are shared across positions within a block, but each transformer block has its own MLP parameters.

### One block, conceptually

```text
Regional input: (T, 64, 768)
Context input:  (768,)

1. Temporal Pre-LN -> temporal attention -> residual
   - regions arranged as (64, T, 768)
   - context interaction follows an explicitly implemented path
2. Spatial Pre-LN -> spatial attention -> residual
   - regions arranged as (T, 64, 768)
   - context token excluded
3. MLP Pre-LN -> MLP -> residual
   - applied to regional tokens and to the updated context token

Outputs:
  regional tokens: (T, 64, 768)
  context token:   (768,)
```

Repeat for 8 blocks. Each block has independent temporal attention, spatial attention, LayerNorms, and MLP weights.

## 3. Hyperparameters

| Setting | Value | Notes |
|---|---:|---|
| Embedding dimension | 768 | Matches upstream representation; no TET boundary projection needed. |
| Number of blocks | 8 | Initial design choice; validate empirically. |
| Attention heads | 12 | For both temporal and spatial attention. |
| Dimension per head | 64 | `768 / 12`. |
| MLP hidden dimension | 3072 | 4× expansion. |
| MLP activation | GELU | Between the two linear layers. |
| Normalization | Pre-LayerNorm | Separate LayerNorm before temporal attention, spatial attention, and MLP. |
| Residual connections | Standard | One around each attention submodule and one around the MLP. |
| MLP dropout | 0.1 | Keep configurable. |
| Attention dropout | 0.1 | On attention probabilities; projection dropout may be configured separately. |
| Weight sharing across blocks | No | Each block has independent parameters. |
| Weight sharing across positions | Yes, within each module | Same module applied across regions/timestamps; not a module per region. |
| Relative temporal attention bias | Not finalized | Exclude from baseline unless explicitly enabled as an ablation. |

These are starting hyperparameters, not claims of proven optimality.

## 4. Outputs

The final TET block exposes three representations. Do not put task-specific SSL projection heads inside TET.

### A. Regional contextual embeddings

```text
regional_out: (T, 64, 768)
```

Final regional tokens after temporal attention, spatial attention, and MLP. Preserve timestamp and region ordering. Intended for the Temporal Change SSL objective and region-level downstream tasks.

### B. Timestamp embeddings

```text
timestamp_out: (T, 768)
timestamp_out[t] = Pool(regional_out[t, :, :])
```

Pool the 64 regional tokens independently at each timestamp. Baseline: mean pooling; use masked mean if padded/invalid regions exist. Intended for Temporal Distance Prediction and timestamp-level downstream tasks.

### C. Sequence Evolution Embedding

```text
sequence_out: (768,)
sequence_out = final_sequence_context_token
```

Use the final context-token hidden state directly. Do **not** derive this output by pooling timestamp embeddings. Intended for Temporal DINO and sequence-level downstream tasks.

## Training and inference notes

- Training should expose TET to variable-length chronological windows and occasional missing timestamps. Variable-T support alone does not ensure robustness to missing observations.
- If batching different sequence lengths, pad to the batch maximum and use correct masks. Padding must not affect attention, pooling, span calculation, or SSL losses.
- Do not insert synthetic timestamps just to complete four seasons per year.
- Future calendar years work without a year-embedding lookup table. Generalization to gaps much longer than those seen in training must be evaluated rather than assumed.
- For downstream tasks, retain the pretrained Multispectral ViT + HPE + TET, remove pretraining-only SSL heads, and attach a task-specific head to the appropriate representation.

## Implementation item that must be resolved explicitly

The context token's communication path through factorized temporal attention is the main remaining tensor-flow ambiguity. A global token cannot simply be appended to `(64, T, 768)` without defining its relationship to the 64 regional sequences. Choose a concrete update/communication mechanism and unit-test it before full training.

Minimum tests:
1. Temporal attention input/output shapes and masks.
2. Context-token update and communication.
3. Spatial attention over exactly 64 regions, with context token excluded.
4. Variable-length and padded batches.
5. Output shapes and timestamp pooling.
