# Hierarchical Patch Embedding (HPE)

## 1. Purpose and scope

The Hierarchical Patch Embedding (HPE) module converts the Spatial ViT's
dense patch-token grid into a smaller grid of regional tokens for the
Temporal Evolution Transformer (TET).

Its responsibilities are to: - reduce spatial token count while
retaining useful glacier features; - aggregate neighboring patches in a
spatially aligned manner; - preserve the 2D layout and a consistent
regional grid across timestamps; - keep the embedding dimension at 768
so the output matches the TET input contract.

HPE does **not** model temporal evolution, compare different timestamps,
predict temporal distance, or compute SSL losses. Each timestamp is
processed independently. Temporal reasoning belongs to TET and the SSL
heads.

This design uses two stages of learned 2×2 patch merging, inspired by
the patch-merging operation used in hierarchical vision transformers. It
is an architecture choice to validate, not a claim that this is the
empirically optimal design for glaciers.

## 2. Input contract

For one glacier sequence:

``` text
ViT output: (T, 1024, 768)
```

-   `T`: number of chronological observations in the sequence; variable,
    not assumed to always equal 36.
-   `1024`: spatial tokens from a 32×32 grid.
-   `768`: feature dimension per token.
-   The ViT uses 16×16 image patches for 512×512 input imagery.
-   No CLS token is present.
-   The input tokens retain their original row-major spatial ordering.

Before merging, reshape the token sequence to:

``` text
(T, 32, 32, 768)
```

The HPE treats the first dimension as a batch of independent timestamps.
It must not merge tokens across timestamps.

## 3. Output contract

``` text
HPE output: (T, 64, 768)
```

This corresponds to an 8×8 regional grid at each timestamp.

-   64 regional tokens per timestamp.
-   768 features per regional token.
-   Row-major spatial ordering is retained.
-   The same spatial location in the 8×8 grid has the same grid index
    across timestamps.
-   The output is passed to TET without temporal pooling or
    sequence-level aggregation.

For a batch of `B` glacier sequences padded to a common temporal length,
the corresponding tensor can be represented as `(B, T, 64, 768)`.
Padding and timestamp-validity masks remain the responsibility of the
data pipeline and TET.

## 4. Architecture overview

``` text
Input from Spatial ViT
(T, 1024, 768)
        |
        v
Reshape to (T, 32, 32, 768)
        |
        v
Patch Merge 1
2×2 neighboring tokens → 1 token
        |
        v
(T, 16, 16, 768)
        |
        v
Patch Merge 2
2×2 neighboring tokens → 1 token
        |
        v
(T, 8, 8, 768)
        |
        v
Add learned 2D regional positional embedding
        |
        v
Output to TET
(T, 64, 768)
```

There are two merging stages, not three. The 8×8 output is the agreed
interface; an additional 4×4 stage would reduce the output below the
required 64 regional tokens.

## 5. Patch-merging operation

### 5.1 Neighborhood selection

For each 2×2 neighborhood, collect four spatially adjacent token vectors
in a fixed order:

``` text
x00  x01
x10  x11
```

Here, `x00`, `x01`, `x10`, and `x11` each have 768 features. Concatenate
them in the fixed order top-left, top-right, bottom-left, bottom-right:

``` text
concat(x00, x01, x10, x11)
```

The concatenated feature has dimension `4 × 768 = 3072`.

Keeping the order fixed allows the projection to learn different
relationships for the four relative positions. The operation is applied
to non-overlapping 2×2 neighborhoods with stride 2.

### 5.2 Normalization and projection

Each merging stage applies:

``` text
Four neighboring tokens
        |
        v
Concatenate: 3072 features
        |
        v
LayerNorm(3072)
        |
        v
Linear(3072 → 768)
        |
        v
One regional token (768 features)
```

The projection is learned during training. It is not fixed average
pooling. It can learn which combinations of neighboring features are
useful for the representation.

The embedding width remains 768 after each merge. This keeps the TET
interface unchanged and avoids increasing the channel width as spatial
resolution decreases.

### 5.3 Stage dimensions

  Quantity                                    Input to merge 1   After merge 1   After merge 2
  ----------------------------------------- ------------------ --------------- ---------------
  Grid height × width                                    32×32           16×16             8×8
  Tokens per timestamp                                   1,024             256              64
  Features per token                                       768             768             768
  Concatenated features per merged region                  ---           3,072           3,072
  Projection                                               ---     3,072 → 768     3,072 → 768

The spatial token count is divided by four at each stage. Across both
stages, the count falls from 1,024 to 64, a 16× reduction.

## 6. Regional positional encoding

The ViT has already used spatial positional embeddings, so its token
features contain spatial context. HPE additionally adds a learned
positional embedding for the final 8×8 regional grid to make each output
region's location explicit to TET.

-   Type: learned 2D-grid positional embedding.
-   Shape: `(1, 64, 768)` after flattening the 8×8 grid in row-major
    order.
-   Operation: add the embedding to the 64 final regional tokens at each
    timestamp.
-   The same positional embedding table is used for every timestamp.
-   Do not add season, elapsed-time, or absolute calendar-year
    embeddings here; these belong to TET.

The final positional embedding identifies a region's grid location. It
does not by itself establish geographic alignment if input images have
inconsistent crops or grids. Preprocessing must maintain a consistent
spatial reference for the sequence, and any crop-based SSL view must use
one consistent crop geometry across its timestamps.

## 7. Hyperparameters

  ------------------------------------------------------------------------
  Parameter                          Initial setting Rationale / status
  --------------------- ---------------------------- ---------------------
  Input grid                                   32×32 Fixed by ViT
                                                     patchification

  Input token dimension                          768 Fixed by ViT output

  Number of merge                                  2 Produces the required
  stages                                             8×8 grid

  Merge window                                   2×2 Aggregates local
                                                     neighbors

  Merge stride                                     2 Non-overlapping
                                                     merging

  Tokens after stage 1                           256 16×16 grid

  Tokens after stage 2                            64 8×8 grid

  Concatenated                                 3,072 Four ×
  dimension                                          768-dimensional
                                                     tokens

  Normalization            LayerNorm(3072) per stage Applied before
                                                     projection

  Projection            Linear(3072 → 768) per stage Learned dimensional
                                                     compression

  Output dimension                               768 Matches TET

  Final positional           Learned, `(1, 64, 768)` Makes final regional
  embedding                                          locations explicit

  Merge-stage                                   None Baseline uses
  activation                                         normalization and
                                                     linear projection
                                                     only

  Merge-stage residual                          None Grid and token count
                                                     change; no residual
                                                     path is required in
                                                     the baseline

  Dropout in HPE                       0.0 initially Keep baseline simple;
                                                     test regularization
                                                     if needed

  Temporal mixing in                            None Timestamps are
  HPE                                                processed
                                                     independently
  ------------------------------------------------------------------------

These are initial design settings. They should be kept fixed for the
first implementation and changed only through controlled experiments.

## 8. Parameter count

Ignoring small implementation details, each merge projection has:

-   Linear weights: `3072 × 768 = 2,359,296`
-   Linear bias: `768`
-   LayerNorm scale and bias: `2 × 3072 = 6,144`

Each stage therefore has approximately 2.37 million trainable
parameters. Two stages have approximately 4.73 million parameters. The
final positional embedding adds `64 × 768 = 49,152` trainable
parameters.

The total HPE parameter count is therefore approximately **4.78
million**, assuming both merge stages have their own projection and
normalization parameters.

## 9. Information preservation and limitations

HPE reduces the number of tokens from 1,024 to 64. This is necessarily a
compression operation; it cannot guarantee that every input feature
survives.

The design aims to preserve useful information by: - merging only
spatial neighbors rather than unrelated or randomly selected tokens; -
using learned projections rather than fixed averaging; - preserving the
2D grid and a consistent regional index; - retaining the 768-dimensional
feature width; - adding an explicit positional embedding at the output
grid.

Potential risks include losing small-scale features, mixing different
surface types within one regional token, and imposing a fixed 8×8 grid
on glaciers with different shapes and extents. These are empirical
questions. Validate the module using training stability, representation
probes, and downstream or proxy tasks that are sensitive to glacier
change.

The HPE output should not be described as lossless. The correct goal is
task-relevant compression.

## 10. Forward path

For one glacier sequence:

``` text
Input: (T, 1024, 768)
  |
  v
Reshape: (T, 32, 32, 768)
  |
  v
Merge 1:
  2×2 neighborhood concatenation
  LayerNorm(3072)
  Linear(3072 → 768)
  |
  v
(T, 16, 16, 768)
  |
  v
Merge 2:
  2×2 neighborhood concatenation
  LayerNorm(3072)
  Linear(3072 → 768)
  |
  v
(T, 8, 8, 768)
  |
  v
Flatten spatial grid: (T, 64, 768)
  |
  v
Add learned regional position embedding
  |
  v
Output to TET: (T, 64, 768)
```

## 11. Implementation checks

Before connecting HPE to TET, verify:

1.  Input shape is `(T, 1024, 768)`.
2.  Input token ordering matches the ViT's row-major 32×32 grid.
3.  The reshape and each merge preserve spatial neighborhood order.
4.  Stage 1 outputs `(T, 256, 768)`.
5.  Stage 2 outputs `(T, 64, 768)`.
6.  No operation combines features from different timestamps.
7.  The final positional embedding has shape `(1, 64, 768)` and is
    broadcast over timestamps.
8.  Output contains no CLS token and no temporal pooling.
9.  Gradients reach both merging stages and the positional embedding
    during training.
10. Tests cover variable `T`, including a single timestamp, without
    changing the per-timestamp output contract.
11. The HPE output shape matches the TET input contract exactly.

## 12. Interface with the rest of the architecture

``` text
Multispectral ViT
  Input per timestamp: (6, 512, 512)
  Output per timestamp: (1024, 768)
        |
        v
HPE
  Input sequence: (T, 1024, 768)
  Output sequence: (T, 64, 768)
        |
        v
Temporal Evolution Transformer
  Input sequence: (T, 64, 768)
  Regional output: (T, 64, 768)
  Timestamp output: (T, 768)
  Sequence evolution embedding: (768,)
```

TET is responsible for adding season and elapsed-time encodings and
modeling temporal and spatial relationships. HPE supplies compact,
spatially ordered regional features.

**Status:** HPE interface and baseline architecture specified. The
two-stage merge design is a reasoned starting point and still requires
empirical validation; it is not yet an experimentally established
optimum.
