# Multispectral Vision Transformer (Spatial Encoder)

## 1. Purpose

Encode **one Sentinel-2 timestamp at a time** into spatially contextualized patch tokens. The Spatial ViT does not model time, classify glacier events, or compute the downstream self-supervised losses. Temporal reasoning belongs to the Temporal Evolution Transformer and its downstream SSL heads.

## 2. Input contract

- **Bands, in fixed order:** `B02, B03, B04, B08, B11, B12`
- **Image size:** `512 × 512` for every band
- **SCL:** quality information only; **not** a seventh model input channel. Use it in preprocessing/quality masking as appropriate.
- **One timestamp tensor:** `(B, 6, 512, 512)`, where `B` is the image batch size.
- **One glacier sequence:** `(T, 6, 512, 512)`, where `T` is variable. Examples: 16 timestamps for 4 years, 24 for 6 years, 36 for 9 years, assuming four seasonal observations per year.
- Keep timestamps in chronological order. The Spatial ViT processes each timestamp independently and does not receive temporal embeddings.

### Preprocessing

- Keep the six-band order identical for every image.
- Use consistent band-wise normalization, with statistics calculated from the **training split only** and then reused for validation, testing, and inference.
- Do not independently normalize each image by its own mean and standard deviation; this can remove useful between-date reflectance differences.
- Handle invalid/cloudy pixels using SCL-derived quality information in the data pipeline. Do not treat SCL class IDs as continuous reflectance values.
- Confirm all bands share the same grid, alignment, extent, and pixel dimensions before stacking.

## 3. Patch embedding

Use a single Conv2D projection over all six input channels.

```python
nn.Conv2d(
    in_channels=6,
    out_channels=768,
    kernel_size=16,
    stride=16,
)
```

- **Patch size:** `16 × 16`
- **Number of patches:** `(512 / 16) × (512 / 16) = 32 × 32 = 1,024`
- **Output feature dimension:** `768`
- The projection fuses the six spectral channels within each patch.
- Do not use separate per-band encoders, grouped spectral tokens, or an additional spectral positional embedding in this version.

The convolution produces `(B, 768, 32, 32)`, then flattening the spatial grid and transposing gives `(B, 1024, 768)`.

## 4. Spatial positional encoding

- **Type:** learnable, absolute, 2D spatial positional embeddings.
- Add one positional vector to each of the 1,024 patch tokens before the Transformer blocks.
- Positional embedding shape: `(1, 1024, 768)`.
- No temporal position is encoded here; temporal/date/season encoding is the responsibility of the temporal module.
- No separate spectral positional encoding is used because each token already fuses all six bands.

## 5. Transformer encoder

| Parameter | Selected value |
|---|---:|
| Encoder blocks | 12 |
| Token/embedding dimension | 768 |
| Attention heads per block | 12 |
| Dimension per attention head | 64 (`768 / 12`) |
| MLP expansion ratio | 4 |
| MLP hidden dimension | 3,072 |
| MLP activation | GELU |
| Normalization | Pre-LayerNorm |
| Residual connections | Yes |
| Standard dropout | 0.0 initially |
| Stochastic depth / DropPath | 0.1 initial target; validate during implementation |

Each block applies multi-head self-attention across the 1,024 spatial tokens, followed by an MLP applied independently to each token, with normalization and residual connections. Attention lets each patch incorporate context from other regions of the image.

## 6. Output contract

- **No CLS token**
- **No global average pooling**
- **No classification head or prediction head inside the Spatial ViT**
- Preserve all 1,024 tokens.

For one timestamp, output shape: `(B, 1024, 768)`.

For a glacier sequence of variable length `T`, the logically assembled output is `(T, 1024, 768)` per glacier, or `(B, T, 1024, 768)` for a batch with a common padded/masked temporal length. The batching and padding/masking policy for variable-length sequences must be handled by the sequence data loader and temporal pipeline; it does not change the Spatial ViT.

The output is passed to the next architectural stage without spatial-token pooling or merging in this module. Hierarchical patch merging is deliberately deferred to a separate design discussion.

## 7. Initialization and pretrained weights

- Prefer a Sentinel-2 / remote-sensing-pretrained ViT checkpoint when a compatible checkpoint is available.
- Remove the checkpoint's task-specific classification head.
- Reuse compatible Transformer-block weights.
- Check the checkpoint's patch size, embedding dimension, depth, input bands, band order, normalization, and positional-embedding format before loading.
- If its input channel count or patch projection differs, adapt or reinitialize the patch projection deliberately; do not assume weights are directly compatible.
- If embedding dimension or block architecture differs from this specification, the weights will not load directly into the corresponding layers. Select a compatible checkpoint or explicitly design and document an adaptation.
- Fine-tuning is intended. A lower learning rate for pretrained spatial weights than for newly initialized modules is a reasonable starting strategy, not a fixed value in this module specification.

## 8. Forward path

```text
One timestamp: (B, 6, 512, 512)
        |
        v
Conv2D patch projection: kernel=16, stride=16, out_channels=768
        |
        v
32 × 32 patch grid
        |
        v
Flatten + transpose: (B, 1024, 768)
        |
        v
Add learnable 2D absolute positional embeddings
        |
        v
12 Transformer encoder blocks
(12 heads, head dimension 64, MLP 768 -> 3072 -> 768)
        |
        v
Spatial tokens: (B, 1024, 768)
        |
        v
Next stage: Temporal Evolution Transformer pipeline
```

## 9. Why a ViT?

The encoder is intended to build globally contextualized multispectral patch representations. Self-attention allows distant glacier regions to interact, while the token sequence provides a direct interface to later temporal modelling. This is a design rationale, not a claim that ViT will necessarily outperform CNN alternatives; that claim would require an empirical comparison.

## 10. Implementation checks

Before connecting this module to the rest of the architecture, verify:

1. Input has exactly six channels in the agreed order.
2. All inputs are `512 × 512` and use the same normalization.
3. Patch embedding returns exactly 1,024 tokens of dimension 768.
4. The encoder output shape remains `(B, 1024, 768)`.
5. No CLS token, pooling, internal prediction head, or temporal operation is applied.
6. The same spatial weights and positional layout are used for every timestamp.
7. Pretrained checkpoint compatibility is explicitly checked rather than assumed.

**Status:** This specification records the agreed Spatial ViT design. Checkpoint selection and any checkpoint-specific weight adaptation remain implementation decisions to verify.
