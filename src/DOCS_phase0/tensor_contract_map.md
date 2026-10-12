# Phase 0 — Tensor-Contract Map

Reference shapes at every boundary. `B` = glacier-sequence batch, `T` = variable
timestamps (padded to batch max with a `(B,T)` validity mask), `V` = views (Head 1),
`P` = pairs (Head 2 / Head 3). Embedding width `D = 768` throughout. Spatial `H=W=512`,
patch grid `32×32=1024`, regional grid `8×8=64`. All integration phases check against this.

## Per-timestamp input
```
one timestamp tensor     (B, 6, 512, 512)          6 bands, reflectance, normalized
validity mask (pixel)    (B, 512, 512)  bool        SCL-derived, True = valid
one glacier sequence     (T, 6, 512, 512)
batched sequences        (B, T, 6, 512, 512)
timestamp validity       (B, T)         bool        True = real observation, False = pad
dates                    (B, T)         float/days   acquisition dates (for temporal math)
```

## ViT (Phase 5)  —  per timestamp, shared weights, no temporal op
```
in   (N, 6, 512, 512)      N = B*T_valid folded into batch
out  (N, 1024, 768)        32×32 tokens, row-major, no CLS, no pooling
```

## HPE (Phase 6)  —  per timestamp, independent, no cross-timestamp mixing
```
in   (N, 1024, 768)  ->  reshape (N, 32, 32, 768)
     merge1 -> (N, 16, 16, 768)
     merge2 -> (N, 8, 8, 768)
     flatten + regional pos emb
out  (N, 64, 768)
```

## TET (Phase 7)  —  factorized temporal→spatial→MLP, 8 blocks
```
in            regional (B, T, 64, 768)
              + SeasonEmbedding(season[t])        (B, T, 1, 768) broadcast over regions
              + ElapsedTimeEncoding(elapsed_yr[t])(B, T, 1, 768) broadcast over regions
              context_0 = LearnableSeqToken + SpanEncoding(span_yr)   (B, 768)
              temporal mask (B, T) bool, region mask (B, T, 64) bool
outputs
  regional_out   (B, T, 64, 768)
  timestamp_out  (B, T, 768)        masked mean over valid regions per timestamp
  sequence_out   (B, 768)           final context-token hidden state (NOT pooled)
```

## Backbone integration (Phase 8)
```
(B,T,6,512,512) + masks + dates/seasons
   -> ViT (fold B,T_valid) -> (B,T,1024,768)
   -> HPE                  -> (B,T,64,768)
   -> TET                  -> regional_out / timestamp_out / sequence_out
padding provably inert at every stage.
```

## Head 1 — Temporal DINO (Phase 9)  —  input: timestamp_out per view
```
per view v: backbone(view_v) -> timestamp_out_v (B, T, 768)
projector:  768 -> 2048 -> 2048 -> 1024 (GELU)        bottleneck z (B,T,1024)
prototypes: WeightNorm Linear 1024 -> K=8192          logits (B,T,K)
loss: cross-view CE over valid (view-pair, timestamp); teacher EMA + center; stop-grad.
teacher views: 2 global ; student views: 2 global + 2 local.
```

## Head 2 — Temporal Distance (Phase 10)  —  input: timestamp_out pairs
```
z_i,z_j = timestamp_out[i], timestamp_out[j]   (P, 768) each
u = LayerNorm(z)
pair_features = concat(|u_i-u_j|, u_i*u_j)      (P, 1536)
MLP 1536->512->128->1, Softplus                 pred_years (P,)
target = log1p(delta_years from dates)
loss = SmoothL1(log1p(pred), log1p(target))
```

## Head 3 — Temporal Change (Phase 11)  —  input: regional_out pairs
```
R_i,R_j = regional_out[i], regional_out[j]      (P, 64, 768) each, d_i < d_j
per-region feat = [LN(r_i), LN(r_j), LN(r_j)-LN(r_i), |LN(r_j)-LN(r_i)|, LN(r_i)*LN(r_j)]
                                                 (P, 64, 3840)
Change Encoder (shared, region-wise) 3840->1536->768   (P, 64, 768)
validity-aware mean pool over 64 regions         change_embedding (P, 768)
projection 768->512->256 + L2norm                contrastive_embedding (P, 256)
two views a,b of same directed interval -> symmetric InfoNCE on contrastive_embedding.
```

## Combined loss (Phase 12)
```
L_total = λ_dino·L_DINO + λ_distance·L_distance + λ_change·L_change
defaults 1.0 / 0.5 / 0.5 ; each logged separately + grad norms to backbone.
teacher = EMA only; all student paths backprop into shared ViT+HPE+TET.
```
