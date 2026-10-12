# Phase 0 — Decision Ledger (fixed architecture vs experiment)

Every non-trivial choice is tagged **FIXED** (architecture contract — hardcode the
*structure*, keep numeric values in config with the file's default) or **EXPERIMENT**
(exposed as a config field with the file's starting value; never silently chosen and
buried in module logic). Config keys refer to `src/configs/default.yaml`.

## Backbone — ViT (`VIT_aarchi.md`)
| Item | Tag | Value / default | Config key |
|---|---|---|---|
| 6-band order B02,B03,B04,B08,B11,B12 | FIXED | — | `data.band_order` |
| Image size 512×512 | FIXED | 512 | `vit.image_size` |
| Conv2d patch proj, k=16, s=16, out=768 | FIXED | — | `vit.patch_size`,`vit.embed_dim` |
| 1024 tokens, no CLS, no pooling, no head | FIXED | — | — |
| Learnable 2D absolute positional emb | FIXED | — | — |
| 12 blocks, 12 heads, dim 768, MLP 3072, GELU, pre-LN | FIXED | — | `vit.depth`,`vit.num_heads`,`vit.mlp_ratio` |
| Standard dropout 0.0 | EXPERIMENT | 0.0 | `vit.dropout` |
| DropPath 0.1 | EXPERIMENT | 0.1 | `vit.drop_path` |
| Pretrained checkpoint selection/adaptation | EXPERIMENT | none | `vit.pretrained_ckpt` |

## Backbone — HPE (`hpe.md`)
| Item | Tag | Value | Config key |
|---|---|---|---|
| 2 merge stages 32→16→8, 2×2 window, stride 2 | FIXED | — | `hpe.num_merge_stages` |
| Fixed neighborhood order (TL,TR,BL,BR), concat 3072 | FIXED | — | — |
| LayerNorm(3072) + Linear(3072→768) per stage | FIXED | — | — |
| Width preserved at 768; no residual; no activation | FIXED | — | — |
| Learned regional pos emb (1,64,768) | FIXED | — | — |
| No temporal mixing in HPE | FIXED | — | — |
| Dropout in HPE 0.0 | EXPERIMENT | 0.0 | `hpe.dropout` |

## Backbone — TET (`Temptrans_aarchi.md`)
| Item | Tag | Value | Config key |
|---|---|---|---|
| Factorized temporal→spatial→MLP per block | FIXED | — | — |
| 8 blocks, 12 heads, dim 768, MLP 3072, GELU, pre-LN | FIXED | — | `tet.depth`,`tet.num_heads`,`tet.mlp_ratio` |
| Season embedding table (4×768), from month→season map | FIXED | — | — |
| Elapsed-time from dates (days/365.2425) | FIXED | — | — |
| `ElapsedTimeEncoding(scalar)->(768,)` swappable interface | EXPERIMENT | `sinusoidal` | `tet.time_encoding` (sinusoidal\|mlp\|linear) |
| Sequence-span encoding family = same as elapsed-time | FIXED (family) | — | `tet.span_encoding` |
| **Context-token communication path** | FIXED choice (was open) | `broadcast_mean` | `tet.context_mode` (broadcast_mean\|cross_attention) |
| Context token excluded from spatial attention | FIXED | — | — |
| `sequence_out` = final context token (not pooled) | FIXED | — | — |
| `timestamp_out` = masked mean over 64 regions | FIXED | — | — |
| MLP dropout 0.1, attn dropout 0.1 | EXPERIMENT | 0.1 | `tet.mlp_dropout`,`tet.attn_dropout` |
| Relative temporal attention bias | EXPERIMENT (off) | false | `tet.relative_temporal_bias` |

**Context-mode resolution.** `broadcast_mean` (the file's first suggestion, chosen as
default): the context token is broadcast as an extra time-step token into every region's
temporal sequence `(T+1,768)`, participates in temporal self-attention, and is updated by a
masked mean over the 64 per-region copies. It is excluded from spatial attention and gets
its own MLP. `cross_attention` (alternative, exposed): context token is a query attending
over all valid regional tokens. Both are unit-tested (Phase 7 / `tests/test_tet.py`).

## Head 1 — Temporal DINO (`head1_archi.md`)
| Item | Tag | Value | Config key |
|---|---|---|---|
| Views built on images before backbone | FIXED | — | — |
| Input = `timestamp_out` per view | FIXED | — | — |
| Projector 768→2048→2048→1024, GELU | FIXED (initial) | — | `head1.hidden_dim`,`head1.bottleneck_dim` |
| WeightNorm prototype 1024→K | FIXED | — | — |
| Prototype count K | EXPERIMENT | 8192 | `head1.num_prototypes` |
| Student τ_s | EXPERIMENT | 0.1 | `head1.student_temp` |
| Teacher τ_t | EXPERIMENT | 0.04 | `head1.teacher_temp` |
| EMA momentum cosine 0.996→0.9999 | EXPERIMENT | — | `head1.ema_momentum_*` |
| Multi-crop: 2 global teacher; 2 global + 2 local student | EXPERIMENT | — | `head1.num_global_*`,`head1.num_local_*` |
| Global/local crop scale ranges | EXPERIMENT | (0.4,1.0)/(0.1,0.4) | `sampling.global_scale`,`sampling.local_scale` |
| Center EMA, stop-grad teacher | FIXED | — | — |
| λ_DINO | EXPERIMENT | 1.0 | `loss.lambda_dino` |

## Head 2 — Temporal Distance (`head2_archi.md`)
| Item | Tag | Value | Config key |
|---|---|---|---|
| Input = `timestamp_out` pairs, same glacier | FIXED | — | — |
| Symmetric pair feature concat(abs(u_i-u_j), u_i*u_j)=1536 | FIXED | — | — |
| MLP 1536→512→128→1, GELU, dropout 0.1, Softplus | FIXED (baseline) | — | `head2.mlp_hidden`,`head2.dropout` |
| LayerNorm pre-pair; no L2 by default | FIXED (baseline) | — | `head2.l2_normalize` (exp) |
| Target = `log1p(delta_years)`, SmoothL1 | FIXED (baseline) | — | `head2.loss` (exp) |
| Pair sampling = gap-stratified (B) | EXPERIMENT (baseline B) | `gap_stratified` | `head2.pair_sampling` |
| Gap bins / stratification edges | EXPERIMENT | [0.5,1,2,4] yr | `head2.gap_bins_years` |
| Never feed delta/index to predictor | FIXED | — | — |
| λ_distance | EXPERIMENT | 0.5 | `loss.lambda_distance` |

## Head 3 — Temporal Change (`head3_archi.md`)
| Item | Tag | Value | Config key |
|---|---|---|---|
| Input = `regional_out`; verify 64-region correspondence | FIXED | — | — |
| Directed interval `i→j`, `d_i<d_j` | FIXED | — | — |
| 5-group per-region feature (64,3840) | FIXED (baseline) | — | `head3.pair_features` (exp) |
| Change Encoder 3840→1536→768, GELU, dropout, LN | FIXED (baseline) | — | `head3.encoder_hidden` |
| Validity-aware mean pooling | FIXED (baseline) | — | `head3.pooling` (exp) |
| Projection 768→512→256, L2-norm | FIXED (baseline) | — | `head3.proj_dim` |
| Objective = two-view instance InfoNCE first | FIXED (baseline) | — | `head3.objective` |
| InfoNCE temperature | EXPERIMENT | 0.1 | `head3.temperature` |
| Prototype/cluster objective (SwAV/DeepCluster) | EXPERIMENT (staged) | off | `head3.cluster.enabled`,`head3.cluster.*` |
| Number of prototypes K | EXPERIMENT | 64 | `head3.cluster.num_prototypes` |
| Never feed gap to encoder; preserve direction | FIXED | — | — |
| Temporal composition consistency | EXPERIMENT (off) | false | `head3.composition.enabled` |
| λ_change, λ_cluster | EXPERIMENT | 0.5 / 0.5 | `loss.lambda_change`,`head3.cluster.weight` |

## Data / sampling / training (build plan + head files)
| Item | Tag | Value | Config key |
|---|---|---|---|
| Split by glacier identity (region-based in DB) | FIXED | — | `data.split_source` |
| Normalization fit on train only, persisted | FIXED | — | `data.norm_stats_path` |
| Reflectance scale (uint16 DN / 10000) | FIXED | 10000 | `data.reflectance_scale` |
| Window sampler min/max length, short/med/long freq | EXPERIMENT | 2/36, uniform-ish | `sampling.window.*` |
| Combined loss coefficients (same in all 3 files) | EXPERIMENT | 1.0/0.5/0.5 | `loss.lambda_*` |
| Seed | FIXED (recorded) | 42 | `run.seed` |
