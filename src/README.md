# Glacier SSL — implementation

Self-supervised pretraining system for glacier-evolution representations, built strictly
to the architecture docs in [`../DOCS/`](../DOCS) and the phased roadmap in
[`../DOCS/glacier_ssl_build_plan.md`](../DOCS/glacier_ssl_build_plan.md). The READMEs are the
spec; where the diagram disagrees, the README wins (see
[`DOCS_phase0/discrepancy_report.md`](DOCS_phase0/discrepancy_report.md)).

## What it is

A shared backbone — **Multispectral ViT → Hierarchical Patch Embedding → Temporal Evolution
Transformer** — pretrained by three SSL heads over the same backbone:

| Head | Objective | Backbone output it consumes |
|------|-----------|------------------------------|
| Head 1 — Temporal DINO | self-distillation across image views | `timestamp_out (T,768)` |
| Head 2 — Temporal Distance | elapsed-time regression on pairs | `timestamp_out (T,768)` |
| Head 3 — Temporal Change | directed change representation + staged clustering | `regional_out (T,64,768)` |

**Variable sequence length is a property of the whole pipeline.** `T` is never fixed; every
layer (dataset → collate → backbone → pooling → losses) preserves it, and all temporal
quantities come from actual acquisition **dates**, never array indices. Padding/validity
masks are inert everywhere.

## Phase → code map

| Phase | What | Where |
|------:|------|-------|
| 0 | Discrepancy report, decision ledger, tensor-contract map | [`DOCS_phase0/`](DOCS_phase0) |
| 1 | Config, logging, seeding, run metadata | `glacier_ssl/config.py`, `glacier_ssl/utils/` |
| 2 | Manifest + per-glacier dated sequences + identity split | `glacier_ssl/data/manifest.py` |
| 3 | Preprocessing, SCL validity masks, train-only normalization | `glacier_ssl/data/preprocessing.py`, `normalization.py` |
| 4 | Window sampler, global/local views, 3-way augmentation, metadata | `glacier_ssl/data/sampling.py`, `augmentation.py`, `dataset.py` |
| 5 | Multispectral Spatial ViT | `glacier_ssl/backbone/vit.py` |
| 6 | Hierarchical Patch Embedding (two-stage merge) | `glacier_ssl/backbone/hpe.py` |
| 7 | Temporal Evolution Transformer (+ context-token path) | `glacier_ssl/backbone/tet.py`, `temporal_encoding.py` |
| 8 | Backbone integration (variable-length, masked) | `glacier_ssl/backbone/backbone.py` |
| 9 | Head 1 — Temporal DINO | `glacier_ssl/heads/head1_dino.py` |
| 10 | Head 2 — Temporal Distance | `glacier_ssl/heads/head2_distance.py` |
| 11 | Head 3 — Temporal Change | `glacier_ssl/heads/head3_change.py` |
| 12 | Combined three-head training loop + checkpointing | `glacier_ssl/train/` |
| 13 | Test suite (acceptance gate) | `tests/` |
| 14 | Eval metrics + sanity run | `glacier_ssl/eval/metrics.py`, `scripts/sanity_run.py` |

## Tensor contract (also in `DOCS_phase0/tensor_contract_map.md`)

```
(B,T,6,512,512) --ViT--> (B,T,1024,768) --HPE--> (B,T,64,768) --TET--> {
    regional_out  (B,T,64,768),   timestamp_out (B,T,768),   sequence_out (B,768) }
Head1: timestamp_out/view -> 768->2048->2048->1024 -> WN prototypes K
Head2: concat(|u_i-u_j|, u_i*u_j) (1536) -> 512->128->1 Softplus
Head3: 5-group region feat (64,3840) -> 1536->768 -> valid-mean -> 768 -> proj 256
```

The context-token communication path (the one ambiguity flagged in `Temptrans_aarchi.md`)
is resolved as `broadcast_mean` by default and is config-swappable to `cross_attention`;
both are unit-tested (`tests/test_backbone.py`).

## Running (inside the project `venv`)

```bash
# 1. build the manifest from the SQLite catalog + on-disk GeoTIFFs (Phase 2)
python src/scripts/build_manifest.py

# 2. fit train-only band normalization (Phase 3)
python src/scripts/fit_normalization.py --max-observations 300

# 3. tests — the acceptance gate (Phase 13)
python -m pytest src/tests -q

# 4. real-data end-to-end sanity run, reduced dims for CPU (Phase 14)
python src/scripts/sanity_run.py --glaciers 8 --steps 4

# 5. full training (keep image_size/window small on CPU via a config override)
python src/scripts/train.py --epochs 1 --limit-glaciers 50
```

All configurable values (everything the docs tag *initial/baseline/experiment*) live in
[`configs/default.yaml`](configs/default.yaml); pass `--config your.yaml` to override. The
default config keeps the full spec dims (768-wide, 512×512, 12+8 blocks, 64 regions). Tests
and the sanity run use a smaller, internally-consistent config so shape/logic is exercised
without a 12-block/512px cost on CPU.

## Data layout consumed

`new_pipeline_dataset/{split}/{region}/{glims_id}/{year}_{season}/{B02,B03,B04,B08,B11,B12,SCL}.tif`
plus `glacier_dataset.sqlite3` (tables `glaciers`, `timestamps`, `bands`, `regions`). Splits
come from the DB's region assignment, enforced by **glacier identity** before any
pair/interval is generated (no leakage). 10m and resampled-20m bands co-register to within
~1 pixel; the alignment check tolerates that (`data.alignment_tolerance_m`) and requires
identical shape+CRS.

## Scope note

This builds and verifies the architecture, data pipeline, and training mechanics. The
sanity run proves the system trains end-to-end with finite losses and real backbone
gradients; it makes **no** scientific-validity claim. Held-out-glacier evaluation, shortcut
diagnostics (Head 2), cluster-occupancy checks (Head 3), and ablations are the next step
(`glacier_ssl/eval/metrics.py` provides the metrics).
