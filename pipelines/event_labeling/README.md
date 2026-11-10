# Glacier event ground truth (`event_labeling`)

Builds **evidence-backed labels** (Stable / Retreat / Advance / Fragmentation / Disappearance)
for glacier image sequences **of any length**, together with the measurements and figures
each label was derived from, so they can be shown in a paper and reused for evaluation.

It replaces `pipelines/evolution` as the source of ground truth. That pipeline is not usable as
ground truth because it:
- mixes winter, spring and autumn scenes (seasonal snow dominates "area");
- counts SCL snow over the whole tile, not the glacier;
- assumes 10 m pixels on tiles resampled to 512×512 (actual pixels are about 5.8 m × 3.0 m);
- takes the southernmost pixel as the "terminus";
- labels nearly every glacier "Fragmentation" because of how it measures bridge width.

## Design: separate *measurement* from *labelling*

```
Sentinel-2 L2A (native 10 m, outline + 400 m, pixel-aligned UTM grid) + Copernicus GLO-30 DEM
        │  one end-of-ablation epoch per year (least seasonal snow, QC-passed)
        ▼
MEASUREMENT LAYER  (epochs.csv, ground_truth.json, masks.npz)  ← the ground truth
  area ± σ (two spectral masks) · components ≥ MMU · terminus elevation · QC status
        │  any window [y0, y1] of ≥ 2 valid epochs
        ▼
LABEL  = rules over independent evidence families  → primary label + attributes + tier
        │
        ▼
HUMAN VERIFICATION (blind, TimeSync-style)  → Gold / Silver / Bronze / Reject
```

Labels are a pure function of the measurement table. Any window length, including an SSL
window given by dates (`GlacierGT.label_dates`), is labelled from the same measurements.
Every number in a label can be traced back to pixels.

## 1 · Scene selection (per glacier and year)
- Candidates: the 8 lowest-cloud scenes found between 20 Jul and 30 Sep, plus any of the SSL
  dataset's own scenes that fall in that window. October is excluded because low sun puts
  cirques in shadow and early snow arrives.
- Each scene is read from Planetary Computer onto one fixed 10 m grid per glacier. The SSL
  tiles are not reused: they are masked to the static outline, so advance and the snow
  context are invisible.
- QC rejects a scene if any of these hold:
  - nodata covers more than 1 % of the envelope;
  - more than 10 % of the glacier is cloud or cloud shadow that is not spectrally ice (this
    unknown area also enters σ);
  - more than 25 % of the glacier is in topographic shadow (NIR and red both < 0.06).
- Selection: least off-glacier snow in a 50–400 m ring. This follows standard inventory
  practice: the least-snow, cloud-free scene at the end of the ablation season.
- An epoch is marked `seasonal_snow` and excluded if its ring snow exceeds the glacier's
  typical end-of-season level by more than 0.15. "Typical" is the 25th percentile of the
  per-year best scenes. A quantile is used rather than the minimum so that one extreme melt
  year (for example 2022 in the Alps) does not flag every other year. Neighbouring permanent
  ice in the ring is part of this baseline.

## 2 · Measurements (per epoch)
| quantity | method |
|---|---|
| ice mask (primary) | red/SWIR ratio B4/B11 > 2.0 and blue > 0.05 (Paul et al. 2016); water removed where NDWI > 0.45 and NIR < 0.08, which keeps shadowed ice |
| ice mask (secondary) | NDSI > 0.4 and NIR > 0.11 (Hall et al. 1995) |
| ice body | ice inside the outline (+20 m), plus connected ice up to 300 m beyond it, counted only in the lowest 30 % of the outline's elevation range (where an advancing tongue would be). Upslope snowfields are excluded. |
| area σ | perimeter × 0.5 px (buffer method, Paul et al. 2013) combined with ½ × the cloud-unknown area; floor 2 % |
| components | 8-connected after a 1-px closing (gaps under about 20 m do not split); counted if ≥ 0.01 km² (RGI 7 minimum) |
| terminus | 2nd-percentile DEM elevation of the largest component |
| validation | IoU and area against independent GLIMS outlines from the same year (`glims_validation.csv`) |

## 3 · Window labels
For a window [y0, y1] with n ≥ 2 valid epochs:
- **Change test**:
  - n ≥ 4: Theil–Sen change, Mann–Kendall p < 0.05, and |Δ| > 2σ (σ combined from the two endpoints).
  - n = 3: monotonic and |Δ| > 2σ.
  - n = 2: |Δ| > 2σ only (low power).
- **Evidence families**. Each votes −, 0 or +:
  - *area*: the red/SWIR mask votes; the NDSI mask must be consistent with it. Both masks share
    snow errors, so they count as one family.
  - *terminus*: a rise of more than 15 m means retreat.
  - *glims*: independent GLIMS outlines dated inside the window, ±5 % each.
- **Retreat / Advance**: at least one family is significant in that direction and none in the
  opposite direction. Opposite significant votes give `Ambiguous`.
- **Fragmentation**: a component splits into at least two parts, each ≥ max(0.05 km², 2 % of
  the parent). Additional requirements:
  - every part is tracked forward: no re-merge, still separate for 2 consecutive valid epochs
    (confirmation may look ahead past the window end);
  - *pre-verified* if the parent was one body in the epoch before the split, which rules out a
    snow bridge;
  - supporting evidence: the same split in the NDSI mask, and GLIMS child outlines that first
    appear inside the window.
- **Disappearance**: area ≥ MMU at the window start, then below MMU in 2 consecutive valid epochs.
- **Stable**: at least 3 valid epochs, no significant family, area SNR ≤ 2.
- **Precedence**: Disappearance > Fragmentation > Retreat > Advance > Stable. The other true
  classes are kept in `attributes` (for example, Fragmentation with `also_retreat`).

**Automatic tier**

| tier | requirements |
|---|---|
| **A** | All of: ≥ 3 valid epochs; ≥ 80 % of the window valid; ≥ 2 independent families agree; none against; a strong signal (SNR ≥ 3, or a fragmentation that is both confirmed and pre-verified, or a confirmed disappearance). |
| **B** | Nothing against, and either 2 families agree or 1 strong family. |
| **C** | Anything else. |

Caps:
- An **Advance** supported by area alone is capped at **C**: residual snow mimics advance, so it
  needs the terminus or GLIMS family to agree.
- **Tier A** drops to **B** when the two masks disagree (IoU < 0.8) or the outline coverage is
  low (debris, or retreat before 2016).

## 4 · Human verification → final ground truth
`review_blind.html` hides the automatic label, so reviewers judge from the evidence alone and
are not anchored to it (as in the TimeSync and LCMAP interpretation protocols). Each reviewer
exports a CSV. Merging them computes Cohen's κ between reviewers and between auto and
consensus, plus a confusion matrix, and assigns the final tier:

| final tier | rule |
|---|---|
| **Gold** | ≥ 2 reviewers agree with each other and with the automatic label (auto tier A/B); no low-confidence vote |
| **Silver** | 1 reviewer agrees with auto; or auto tier A, unreviewed; or ≥ 2 reviewers overrule auto |
| **Bronze** | unreviewed tier B/C, or reviewers disagree |
| **Reject** | consensus "Insufficient" or "Ambiguous" |

Recommended use: report test metrics on **Gold**, use **Gold + Silver** for probes and training,
and keep Bronze for analysing label noise.

## Run
```bash
cd pipelines/event_labeling
../../venv/Scripts/python.exe run_labeling.py --n-glaciers 6 --windows 3 --seed 7 --require-glims
../../venv/Scripts/python.exe run_labeling.py --glims-id G041523E43281N --windows 4
# after reviewers export CSVs from review_blind.html:
../../venv/Scripts/python.exe run_labeling.py --run output/run_seed7 --merge-reviews output/run_seed7/review_A.csv output/run_seed7/review_B.csv
../../venv/Scripts/python.exe -m pytest tests -q        # rule tests (synthetic, offline)
```
Label an SSL window by its dates (for example, inside an evaluation script):
```python
from glacier_gt.config import load_config
from glacier_gt.pipeline import build_glacier
gt = build_glacier("G041523E43281N", load_config())     # uses the cache after the first run
gt.label_dates("2017-06-01", "2023-11-30")               # -> label record
```
Windows shorter than one year return `Insufficient`: change cannot be resolved below the
annual epoch spacing.

All thresholds are in `config.yaml` and should be reported in a methods table.

## Known limits (state them in the paper)
- **Debris-covered ice** is not mapped spectrally. Glaciers with low outline coverage are
  flagged; their area-based votes are weak, and the terminus and human review carry more
  weight.
- **Advance** beyond 300 m from the outline is truncated.
- **Surge** is only a flag (an abrupt significant gain of more than 10 %). Confirm it against
  the surge inventories (Sevestre & Benn 2015; Guillet et al. 2022).
- **WGMS front-variation** records (every glacier here has a `wgms_id` in GLIMS) can be added
  as a fourth evidence family once the Fluctuations of Glaciers (FoG) CSV is downloaded.
