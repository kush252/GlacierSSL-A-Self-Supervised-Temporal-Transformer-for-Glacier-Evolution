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

# 1b. record each scene's Sentinel-2 processing baseline (needed to remove the +1000 DN
#     offset of baseline >= 04.00; see DOCS_phase0/architecture_review.md section 3B)
python src/scripts/fetch_processing_baselines.py

# 2. fit train-only band normalization (Phase 3) -- on harmonised reflectance
python src/scripts/fit_normalization.py --max-observations 300

# 3. tests — the acceptance gate (Phase 13)
python -m pytest src/tests -q

# 4. wiring audit — gradient/optimizer coverage, masks, contracts (see below)
python src/scripts/verify_system.py

# 5. real-data end-to-end sanity run, reduced dims for CPU (Phase 14)
#    NOTE: reduced dims crop a 128px corner out of the native 512px scenes, so
#    `dino 0.0 / intervals 0` steps are expected here — see the caveat in the script.
python src/scripts/sanity_run.py --glaciers 8 --steps 4

# 5. full training (keep image_size/window small on CPU via a config override)
python src/scripts/train.py --epochs 1 --limit-glaciers 50
```

All configurable values (everything the docs tag *initial/baseline/experiment*) live in
[`configs/default.yaml`](configs/default.yaml); pass `--config your.yaml` to override. The
default config keeps the full spec dims (768-wide, 512×512, 12+8 blocks, 64 regions). Tests
and the sanity run use a smaller, internally-consistent config so shape/logic is exercised
without a 12-block/512px cost on CPU.

## GPU acceleration (incl. Kaggle dual-GPU)

Acceleration is config-driven (`train:` in `default.yaml`) and implemented in
[`glacier_ssl/train/accel.py`](glacier_ssl/train/accel.py). All of it is a no-op on CPU, so
the CPU path and the tests are unchanged.

| Config | Effect |
|--------|--------|
| `amp: true` + `amp_dtype: float16` | Mixed precision (autocast + GradScaler). ~2-3× faster + less memory on GPU. Kaggle T4 → use `float16`; A100 → `bfloat16`. Auto-off on CPU. |
| `data_parallel: auto` | `nn.DataParallel` across **all visible GPUs** when >1 is present (e.g. Kaggle's 2× T4). `auto`\|`true`\|`false`. |
| `tf32: true` | Allows TF32 matmul/conv on Ampere+ (harmless elsewhere). |
| `pin_memory: auto` | Pinned host memory on CUDA for faster H2D copies. |

`--device auto` picks CUDA when available. The base backbone modules stay unwrapped (so EMA,
optimizer, and checkpointing are correct); forward passes go through autocast + DataParallel
runners that share the same parameters. Season ids are tensorized before the backbone so
DataParallel scatters them in lock-step with the images.

**Note on efficiency:** `DataParallel` is the drop-in choice for a single-process notebook.
For maximum throughput, `DistributedDataParallel` (one process per GPU, launched with
`torchrun`/`mp.spawn`) is faster but needs a multi-process entrypoint; not wired here because
Kaggle notebooks run single-process and this trainer calls the backbone many times per step.

## Pre-training audit (run this before every launch)

```bash
python src/scripts/verify_system.py                 # 55 checks, ~40s on CPU
python src/scripts/verify_system.py --device cuda    # also exercises the AMP path
python src/scripts/verify_system.py --full-config    # + build the real 167M-param backbone
```

The pytest suite checks shapes, masks and directionality. This audit answers the questions
that otherwise waste a whole run, and exits non-zero if any fail:

| Group | What it proves |
|-------|----------------|
| A | derived dims agree across ViT/HPE/TET/heads (grid → regions → head widths) |
| B | real forward shapes match `DOCS_phase0/tensor_contract_map.md` |
| **C** | **every trainable parameter receives a finite, non-zero gradient** from the 3-head loss |
| **D** | **every trainable parameter is actually inside the optimizer** (grads alone don't train) |
| E | every gradient-receiving parameter moves after `optimizer.step()` |
| F | teacher takes no gradient and moves only by EMA |
| G | perturbing padded timesteps cannot change valid outputs |
| H | T = 1,2,3,5,7 all run; a T=3 sequence is bit-identical padded to T=6 |
| I | Head 2 is swap-symmetric; Head 3 preserves direction; neither receives what it predicts |
| J | multi-step losses stay finite; Heads 2 and 3 descend on a fixed batch |

Two parameters (`tet.context_norm.weight/bias`) are *expected* to receive no gradient and
the audit asserts they are the only ones. They are the terminal LayerNorm producing
`sequence_out`, which no baseline head consumes (Heads 1/2 take `timestamp_out`, Head 3
takes `regional_out` — `head3_archi.md`). The context **token** is trained; only this final
norm is outside the loss path, and at init it is exactly standard normalization. Attaching a
loss there would invent architecture the plan does not specify.

## Architecture review (read before the next training run)

[`DOCS_phase0/architecture_review.md`](DOCS_phase0/architecture_review.md) records why the
first full run learned little and what changed: a **temporal firewall** (Heads 2/3 read a
time-blind TET pass, so their temporal targets are not handed to them by the date
encodings), **Sentinel-2 baseline harmonisation** (33% of scenes carried a +1000 DN
offset), **sparse multi-year windows**, and a **pretrained ViT** (Prithvi-EO-1.0-100M; the
loader existed but was never called). Every change is a config flag, for ablation.

The evidence comes from the first evaluation run, committed at
[`artifacts/eval/output__step6720__20261109_103531__full/`](artifacts/eval/output__step6720__20261109_103531__full)
(start with its `REPORT.md`; 40 held-out glaciers, original checkpoint at step 6720).

**Consequences for the next run:**

- **Train from scratch, never `--resume` an older checkpoint.** The input normalisation,
  the ViT's starting weights and the optimizer's parameter groups have all changed.
- **Download Prithvi first.** `configs/kaggle.yaml` expects it at
  `/kaggle/working/Prithvi_EO_V1_100M.pt` (step 3 of the Kaggle flow below).
- **Head 2's loss will stay far above the old ~0.015.** That is intended: the answer is no
  longer in its input.
- **Old checkpoints still evaluate exactly as they were trained.** Their stored config
  predates these keys, so they fall back to no harmonisation, the old normalisation
  statistics, and a time-aware Head 2.

## Evaluation after pretraining

`src/scripts/evaluate.py` runs the label-free evaluation suite on **held-out glaciers**
(library code in `glacier_ssl/eval/`, tests in `tests/test_eval.py`):

```bash
python src/scripts/evaluate.py --ckpt <runs/<run>/ckpt_latest.pt> --device cuda --zip
```

| Test | Question | Module |
|------|----------|--------|
| `head2` | Head 2 error in years, calibration, per-gap MAE, pair-order invariance, metadata-only and visual-only controls (head2_archi.md §8) | `eval/head2_eval.py` |
| `embedding` | do features organise by season / region / size / latitude / year? health, probes, t-SNE | `eval/embedding_eval.py` |
| `trajectory` | does a glacier's embedding move smoothly and track scene change? | `eval/trajectory_eval.py` |
| `retrieval` | nearest-neighbour identity and same-state retrieval vs untrained + raw-pixel baselines | `eval/retrieval_eval.py` |
| `robustness` | missing timestamps, short sequences, observed + simulated cloud cover | `eval/robustness_eval.py` |

Every test also scores same-architecture **baselines** (`--baselines auto`, the default):

- **`random`**: everything untrained. Asks whether the model beats random weights.
- **`pretrained`**: the pretrained ViT (Prithvi) with an untrained HPE/TET. Asks whether
  glacier SSL adds anything beyond the off-the-shelf encoder it started from. Added
  automatically when the config sets `vit.pretrained_ckpt`; `--pretrained-ckpt <path>`
  overrides the file.

`--baselines random` picks a subset; `--no-baseline` skips them all. When the checkpoint was
trained with `head2.time_blind`, the Head 2 tests read the time-blind features
(`timestamp_out_blind`), the same path the head was trained on.

Useful flags: `--tests head2,retrieval`, `--backbone teacher`, `--limit-glaciers 20` (quick
look), `--occlusion-glaciers 0` (skip the only image-rereading robustness part),
`--reuse-cache <earlier run>/cache` (skip re-extraction).

**Where results go.** One folder per run under `src/artifacts/eval/` (on Kaggle:
`/kaggle/working/eval_results/`): `REPORT.md`, `summary.json`, `run_manifest.json`
(checkpoint, config, seed, git, args), `eval.log`, one sub-folder per test with tables
(`.csv` + `.md`) and figures (`.png` + `.pdf`), and a `cache/` of extracted features
(large; not committed). `results_index.jsonl` in the same root accumulates one headline
line per (run, test) so checkpoints and ablations can be compared later.

The feature extraction runs the backbone once per glacier; the robustness test then
re-runs only the temporal transformer on dropped/truncated timestamps (exact, because
ViT+HPE act per timestamp). The manifest has no elevation, so size and latitude are the
physical covariates used.

## Kaggle flow (2× T4)

Notebook settings: **GPU T4 ×2**, **Internet ON** (needed for `git` and the Prithvi
download). Attach the imagery dataset as an input. The code comes from GitHub; the
repository root is this `implementation/` folder.

**1. Get the code.**

```python
# first session
!git clone https://github.com/kush252/GlacierSSL-A-Self-Supervised-Temporal-Transformer-for-Glacier-Evolution.git /kaggle/working/glacier
# later sessions
!cd /kaggle/working/glacier && git pull
```

If `git pull` refuses after history on GitHub was rewritten, use
`git fetch && git reset --hard origin/main` instead. The clone holds no local work.

**2. Point the config at the data (Cell 2).** `src/configs/kaggle.yaml` holds the GPU
settings; only `data.data_root` needs the real path. The manifest stores *relative* paths
beginning `new_pipeline_dataset/`, so `data_root` is the folder that **contains**
`new_pipeline_dataset/`, not the inner folder. The patched copy is written *outside* the
repo so `git pull` never conflicts with it. Re-run this cell after every `git pull`:

```python
import glob, yaml, pathlib

hits = []
for d in range(6):
    hits = glob.glob('/kaggle/input/' + '*/' * d + 'new_pipeline_dataset')
    if hits:
        break
assert hits, f"dataset not found: {glob.glob('/kaggle/input/*')}"
data_root = str(pathlib.Path(hits[0]).parent)

base = yaml.safe_load(pathlib.Path('/kaggle/working/glacier/src/configs/kaggle.yaml').read_text())
base['data']['data_root'] = data_root
local = pathlib.Path('/kaggle/working/kaggle_local.yaml')
local.write_text(yaml.safe_dump(base))
print('data_root =', data_root)
```

**3. Download the pretrained ViT** (Prithvi-EO-1.0-100M, about 450 MB, once per session):

```python
!wget -q -O /kaggle/working/Prithvi_EO_V1_100M.pt https://huggingface.co/ibm-nasa-geospatial/Prithvi-EO-1.0-100M/resolve/main/Prithvi_EO_V1_100M.pt
!ls -la /kaggle/working/Prithvi_EO_V1_100M.pt
```

If the file is missing, training stops with an error rather than silently starting the
ViT from scratch. To train from scratch on purpose (an ablation row), set
`vit.pretrained_ckpt: null` in Cell 2.

**4. Audit, then train.** Always run the audit first. It takes under a minute and catches a
broken mount or a bad config before you spend GPU quota:

```python
import os; os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
!cd /kaggle/working/glacier && python src/scripts/verify_system.py --device cuda
# smoke test
!cd /kaggle/working/glacier && python src/scripts/train.py --config /kaggle/working/kaggle_local.yaml \
    --device cuda --limit-glaciers 20 --epochs 1
# full run (start from an empty output folder: rm -rf /kaggle/working/runs)
!cd /kaggle/working/glacier && python src/scripts/train.py \
    --config /kaggle/working/kaggle_local.yaml --device cuda --epochs 20
# continue a run of the SAME configuration in a later session
!cd /kaggle/working/glacier && python src/scripts/train.py \
    --config /kaggle/working/kaggle_local.yaml --device cuda --epochs 20 \
    --resume /kaggle/input/glacier-ssl-checkpoints/ckpt_latest.pt
```

The log should print:
- `CUDA available: 2 GPU(s)`
- `DataParallel enabled across 2 GPUs`
- `Pretrained ViT from /kaggle/working/Prithvi_EO_V1_100M.pt: N/N tensors loaded, 0 still fresh.`

**5. Memory tuning — the part that actually bites.** One step runs the student backbone
**seven times with gradients retained** (1 base + 4 DINO student views + 2 Head-3 interval
views), and every call's activations stay alive until backward, so peak memory is their
**sum**, not the largest one. Cost scales as `7 × batch_size × window_length`
timestep-encodings. GPU 0 additionally carries the AdamW state (~2.7 GB for 167M params)
because DataParallel keeps the master copy there, so GPU 0 is the binding constraint.

Two optimizations make this fit in 16 GB, both verified to change memory and nothing else:

| Optimization | Effect |
|---|---|
| Fused SDPA attention (always on) | never materializes the (heads, 1024, 1024) score matrix; identical to ~1e-15 on values and gradients |
| `vit.grad_checkpoint` / `tet.grad_checkpoint` | recompute block activations in backward: ~300 MB → ~20 MB per timestep-encoding, ~30% slower, **bit-identical** outputs and gradients |

Without checkpointing a T4 OOMs even at `max_length: 4`. With it enabled (as `kaggle.yaml`
does), `max_length: 8` fits comfortably — raise toward 12–16 until you OOM, then step back
one notch. Keep `batch_size` even so both T4s are fed. Audit section M re-proves the
bit-exactness on every run.

**6. Persist checkpoints — `/kaggle/working` is wiped when the session ends.** `output_root`
is `/kaggle/working/runs`, which is ephemeral. Before the session dies, push the run to a
permanent Kaggle Dataset:

```python
import json, pathlib
out = pathlib.Path('/kaggle/working/runs')
(out/'dataset-metadata.json').write_text(json.dumps(
    {"title": "glacier-ssl-checkpoints", "id": "<username>/glacier-ssl-checkpoints",
     "licenses": [{"name": "CC0-1.0"}]}))
!kaggle datasets create -p /kaggle/working/runs --dir-mode zip      # first time
# !kaggle datasets version -p /kaggle/working/runs -m "epoch 20" --dir-mode zip   # later
```

Next session, attach that dataset as an input and resume from
`/kaggle/input/glacier-ssl-checkpoints/ckpt_latest.pt` (step 4). Kaggle pins the dataset
**version** a notebook was attached with. After pushing a new version, update the input in
the notebook's side panel, then check the step printed by the resume log.

**7. Evaluate and save the results.**

```python
!python /kaggle/working/glacier/src/scripts/evaluate.py \
    --ckpt /kaggle/working/runs/<run>/ckpt_latest.pt \
    --config /kaggle/working/kaggle_local.yaml \
    --device cuda --workers 4 --name full36 --zip
```

Results land in `/kaggle/working/eval_results/`: one `<run>/` folder per run plus, with
`--zip`, a `<run>.zip` next to it (without the feature cache). Publish **only the zips and
the index** from a separate folder. Uploading `eval_results/` itself with `--dir-mode zip`
makes Kaggle zip each `<run>/` into a second `<run>.zip`, and the upload fails with
*"Path must be unique"*.

```python
import json, pathlib, shutil
src = pathlib.Path('/kaggle/working/eval_results')
up = pathlib.Path('/kaggle/working/eval_upload')
shutil.rmtree(up, ignore_errors=True); up.mkdir()
for z in src.glob('*.zip'):
    shutil.copy2(z, up)
shutil.copy2(src / 'results_index.jsonl', up)
(up / 'dataset-metadata.json').write_text(json.dumps({
    "title": "glacier-ssl-eval", "id": "<username>/glacier-ssl-eval",
    "licenses": [{"name": "CC0-1.0"}]}, indent=2))
!kaggle datasets status <username>/glacier-ssl-eval        # "not found" -> create, else version
!kaggle datasets create -p /kaggle/working/eval_upload      # first time
# !kaggle datasets version -p /kaggle/working/eval_upload -m "<what changed>"   # later
```

A new version **replaces** the dataset's files. To keep earlier runs, attach the dataset
as an input and copy its files into `eval_upload/` before versioning.

To keep results in git, unzip each run into `src/artifacts/eval/` locally and commit.

## Uploading the dataset to Kaggle (name sanitization)

Kaggle Datasets reject file/folder names with characters outside `[A-Za-z0-9._-]`. **128
region folders** here are named after GLIMS glaciers and contain apostrophes, accents, or
parentheses (e.g. `DES_BRAMES_DE_L'OURS_1`, `BREITHORNGLETSCHER_(ZERMATT)`), so the upload
fails with *"contains a forbidden character in name"*. Everything below region level is
already safe. Fix with:

```bash
python src/scripts/sanitize_for_kaggle.py                 # dry run: shows what changes
python src/scripts/sanitize_for_kaggle.py --apply --manifest   # rename on disk + fix manifest
```

It renames offending path components to safe equivalents and rewrites the manifest's stored
paths with the same rule (keeping a `.bak`). The local SQLite DB is left untouched (not used
on Kaggle). Re-zip `new_pipeline_dataset/` afterwards and upload.

Upload the whole tree as **one zip** rather than 262k loose files — Kaggle's uploader makes
roughly one request per file. The zip must keep `new_pipeline_dataset/` as its single
top-level folder, because that is the prefix the manifest's relative paths carry.

## Data layout consumed

`new_pipeline_dataset/{split}/{region}/{glims_id}/{year}_{season}/{B02,B03,B04,B08,B11,B12,SCL}.tif`
plus `glacier_dataset.sqlite3` (tables `glaciers`, `timestamps`, `bands`, `regions`). Splits
come from the DB's region assignment, enforced by **glacier identity** before any
pair/interval is generated (no leakage). 10m and resampled-20m bands co-register to within
~1 pixel; the alignment check tolerates that (`data.alignment_tolerance_m`) and requires
identical shape+CRS.

## What is and is not in git

Kept in git: code, configs, the manifest, normalisation statistics, evaluation results
(`artifacts/eval/<run>/`), and the event-labelling pipeline with its labels and figures
(`../pipelines/event_labeling/`, which has its own README).

Ignored because they are large and can be regenerated (`.gitignore`):

| Path | What | Regenerate with |
|---|---|---|
| `src/output/` | training checkpoints, about 3 GB each | training; keep them in a Kaggle dataset |
| `src/artifacts/eval/**/cache/` | extracted features per eval run | `evaluate.py` (or `--reuse-cache`) |
| `pipelines/event_labeling/cache/` | downloaded scenes and DEMs for labelling | `run_labeling.py` |
| `src/artifacts/manifests/*.bak` | manifest backups written by the baseline / sanitise scripts | the script that wrote them |

## Scope note

This builds and verifies the architecture, data pipeline, training mechanics and a
label-free evaluation suite. The first full run's evaluation
(`artifacts/eval/output__step6720__20261109_103531__full/`) found the problems fixed in the
architecture review. Next:
1. Retrain with the review's changes.
2. Evaluate against the random and pretrained baselines.
3. Run the ablation grid in `DOCS_phase0/architecture_review.md` §7.
4. Run the downstream event-classification task on the labels from
   `pipelines/event_labeling`.
