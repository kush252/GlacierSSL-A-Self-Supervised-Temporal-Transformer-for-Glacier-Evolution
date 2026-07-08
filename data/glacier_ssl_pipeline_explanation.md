# Glacier SSL Pipeline - In-Depth Explanation

## Overview
The `glacier_ssl_pipeline.py` script is a robust, end-to-end data engineering pipeline designed to generate a large-scale, temporal, multi-spectral dataset of glaciers for Self-Supervised Learning (SSL). 

It intelligently combines glacier polygons and metadata from the **GLIMS Glacier Inventory** with satellite imagery from **Sentinel-2 L2A**, queried via the Microsoft Planetary Computer (STAC) API.

### Core Design Principles
1. **Fully Resumable:** The pipeline relies on a persistent SQLite state database (`dataset_state.db`). If the script crashes or is interrupted, it can be rerun, and it will resume exactly where it left off, avoiding redundant downloads.
2. **Atomic Writes:** Data processing outputs are initially written to a temporary directory. Only when all 6 spectral bands are successfully downloaded, verified, and processed are they moved to their final directory. This prevents corrupted or partial datasets.
3. **Defensive Networking:** All API calls and remote downloads are wrapped in an exponential backoff retry mechanism to tolerate network instability.
4. **Data Integrity:** Stringent checks are performed, including ensuring minimum glacier area, 9 continuous years of low-cloud imagery, and proper TIFF data structure verification.

---

## Detailed Overall Flow & Phases

The execution flow of the pipeline is divided into 8 distinct phases, orchestrated by the `run_pipeline` function. Users can select which phases to execute via command-line arguments.

### Phase 1: Glacier Filtering (`phase1_filter_glaciers`)
**Goal:** Identify a high-quality subset of glaciers that meet stringent size and observational criteria, whilst balancing them geographically.
- **Loading GLIMS:** `load_glims_inventory` parses the local GLIMS dataset using `geopandas`, normalizes columns, removes duplicates, and extracts centroid and geometry data.
- **STAC Queries:** `query_sentinel2_years` checks the Planetary Computer for Sentinel-2 imagery intersecting the glacier bounding box across a specified year range (e.g., 2015-2024).
- **Filtering Logic:** 
  - Filters for glaciers within an area band (default 5 to 1000 km²).
  - Determines "valid years" where the glacier has imagery with cloud cover below a specified threshold (`CLOUD_COVER_THRESHOLD`).
  - Ensures the glacier has a continuous block of at least 9 valid years (`find_continuous_block`).
- **Expansion/Trimming:** If the initial filter produces too few glaciers compared to `TARGET_GLACIER_COUNT`, it relaxes the area band to include smaller/larger glaciers. If it produces too many, it proportionally downsamples them while preserving geographic representation ("region balance").
- **Caching:** STAC queries are cached in `stac_valid_year_cache.json` to speed up subsequent runs.

### Phase 2: Region-level Splits (`phase2_region_split`)
**Goal:** Divide the selected glaciers into `train`, `val` (validation), and `test` splits.
- **Region Grouping:** Ensures that all glaciers from a specific region are assigned to the *same* split. This prevents data leakage (where spatially adjacent glaciers might end up in both training and testing sets).
- **Distribution:** 
  - Iterates through regions ordered by glacier count.
  - Assigns regions to the `train` split until `TRAIN_REGION_FRACTION` (70%) of total glaciers are accounted for.
  - The remainder is divided between `val` (20%) and `test` (80% of the remainder).
- **Output:** Generates a `split_manifest.csv` containing the split assignment for each glacier.

### Phase 3: Temporal Sampling (`phase3_select_timestamps`)
**Goal:** For each selected glacier, pick the absolute best satellite acquisition for every season across 9 years (36 timestamps total).
- Defines four seasons (`spring`, `summer`, `autumn`, `winter`), mapping them to specific months and "midpoint" days of the year. Note that winter wraps around the calendar year (Dec-Feb).
- `query_seasonal_candidates` searches the STAC catalog for imagery within each seasonal window.
- `select_best_seasonal_acquisition` ranks these images based on:
  1. Lowest cloud cover.
  2. Highest percentage of valid pixels.
  3. Lowest snow/cloud artifact score.
  4. Temporal proximity to the middle of the season.
- Retains a ranked list of backups in case the "best" image fails to download in Phase 4.

### Phase 4: Image Processing (`download_and_process_band`)
**Goal:** Download, crop, reproject, and resize the raw Sentinel-2 imagery into standardized GeoTIFFs.
- Iterates over required bands: B02, B03, B04 (RGB), B08 (NIR), B11, B12 (SWIR), and SCL (Scene Classification Layer).
- `rasterio` is used to stream and process the image data.
- **Cropping & Reprojecting:** Converts the glacier geometry into the raster's native Coordinate Reference System (CRS). It applies a `SAFETY_MARGIN_FRACTION` (15%) around the bounding box, extracts the window, and calculates the exact intersection.
- **Occupancy Check:** Ensures that the glacier geometry occupies at least `MIN_GLACIER_OCCUPANCY` (10%) of the cropped tile.
- **Resizing:** Resamples the imagery down to a standard tensor grid (e.g., 512x512 pixels) using bilinear interpolation.
- **Verification (`verify_geotiff`):** Checks the structural integrity of the written GeoTIFF (correct dimensions, non-empty, finite float values).

### Phase 5: Folder Structure
**Goal:** Organized data persistence.
- This is intrinsically handled during Phase 4 and Phase 7. The final structured path format is: `output_dir / {split} / {region} / {glacier_id} / {band} / {year}_{season}_{date}.tif`.
- Temporary files are kept in a `.tmp` directory to prevent partial state corruption. 

### Phase 6: Metadata Generation (`write_glacier_metadata` & `build_global_manifest`)
**Goal:** Create easily parsable manifests for PyTorch/TensorFlow dataloaders.
- **Per-Glacier Metadata:** Generates a `metadata.json` inside each glacier's folder documenting its ID, region, area, valid years, and expected bands.
- **Global Manifest:** `build_global_manifest` reads the completed state database and exports a large `dataset_manifest.parquet` file containing a tabular registry of every successfully downloaded image, band, and date across the entire dataset.

### Phase 7: Persistent State Database (`StateDB` class)
**Goal:** Provide the robust, resumable backbone of the pipeline.
- Uses Python's native `sqlite3` to maintain `dataset_state.db`.
- **Tables:**
  - `glaciers`: Tracks the overall processing status of each glacier.
  - `timestamps`: Tracks the status of each of the 36 (year, season) acquisitions.
  - `bands`: Tracks the download status of individual spectral bands (e.g., B02).
- Ensures that if the script crashes on image 15 of 36, restarting the script will bypass the first 14, resume the 15th, and continue seamlessly.
- Allows the orchestrator (`process_timestamp`) to atomically handle rollbacks. If 5 bands download but the 6th fails, the candidate is discarded, local files are trashed, and it tries the next best candidate from Phase 3.

### Phase 8: Quality-Check Reports (`phase8_quality_reports`)
**Goal:** Produce analytical summaries of the finalized dataset.
- Aggregates state data to output various CSVs and JSONs in a `reports/` folder.
- Reports include:
  - Glacier count per region (`01_glacier_count_per_region.csv`).
  - Train/Val/Test split statistics.
  - Area and cloud cover distributions.
  - Missing season statistics (to see where STAC lacked data).
  - Overall storage usage and final dataset summary JSON.

---

## Key Functions Explained

- **`retry(...)`**: A utility decorator/function that wraps network or filesystem calls. If an error occurs, it waits with an exponential backoff (e.g., 2s, 4s, 8s...) up to `MAX_RETRIES` before fully failing. It prevents transient API errors from crashing the hours-long pipeline.
  
- **`StateDB.lock(...)`**: Uses SQLite's `BEGIN IMMEDIATE` to lock the database globally. Crucial if multiple processes are trying to write to the state concurrently, preventing race conditions.

- **`select_best_seasonal_acquisition(...)`**: A heuristic function. Because glaciers are heavily obscured by clouds and snow, picking the "best" image isn't just about the date. This function weights low cloud cover and high valid pixel count above purely being in the middle of the season.

- **`process_timestamp(...)`**: The core orchestrator for Phase 4. For a single point in time (e.g., Summer 2020), it takes the ranked list of candidate images from STAC. It loops through the candidates, attempting to download all 6 bands. If a band fails (e.g., corrupt file, URL 404), it cleans up the temp files and tries the *next* candidate image in the ranked list. Once all bands succeed, it moves them from `.tmp` to the final folder structure.

- **`run_pipeline(cfg)`**: The main entry point. It instantiates the `StateDB`, sets up logging, and sequentially executes requested phases using `if X in cfg.phases:` blocks. It integrates graceful error handling and limits execution for debugging if `--max-glaciers` is provided.
