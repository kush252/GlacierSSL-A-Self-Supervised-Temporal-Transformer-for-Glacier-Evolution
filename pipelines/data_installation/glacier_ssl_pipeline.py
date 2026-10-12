#!/usr/bin/env python3
"""
glacier_ssl_pipeline.py
========================

Robust, restart-safe pipeline for building a large-scale temporal multi-spectral
glacier evolution dataset for self-supervised learning, using:

  - GLIMS Glacier Inventory (polygons + metadata)
  - Sentinel-2 L2A imagery via Microsoft Planetary Computer (STAC)

The pipeline implements 8 phases:

  Phase 1  Glacier filtering (area, valid-year count, region balance, target size)
  Phase 2  Region-level train/val/test split (no glacier in >1 split)
  Phase 3  Temporal sampling (9 years x 4 seasons = 36 timestamps per glacier)
  Phase 4  Image processing (download, crop, reproject, resample, resize, save)
  Phase 5  Folder structure (split/region/glacier_id/BAND/*.tif)
  Phase 6  Metadata (per-glacier metadata.json + global dataset_manifest.parquet)
  Phase 7  Robust download manager (persistent SQLite state, resumable, atomic)
  Phase 8  Dataset quality-check reports

Design goals
------------
  * Fully resumable: a SQLite state database (dataset_state.db) tracks the
    status of every (glacier, year, season) timestamp. Re-running the script
    picks up exactly where it left off and never re-downloads completed work.
  * Atomic writes: every timestamp is written to a temp directory first and
    only moved into the final location once all 6 bands have been verified
    (file exists, correct dtype/shape, non-corrupt, all bands present).
  * Defensive networking: every Planetary Computer / STAC call is wrapped in
    retry-with-backoff logic; partial failures never corrupt previously
    completed data.
  * Single file, dependency-light where reasonably possible, heavily logged.

Usage
-----
    python glacier_ssl_pipeline.py --glims-zip /path/to/glims_db.zip \
        --output-dir /data/glacier_ssl --target-glaciers 12000

    # Resume an interrupted run (just run the same command again):
    python glacier_ssl_pipeline.py --output-dir /data/glacier_ssl --resume

    # Run only specific phases (useful for debugging / staged execution):
    python glacier_ssl_pipeline.py --output-dir /data/glacier_ssl --phases 1,2

Required third-party packages
------------------------------
    geopandas, shapely, pandas, pyarrow, numpy, rasterio, pystac-client,
    planetary-computer, requests, tqdm

Install with:
    pip install geopandas shapely pandas pyarrow numpy rasterio \
        pystac-client planetary-computer requests tqdm --break-system-packages
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import logging
import math
import os
os.environ["VSI_CACHE"] = "TRUE"
os.environ["VSI_CACHE_SIZE"] = "50000000"
os.environ["GDAL_DISABLE_READDIR_ON_OPEN"] = "EMPTY_DIR"
os.environ["GDAL_HTTP_MAX_RETRY"] = "3"
import shutil
import sqlite3
import sys
import threading
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, date
from pathlib import Path
from typing import Iterable, Optional
import planetary_computer
import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Optional heavy dependencies are imported lazily / defensively so that the
# script can at least be inspected / unit-tested without the full geospatial
# stack installed. Each is checked at runtime where actually used.
# ----------------------------------------------------------------------------
try:
    import geopandas as gpd
except ImportError:  # pragma: no cover
    gpd = None

try:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject, transform_bounds
    from rasterio.mask import mask as rio_mask
    from rasterio.windows import from_bounds as window_from_bounds
except ImportError:  # pragma: no cover
    rasterio = None

try:
    from pystac_client import Client as StacClient
    import planetary_computer as pc
except ImportError:  # pragma: no cover
    StacClient = None
    pc = None

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None




# ==============================================================================
# Constants
# ==============================================================================

PC_STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
SENTINEL2_COLLECTION = "sentinel-2-l2a"
GLIMS_DOWNLOAD_URL = "https://www.glims.org/download/current"  # GLIMS current snapshot

BANDS = ["B02", "B03", "B04", "B08", "B11", "B12","SCL"]
TILE_SIZE = 512

SEASONS = {
    "spring": {"months": (3, 4, 5), "midpoint_doy": 105},   # mid-April
    "summer": {"months": (6, 7, 8), "midpoint_doy": 197},   # mid-July
    "autumn": {"months": (9, 10, 11), "midpoint_doy": 289}, # mid-October
    "winter": {"months": (12, 1, 2), "midpoint_doy": 15},   # mid-January (wraps year)
}
SEASON_ORDER = ["spring", "summer", "autumn", "winter"]

MIN_AREA_KM2 = 1.0
MAX_AREA_KM2 = 1000.0
MIN_VALID_YEARS = 9
REQUIRED_YEARS = 9
REQUIRED_TIMESTAMPS = 36  # 9 years * 4 seasons
CLOUD_COVER_THRESHOLD = 20.0  # percent, for "valid year" determination
TARGET_GLACIER_COUNT = 10
TRAIN_REGION_FRACTION = 0.70  # cumulative glacier-count fraction for train regions
VAL_FRACTION_OF_REMAINDER = 0.20  # of remaining regions (not glaciers)
SAFETY_MARGIN_FRACTION = 0.15  # 15% buffer added to glacier bbox when cropping
MIN_GLACIER_OCCUPANCY = 0.10
MAX_RETRIES = 5
RETRY_BACKOFF_BASE = 2.0  # seconds, exponential backoff
BYTES_PER_PIXEL = 2
STATES = ("PENDING", "DOWNLOADING", "COMPLETE", "FAILED")
PHASE1_EVALUATION_STATUSES = ("COMPLETE", "PERMANENT_REJECT", "TEMPORARY_FAILURE", "PENDING")
PIPELINE_CACHE_SCHEMA_VERSION = "phase1-cache-v1"
GLOBAL_CACHE_STATUSES = ("accepted", "rejected", "processing")
GLOBAL_CACHE_PATH = Path(__file__).resolve().parent / "global_glacier_cache.db"


# ==============================================================================
# Logging
# ==============================================================================

def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "pipeline.log"
    logger = logging.getLogger("glacier_ssl")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


LOG = logging.getLogger("glacier_ssl")  # reconfigured in main()


# ==============================================================================
# Retry helper
# ==============================================================================

def retry(
    fn,
    *args,
    max_retries: int = MAX_RETRIES,
    base_delay: float = RETRY_BACKOFF_BASE,
    exceptions: tuple = (Exception,),
    on_fail_return=None,
    description: str = "operation",
    **kwargs,
):
    """Call fn(*args, **kwargs) with exponential-backoff retries.

    Returns fn's result, or `on_fail_return` if all retries are exhausted
    (failure is logged, never raised, so the pipeline can keep going).
    """
    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            return fn(*args, **kwargs)
        except exceptions as exc:  # noqa: BLE001
            last_exc = exc
            delay = base_delay * (2 ** (attempt - 1))
            LOG.warning(
                "Attempt %d/%d failed for %s: %s (retrying in %.1fs)",
                attempt, max_retries, description, exc, delay,
            )
            time.sleep(delay)
    LOG.error("All %d attempts failed for %s: %s", max_retries, description, last_exc)
    return on_fail_return


# ==============================================================================
# Phase 7 (defined early since other phases use it): Persistent state database
# ==============================================================================

class StateDB:
    """SQLite-backed persistent state tracker for resumable dataset generation.

    Tables
    ------
    glaciers(glacier_id PK, region, db_area, split, status, updated_at)
    phase1_glacier_evaluations(glacier_id, pipeline_version, evaluation_status,
               failure_reason, timestamp, result_json,
               PRIMARY KEY(glacier_id, pipeline_version))
    timestamps(glacier_id, year, season, status, date, cloud_cover, updated_at,
               PRIMARY KEY(glacier_id, year, season))
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=60,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=60000;")
        self._init_schema()

    def _init_schema(self):
        with self._lock:
            cur = self._conn.cursor()
            cur.executescript(
                """
                CREATE TABLE IF NOT EXISTS glaciers (
                    glacier_id TEXT PRIMARY KEY,
                    region TEXT,
                    db_area REAL,
                    split TEXT,
                    status TEXT DEFAULT 'PENDING',
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS phase1_glacier_evaluations (
                    glacier_id TEXT NOT NULL,
                    evaluation_status TEXT NOT NULL,
                    failure_reason TEXT,
                    pipeline_version TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    result_json TEXT,
                    PRIMARY KEY (glacier_id, pipeline_version),
                    CHECK (evaluation_status IN ('COMPLETE', 'PERMANENT_REJECT', 'TEMPORARY_FAILURE', 'PENDING'))
                );

                CREATE TABLE IF NOT EXISTS timestamps (
                    glacier_id TEXT,
                    year INTEGER,
                    season TEXT,
                    status TEXT DEFAULT 'PENDING',
                    acq_date TEXT,
                    cloud_cover REAL,
                    updated_at TEXT,
                    PRIMARY KEY (glacier_id, year, season)
                );

                CREATE TABLE IF NOT EXISTS bands (
                    glacier_id TEXT,
                    year INTEGER,
                    season TEXT,
                    band TEXT,
                    status TEXT DEFAULT 'PENDING',
                    file_path TEXT,
                    updated_at TEXT,
                    PRIMARY KEY (glacier_id, year, season, band)
                );

                CREATE INDEX IF NOT EXISTS idx_phase1_eval_status
                    ON phase1_glacier_evaluations(pipeline_version, evaluation_status);
                CREATE INDEX IF NOT EXISTS idx_ts_glacier ON timestamps(glacier_id);
                CREATE INDEX IF NOT EXISTS idx_bands_ts ON bands(glacier_id, year, season);
                """
            )

    @contextmanager
    def lock(self):
        """Coarse cross-process lock using SQLite's own locking via BEGIN IMMEDIATE."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE;")
            try:
                yield
                self._conn.execute("COMMIT;")
            except Exception:
                self._conn.execute("ROLLBACK;")
                raise

    def upsert_glacier(self, glacier_id: str, region: str, db_area: float, split: str = None):
        now = datetime.utcnow().isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO glaciers (glacier_id, region, db_area, split, status, updated_at)
                VALUES (?, ?, ?, ?, COALESCE((SELECT status FROM glaciers WHERE glacier_id=?), 'PENDING'), ?)
                ON CONFLICT(glacier_id) DO UPDATE SET
                    region=excluded.region, db_area=excluded.db_area,
                    split=COALESCE(excluded.split, glaciers.split), updated_at=excluded.updated_at
                """,
                (glacier_id, region, db_area, split, glacier_id, now),
            )

    def set_glacier_status(self, glacier_id: str, status: str):
        assert status in STATES
        with self._lock:
            self._conn.execute(
                "UPDATE glaciers SET status=?, updated_at=? WHERE glacier_id=?",
                (status, datetime.utcnow().isoformat(), glacier_id),
            )

    def get_glacier_status(self, glacier_id: str) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM glaciers WHERE glacier_id=?", (glacier_id,)
            ).fetchone()
        return row[0] if row else None

    def set_phase1_evaluation(
        self,
        glacier_id: str,
        evaluation_status: str,
        failure_reason: Optional[str],
        pipeline_version: str,
        result: Optional[dict] = None,
    ):
        assert evaluation_status in PHASE1_EVALUATION_STATUSES
        result_json = json.dumps(result) if result is not None else None
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO phase1_glacier_evaluations
                    (glacier_id, evaluation_status, failure_reason, pipeline_version, timestamp, result_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(glacier_id, pipeline_version) DO UPDATE SET
                    evaluation_status=excluded.evaluation_status,
                    failure_reason=excluded.failure_reason,
                    timestamp=excluded.timestamp,
                    result_json=excluded.result_json
                """,
                (
                    glacier_id,
                    evaluation_status,
                    failure_reason,
                    pipeline_version,
                    datetime.utcnow().isoformat(),
                    result_json,
                ),
            )

    def get_phase1_evaluation(self, glacier_id: str, pipeline_version: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT evaluation_status, failure_reason, pipeline_version, timestamp, result_json
                FROM phase1_glacier_evaluations
                WHERE glacier_id=? AND pipeline_version=?
                """,
                (glacier_id, pipeline_version),
            ).fetchone()
        if not row:
            return None
        result = json.loads(row[4]) if row[4] else None
        return {
            "evaluation_status": row[0],
            "failure_reason": row[1],
            "pipeline_version": row[2],
            "timestamp": row[3],
            "result": result,
        }

    def set_timestamp_status(self, glacier_id, year, season, status, acq_date=None, cloud_cover=None):
        assert status in STATES
        now = datetime.utcnow().isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO timestamps (glacier_id, year, season, status, acq_date, cloud_cover, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(glacier_id, year, season) DO UPDATE SET
                    status=excluded.status,
                    acq_date=COALESCE(excluded.acq_date, timestamps.acq_date),
                    cloud_cover=COALESCE(excluded.cloud_cover, timestamps.cloud_cover),
                    updated_at=excluded.updated_at
                """,
                (glacier_id, year, season, status, acq_date, cloud_cover, now),
            )

    def get_timestamp_status(self, glacier_id, year, season) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT status FROM timestamps WHERE glacier_id=? AND year=? AND season=?",
                (glacier_id, year, season),
            ).fetchone()
        return row[0] if row else None

    def set_band_status(self, glacier_id, year, season, band, status, file_path=None):
        assert status in STATES
        now = datetime.utcnow().isoformat()
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO bands (glacier_id, year, season, band, status, file_path, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(glacier_id, year, season, band) DO UPDATE SET
                    status=excluded.status,
                    file_path=COALESCE(excluded.file_path, bands.file_path),
                    updated_at=excluded.updated_at
                """,
                (glacier_id, year, season, band, status, file_path, now),
            )

    def count_complete_timestamps(self, glacier_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM timestamps WHERE glacier_id=? AND status='COMPLETE'",
                (glacier_id,),
            ).fetchone()
        return row[0] if row else 0

    def bands_complete(self, glacier_id, year, season) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) FROM bands WHERE glacier_id=? AND year=? AND season=? AND status='COMPLETE'",
                (glacier_id, year, season),
            ).fetchone()
        return (row[0] if row else 0) == len(BANDS)

    def manifest_rows(self) -> list[dict]:
        with self._lock:
            cur = self._conn.execute(
                """
                SELECT b.glacier_id, g.region, g.split, g.db_area, b.year, b.season,
                       t.acq_date, t.cloud_cover, b.band, b.file_path
                FROM bands b
                JOIN glaciers g ON g.glacier_id = b.glacier_id
                JOIN timestamps t ON t.glacier_id = b.glacier_id AND t.year = b.year AND t.season = b.season
                WHERE b.status = 'COMPLETE'
                """
            )
            cols = [c[0] for c in cur.description]
            rows = cur.fetchall()
        return [dict(zip(cols, row)) for row in rows]

    def close(self):
        with self._lock:
            self._conn.close()


class GlobalGlacierCacheDB:
    """Cross-run SQLite cache for glacier-level terminal outcomes and expensive metadata."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(
            str(self.db_path),
            timeout=60,
            isolation_level=None,
            check_same_thread=False,
        )
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.execute("PRAGMA busy_timeout=60000;")
        self._init_schema()
        self.skipped_accepted = 0
        self.skipped_rejected = 0
        self.new_accepted = 0
        self.new_rejected = 0

    def _init_schema(self):
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS glacier_cache (
                    glacier_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    failed_phase INTEGER,
                    failure_reason TEXT,
                    region TEXT,
                    db_area REAL,
                    valid_year_count INTEGER,
                    valid_years TEXT,
                    available_years TEXT,
                    cloud_statistics TEXT,
                    seasonal_observation_count INTEGER,
                    processed_at TEXT NOT NULL,
                    is_complete INTEGER DEFAULT 0,
                    CHECK (status IN ('accepted', 'rejected', 'processing'))
                );

                CREATE INDEX IF NOT EXISTS idx_global_glacier_cache_status
                    ON glacier_cache(status);
                """
            )
            columns = {
                row[1]
                for row in self._conn.execute("PRAGMA table_info(glacier_cache)").fetchall()
            }
            if "valid_years" not in columns:
                self._conn.execute("ALTER TABLE glacier_cache ADD COLUMN valid_years TEXT")

    def summary_counts(self) -> dict[str, int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT status, COUNT(*) FROM glacier_cache GROUP BY status"
            ).fetchall()
        counts = {status: 0 for status in GLOBAL_CACHE_STATUSES}
        counts.update({status: count for status, count in rows})
        return counts

    def get(self, glacier_id: str) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(
                """
                SELECT glacier_id, status, failed_phase, failure_reason, region, db_area,
                       valid_year_count, valid_years, available_years, cloud_statistics,
                       seasonal_observation_count, processed_at, is_complete
                FROM glacier_cache
                WHERE glacier_id=?
                """,
                (glacier_id,),
            ).fetchone()
        if not row:
            return None
        return {
            "glacier_id": row[0],
            "status": row[1],
            "failed_phase": row[2],
            "failure_reason": row[3],
            "region": row[4],
            "db_area": row[5],
            "valid_year_count": row[6],
            "valid_years": json.loads(row[7]) if row[7] else None,
            "available_years": json.loads(row[8]) if row[8] else None,
            "cloud_statistics": json.loads(row[9]) if row[9] else None,
            "seasonal_observation_count": row[10],
            "processed_at": row[11],
            "is_complete": bool(row[12]),
        }

    def upsert(
        self,
        glacier_id: str,
        status: str,
        failed_phase: Optional[int] = None,
        failure_reason: Optional[str] = None,
        region: Optional[str] = None,
        db_area: Optional[float] = None,
        valid_years: Optional[list[int]] = None,
        available_years: Optional[list[int]] = None,
        cloud_statistics: Optional[dict] = None,
        seasonal_observation_count: Optional[int] = None,
        is_complete: bool = False,
    ):
        assert status in GLOBAL_CACHE_STATUSES
        existing = self.get(glacier_id)
        if status == "accepted" and (existing is None or existing.get("status") != "accepted"):
            self.new_accepted += 1
        if status == "rejected" and (existing is None or existing.get("status") != "rejected"):
            self.new_rejected += 1

        valid_year_count = len(valid_years) if valid_years is not None else None
        valid_years_json = json.dumps([int(year) for year in valid_years]) if valid_years is not None else None
        available_json = json.dumps([int(year) for year in available_years]) if available_years is not None else None
        cloud_json = json.dumps(cloud_statistics, sort_keys=True) if cloud_statistics is not None else None

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO glacier_cache (
                    glacier_id, status, failed_phase, failure_reason, region, db_area,
                    valid_year_count, valid_years, available_years, cloud_statistics,
                    seasonal_observation_count, processed_at, is_complete
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(glacier_id) DO UPDATE SET
                    status=excluded.status,
                    failed_phase=excluded.failed_phase,
                    failure_reason=excluded.failure_reason,
                    region=COALESCE(excluded.region, glacier_cache.region),
                    db_area=COALESCE(excluded.db_area, glacier_cache.db_area),
                    valid_year_count=COALESCE(excluded.valid_year_count, glacier_cache.valid_year_count),
                    valid_years=COALESCE(excluded.valid_years, glacier_cache.valid_years),
                    available_years=COALESCE(excluded.available_years, glacier_cache.available_years),
                    cloud_statistics=COALESCE(excluded.cloud_statistics, glacier_cache.cloud_statistics),
                    seasonal_observation_count=COALESCE(
                        excluded.seasonal_observation_count,
                        glacier_cache.seasonal_observation_count
                    ),
                    processed_at=excluded.processed_at,
                    is_complete=excluded.is_complete
                """,
                (
                    glacier_id,
                    status,
                    failed_phase,
                    failure_reason,
                    region,
                    db_area,
                    valid_year_count,
                    valid_years_json,
                    available_json,
                    cloud_json,
                    seasonal_observation_count,
                    datetime.utcnow().isoformat(),
                    1 if is_complete else 0,
                ),
            )

    def mark_processing(
        self,
        glacier_id: str,
        region: Optional[str] = None,
        db_area: Optional[float] = None,
        valid_years: Optional[list[int]] = None,
        available_years: Optional[list[int]] = None,
        cloud_statistics: Optional[dict] = None,
        seasonal_observation_count: Optional[int] = None,
    ):
        self.upsert(
            glacier_id,
            "processing",
            region=region,
            db_area=db_area,
            valid_years=valid_years,
            available_years=available_years,
            cloud_statistics=cloud_statistics,
            seasonal_observation_count=seasonal_observation_count,
        )

    def mark_rejected(
        self,
        glacier_id: str,
        failed_phase: int,
        failure_reason: str,
        region: Optional[str] = None,
        db_area: Optional[float] = None,
        valid_years: Optional[list[int]] = None,
        available_years: Optional[list[int]] = None,
        cloud_statistics: Optional[dict] = None,
        seasonal_observation_count: Optional[int] = None,
    ):
        self.upsert(
            glacier_id,
            "rejected",
            failed_phase=failed_phase,
            failure_reason=failure_reason or "Unknown",
            region=region,
            db_area=db_area,
            valid_years=valid_years,
            available_years=available_years,
            cloud_statistics=cloud_statistics,
            seasonal_observation_count=seasonal_observation_count,
            is_complete=False,
        )

    def mark_accepted(
        self,
        glacier_id: str,
        region: Optional[str] = None,
        db_area: Optional[float] = None,
        valid_years: Optional[list[int]] = None,
        seasonal_observation_count: Optional[int] = None,
    ):
        self.upsert(
            glacier_id,
            "accepted",
            failed_phase=None,
            failure_reason=None,
            region=region,
            db_area=db_area,
            valid_years=valid_years,
            available_years=valid_years,
            seasonal_observation_count=seasonal_observation_count,
            is_complete=True,
        )

    def close(self):
        with self._lock:
            self._conn.close()


def summarize_year_items(year_items: dict[int, list[dict]]) -> tuple[list[int], dict, int]:
    available_years = sorted(int(year) for year, items in year_items.items() if items)
    cloud_values = [
        float(item["cloud_cover"])
        for items in year_items.values()
        for item in items
        if item.get("cloud_cover") is not None
    ]
    cloud_statistics = {
        "item_count": len(cloud_values),
        "min": min(cloud_values) if cloud_values else None,
        "max": max(cloud_values) if cloud_values else None,
        "mean": float(np.mean(cloud_values)) if cloud_values else None,
    }
    return available_years, cloud_statistics, len(cloud_values)


# ==============================================================================
# Phase 1: Glacier filtering
# ==============================================================================

def load_glims_inventory(glims_path: Path):
    """Load the GLIMS glacier inventory (shapefile / geopackage / geojson) into
    a GeoDataFrame with normalized columns: glacier_id, glacier_name, region,
    db_area, centroid, geometry, year (observation year).
    """
    if gpd is None:
        raise RuntimeError("geopandas is required for GLIMS loading. pip install geopandas")

    

    LOG.info("Loading GLIMS inventory from %s", glims_path)
    gdf = gpd.read_file(glims_path)

    # GLIMS attribute names vary by export; normalize defensively.
    rename_map = {}
    colset = {c.lower(): c for c in gdf.columns}
    candidates = {
        "glacier_id": ["glac_id", "glacier_id", "anlys_id", "line_type", "glims_id"],
        "glacier_name": ["glac_name", "name", "glacier_name"],
        "region": ["glims_rgi_o", "region", "rgi_region", "rgi_o1", "src_date","geog_area"],
        "db_area": ["db_area", "area", "rec_area"],
        "year": ["src_date", "date", "anlys_time", "obs_year"],
    }
    for std_name, opts in candidates.items():
        for opt in opts:
            if opt in colset:
                rename_map[colset[opt]] = std_name
                break

    gdf = gdf.rename(columns=rename_map)
    if "geog_area" in gdf.columns:
        gdf["region"] = gdf["geog_area"]
    for required in ["glacier_id", "region", "db_area"]:
        if required not in gdf.columns:
            raise ValueError(
                f"GLIMS source is missing a usable '{required}' column; "
                f"available columns: {list(gdf.columns)}"
            )

    if "year" in gdf.columns:
        gdf["year"] = pd.to_datetime(gdf["year"], errors="coerce").dt.year

    gdf["db_area"] = pd.to_numeric(gdf["db_area"], errors="coerce")
    gdf = gdf.dropna(subset=["glacier_id", "region", "db_area", "geometry"])
    gdf["glacier_id"] = gdf["glacier_id"].astype(str)
    gdf["region"] = gdf["region"].astype(str)

    if gdf.crs is None:
        LOG.warning("GLIMS layer has no CRS set; assuming EPSG:4326")
        gdf.set_crs(epsg=4326, inplace=True)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(epsg=4326)

    projected = gdf.to_crs("EPSG:3857")
    gdf["centroid"] = projected.centroid.to_crs(gdf.crs)

    # Collapse duplicate observations of the same glacier_id to the most recent.
    if "year" in gdf.columns:
        gdf = gdf.sort_values("year").drop_duplicates(subset="glacier_id", keep="last")
    else:
        gdf = gdf.drop_duplicates(subset="glacier_id", keep="first")

    LOG.info("Loaded %d unique glacier records across %d regions", len(gdf), gdf["region"].nunique())
    return gdf


def get_stac_client():
    if StacClient is None or pc is None:
        raise RuntimeError(
            "pystac-client and planetary-computer are required. "
            "pip install pystac-client planetary-computer"
        )
    return StacClient.open(PC_STAC_URL, modifier=pc.sign_inplace)


def query_sentinel2_years(client, geometry_geojson: dict, year_range=(2015, 2024)) -> Optional[dict[int, list[dict]]]:
    """Query Sentinel-2 L2A items intersecting a glacier geometry across years.

    Returns {year: [ {date, cloud_cover, item_id} , ... ]} restricted to items
    with cloud_cover < CLOUD_COVER_THRESHOLD (i.e. usable for "valid year"
    determination, and as raw candidates for Phase 3 seasonal selection).
    """
    results: dict[int, list[dict]] = {}

    def _search():
        return client.search(
            collections=[SENTINEL2_COLLECTION],
            intersects=geometry_geojson,
            datetime=f"{year_range[0]}-01-01/{year_range[1]}-12-31",
            query={"eo:cloud_cover": {"lt": 50}},  # broad pull; we filter below
            max_items=2000,
        )

    search = retry(_search, description="STAC search (year scan)", on_fail_return=None)
    if search is None:
        return None

    items = retry(lambda: list(search.items()), description="STAC item iteration", on_fail_return=None)
    if items is None:
        return None
    for item in items:
        try:
            dt = item.datetime
            cc = item.properties.get("eo:cloud_cover")
            if dt is None or cc is None:
                continue
            yr = dt.year
            results.setdefault(yr, []).append(
                {"date": dt.date().isoformat(), "cloud_cover": float(cc), "item_id": item.id}
            )
        except Exception as exc:  # noqa: BLE001
            LOG.debug("Skipping malformed STAC item: %s", exc)
    return results


def determine_valid_years(year_items: dict[int, list[dict]]) -> list[int]:
    valid = []
    for yr, items in year_items.items():
        if any(it["cloud_cover"] < CLOUD_COVER_THRESHOLD for it in items):
            valid.append(yr)
    return sorted(valid)


def phase1_filter_glaciers(
    gdf,
    client,
    cache_path: Path,
    state: StateDB,
    global_cache: GlobalGlacierCacheDB,
    pipeline_version: str,
    year_range=(2015, 2024),
):
    """Filter glaciers by area + valid-year count, balance by region, hit the
    target dataset size, and persist a per-glacier cache of valid years /
    candidate Sentinel-2 items so Phase 3 doesn't re-query STAC.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache: dict = {}
    if cache_path.exists():
        LOG.info("Loading existing STAC valid-year cache from %s", cache_path)
        with open(cache_path, "r") as f:
            cache = json.load(f)

    def area_band(df, lo, hi):
        return df[(df["db_area"] >= lo) & (df["db_area"] <= hi)]

    def evaluate(df_subset):
        """Returns list of dicts for glaciers passing the valid_years>=9 test,
        populating/using `cache` to avoid repeat STAC queries."""
        accepted = []
        for _, row in df_subset.iterrows():
            gid = row["glacier_id"]
            region = str(row["region"])
            db_area = float(row["db_area"])
            global_entry = global_cache.get(gid)
            if global_entry:
                if global_entry["status"] == "accepted":
                    global_cache.skipped_accepted += 1
                    LOG.info("Skipped accepted glacier %s", gid)
                    continue
                if global_entry["status"] == "rejected":
                    global_cache.skipped_rejected += 1
                    LOG.info("Skipped rejected glacier %s", gid)
                    continue

            cached_evaluation = state.get_phase1_evaluation(gid, pipeline_version)
            if cached_evaluation:
                status = cached_evaluation["evaluation_status"]
                if status == "COMPLETE":
                    result = cached_evaluation.get("result")
                    if result:
                        accepted.append(result)
                    LOG.debug("Phase 1 cache hit for %s: COMPLETE", gid)
                    continue
                if status in ("PERMANENT_REJECT", "TEMPORARY_FAILURE"):
                    LOG.debug(
                        "Phase 1 cache hit for %s: %s (%s)",
                        gid,
                        status,
                        cached_evaluation.get("failure_reason"),
                    )
                    continue

            state.set_phase1_evaluation(gid, "PENDING", None, pipeline_version)
            global_cache.mark_processing(gid, region=region, db_area=db_area)

            geom_obj = row["geometry"]
            if geom_obj is None or geom_obj.is_empty or not geom_obj.is_valid:
                state.set_phase1_evaluation(
                    gid,
                    "PERMANENT_REJECT",
                    "invalid geometry",
                    pipeline_version,
                )
                global_cache.mark_rejected(
                    gid,
                    failed_phase=1,
                    failure_reason="Invalid geometry",
                    region=region,
                    db_area=db_area,
                )
                continue

            try:
                if global_entry and global_entry["status"] == "processing" and global_entry.get("valid_years"):
                    valid_years = [int(year) for year in global_entry["valid_years"]]
                    year_items = None
                    LOG.debug("Loaded valid-year metadata for glacier %s from global cache", gid)
                elif gid in cache:
                    year_items = {int(k): v for k, v in cache[gid]["year_items"].items()}
                    valid_years = determine_valid_years(year_items)
                else:
                    geom = geom_obj.__geo_interface__
                    year_items = query_sentinel2_years(client, geom, year_range)
                    if year_items is None:
                        state.set_phase1_evaluation(
                            gid,
                            "TEMPORARY_FAILURE",
                            "STAC query or item iteration failed",
                            pipeline_version,
                        )
                        continue
                    cache[gid] = {"year_items": {str(k): v for k, v in year_items.items()}}
                    valid_years = determine_valid_years(year_items)
                    available_years, cloud_statistics, observation_count = summarize_year_items(year_items)
                    global_cache.mark_processing(
                        gid,
                        region=region,
                        db_area=db_area,
                        valid_years=valid_years,
                        available_years=available_years,
                        cloud_statistics=cloud_statistics,
                        seasonal_observation_count=observation_count,
                    )
                    # periodically flush cache to disk for crash safety
                    if len(cache) % 25 == 0:
                        with open(cache_path, "w") as f:
                            json.dump(cache, f)
                        LOG.info("Flushed STAC cache (%d glaciers scanned)", len(cache))
            except (KeyboardInterrupt, SystemExit) as exc:
                state.set_phase1_evaluation(
                    gid,
                    "TEMPORARY_FAILURE",
                    "interrupted execution",
                    pipeline_version,
                )
                raise
            except Exception as exc:  # noqa: BLE001
                state.set_phase1_evaluation(
                    gid,
                    "TEMPORARY_FAILURE",
                    f"unexpected exception during Phase 1 evaluation: {exc}",
                    pipeline_version,
                )
                LOG.warning("Temporary Phase 1 failure for glacier %s: %s", gid, exc)
                continue

            continuous_years = find_continuous_block(valid_years, REQUIRED_YEARS)

            if continuous_years:
                result = {
                    "glacier_id": str(gid),
                    "region": region,
                    "db_area": db_area,
                    "valid_years": [int(year) for year in continuous_years],
                }
                accepted.append(result)
                state.set_phase1_evaluation(
                    gid,
                    "COMPLETE",
                    None,
                    pipeline_version,
                    result=result,
                )
            else:
                state.set_phase1_evaluation(
                    gid,
                    "PERMANENT_REJECT",
                    f"fewer than REQUIRED_YEARS valid years ({len(valid_years)}/{REQUIRED_YEARS})",
                    pipeline_version,
                )
                global_cache.mark_rejected(
                    gid,
                    failed_phase=1,
                    failure_reason="Insufficient valid years",
                    region=region,
                    db_area=db_area,
                    valid_years=valid_years,
                    available_years=valid_years,
                )
        return accepted

    def find_continuous_block(years, length=9):
        years = sorted(set(years))

        for i in range(len(years) - length + 1):
            block = years[i:i+length]

            if block[-1] - block[0] == length - 1:
                return block

        return None

    LOG.info("Phase 1: scanning core area band [%s, %s] km^2", MIN_AREA_KM2, MAX_AREA_KM2)
    core = area_band(gdf, MIN_AREA_KM2, MAX_AREA_KM2)
    accepted = evaluate(core)

    with open(cache_path, "w") as f:
        json.dump(cache, f)

    LOG.info("Phase 1: %d glaciers passed area+valid-year filter (core band)", len(accepted))

    accepted_df = pd.DataFrame(accepted)
    if accepted_df.empty:
        outside_configured_area = gdf[
            (gdf["db_area"] < MIN_AREA_KM2) | (gdf["db_area"] > MAX_AREA_KM2)
        ]
        for _, row in outside_configured_area.iterrows():
            gid = row["glacier_id"]
            global_entry = global_cache.get(gid)
            if global_entry and global_entry["status"] in ("accepted", "rejected"):
                continue
            global_cache.mark_rejected(
                gid,
                failed_phase=1,
                failure_reason="Outside configured area range",
                region=str(row["region"]),
                db_area=float(row["db_area"]),
            )
        LOG.warning("No glaciers passed the core filter; nothing to expand from.")
        return accepted_df, pd.DataFrame()

    region_counts = (
        accepted_df.groupby("region").size().sort_values(ascending=False).rename("glacier_count")
    )
    total = region_counts.sum()
    cum_frac = (region_counts.cumsum() / total)
    train_regions = cum_frac[cum_frac <= TRAIN_REGION_FRACTION].index.tolist()
    if not train_regions:  # ensure at least the top region is included
        train_regions = [region_counts.index[0]]

    region_stats = pd.DataFrame(
        {
            "region": region_counts.index,
            "glacier_count": region_counts.values,
            "cumulative_percentage": (cum_frac.values * 100).round(2),
        }
    )

    # Expand toward target size if needed, preserving region balance, by
    # progressively relaxing the area band.
    if len(accepted_df) < TARGET_GLACIER_COUNT:
        LOG.info(
            "Phase 1: %d < target %d; expanding area band to include outliers",
            len(accepted_df), TARGET_GLACIER_COUNT,
        )
        expansion_bands = [
            (MAX_AREA_KM2, math.inf),  # large glaciers
            (0.0, MIN_AREA_KM2),       # small glaciers
        ]
        allowed_regions = set(region_counts.index)
        for lo, hi in expansion_bands:
            if len(accepted_df) >= TARGET_GLACIER_COUNT:
                break
            extra_pool = area_band(gdf, lo, hi)
            extra_pool = extra_pool[extra_pool["region"].isin(allowed_regions)]
            extra_pool = extra_pool[~extra_pool["glacier_id"].isin(accepted_df["glacier_id"])]

            batch_size = 500
            offset = 0

            while len(accepted_df) < TARGET_GLACIER_COUNT and offset < len(extra_pool):

                batch = extra_pool.iloc[offset:offset + batch_size]
                offset += batch_size

                extra_accepted = evaluate(batch)
                print(
                    f"Batch processed: {len(batch)} | "
                    f"Accepted this batch: {len(extra_accepted)} | "
                    f"Total accepted: {len(accepted_df)}"
                )
                if extra_accepted:
                    accepted_df = pd.concat(
                        [accepted_df, pd.DataFrame(extra_accepted)],
                        ignore_index=True,
                    )

                    accepted_df = accepted_df.drop_duplicates(
                        subset="glacier_id",
                        keep="first",
                    )

                    LOG.info(
                        "Accepted %d / %d glaciers",
                        len(accepted_df),
                        TARGET_GLACIER_COUNT,
                    )

                if len(accepted_df) >= TARGET_GLACIER_COUNT:
                    break
            with open(cache_path, "w") as f:
                json.dump(cache, f)

    accepted_ids = set(accepted_df["glacier_id"]) if not accepted_df.empty else set()
    outside_configured_area = gdf[
        ((gdf["db_area"] < MIN_AREA_KM2) | (gdf["db_area"] > MAX_AREA_KM2))
        & (~gdf["glacier_id"].isin(accepted_ids))
    ]
    for _, row in outside_configured_area.iterrows():
        gid = row["glacier_id"]
        cached_evaluation = state.get_phase1_evaluation(gid, pipeline_version)
        if cached_evaluation is None or cached_evaluation["evaluation_status"] == "PENDING":
            state.set_phase1_evaluation(
                gid,
                "PERMANENT_REJECT",
                "outside configured area range",
                pipeline_version,
            )
            global_cache.mark_rejected(
                gid,
                failed_phase=1,
                failure_reason="Outside configured area range",
                region=str(row["region"]),
                db_area=float(row["db_area"]),
            )

    if len(accepted_df) > TARGET_GLACIER_COUNT:
        # Trim proportionally per region to respect the target while keeping
        # region balance (largest regions trimmed first, proportionally).
        LOG.info("Phase 1: trimming %d -> target %d, preserving region balance",
                  len(accepted_df), TARGET_GLACIER_COUNT)
        frac = TARGET_GLACIER_COUNT / len(accepted_df)
        accepted_df = (
            accepted_df.groupby("region", group_keys=False)
            .apply(lambda g: g.sample(frac=frac, random_state=42) if len(g) > 1 else g)
            .reset_index(drop=True)
        )

    LOG.info("Phase 1 complete: %d glaciers selected across %d regions",
              len(accepted_df), accepted_df["region"].nunique())

    def estimate_storage(glacier_count):

        images_per_glacier = 36 * 6

        bytes_per_image = (
            TILE_SIZE
            * TILE_SIZE
            * BYTES_PER_PIXEL
        )

        total_bytes = (
            glacier_count
            * images_per_glacier
            * bytes_per_image
        )

        return total_bytes / (1024**4)

    estimated_tb = estimate_storage(len(accepted_df))

    LOG.info(
        "Estimated dataset size: %.2f TB",
        estimated_tb
    )

    print("=" * 80)
    print("accepted_df shape:", accepted_df.shape)
    print("accepted_df columns:", accepted_df.columns.tolist())

    print("region_stats shape:", region_stats.shape)
    print("region_stats columns:", region_stats.columns.tolist())

    print(accepted_df.head())
    print(region_stats.head())
    print("=" * 80)

    return accepted_df, region_stats


# ==============================================================================
# Phase 2: Region-level splits
# ==============================================================================




def phase2_region_split(accepted_df: pd.DataFrame, region_stats: pd.DataFrame, out_path: Path):
    region_counts = region_stats.set_index("region")["glacier_count"]
    sorted_regions = region_counts.sort_values(ascending=False)
    total = sorted_regions.sum()
    cum_frac = sorted_regions.cumsum() / total

    train_regions = cum_frac[cum_frac <= TRAIN_REGION_FRACTION].index.tolist()
    if not train_regions:
        train_regions = [sorted_regions.index[0]]

    remaining_regions = [r for r in sorted_regions.index if r not in train_regions]
    n_val = max(1, round(len(remaining_regions) * VAL_FRACTION_OF_REMAINDER)) if remaining_regions else 0
    val_regions = remaining_regions[:n_val]
    test_regions = remaining_regions[n_val:]

    region_to_split = {r: "train" for r in train_regions}
    region_to_split.update({r: "val" for r in val_regions})
    region_to_split.update({r: "test" for r in test_regions})

    accepted_df = accepted_df.copy()
    accepted_df["split"] = accepted_df["region"].map(region_to_split)
    # Any region not classified (shouldn't happen) defaults to train to avoid data loss.
    accepted_df["split"] = accepted_df["split"].fillna("train")

    manifest = accepted_df[["glacier_id", "region", "split", "db_area"]].copy()
    manifest.to_csv(out_path, index=False)

    LOG.info(
        "Phase 2 complete: train=%d glaciers (%d regions), val=%d glaciers (%d regions), "
        "test=%d glaciers (%d regions)",
        (manifest["split"] == "train").sum(), len(train_regions),
        (manifest["split"] == "val").sum(), len(val_regions),
        (manifest["split"] == "test").sum(), len(test_regions),
    )
    return manifest


# ==============================================================================
# Phase 3: Temporal sampling
# ==============================================================================

def _season_for_month(month: int) -> str:
    for season, info in SEASONS.items():
        if month in info["months"]:
            return season
    raise ValueError(f"Unmapped month: {month}")


def _winter_year_key(d: date) -> int:
    """Winter spans Dec-Feb; bucket Dec into the *following* year's winter so
    that 'winter 2020' = Dec 2019 + Jan/Feb 2020, matching common climatological
    convention and keeping each (year, season) bucket well-defined."""
    if d.month == 12:
        return d.year + 1
    return d.year


def select_best_seasonal_acquisition(candidates: list[dict], season: str, year: int) -> Optional[dict]:
    """Rank candidate Sentinel-2 acquisitions for a given (year, season) bucket
    and return the best one, per the ranking rules:
        1. lowest cloud cover
        2. highest valid pixel percentage
        3. lowest snow/cloud artifact score
        4. closest to seasonal midpoint
    candidates: list of dicts with keys date,cloud_cover,item_id,valid_pixel_pct,
    snow_cloud_artifact_score (the latter two default to neutral values when
    the STAC item doesn't expose them).
    """
    if not candidates:
        return None

    midpoint_doy = SEASONS[season]["midpoint_doy"]

    def doy_distance(d: date) -> int:
        doy = d.timetuple().tm_yday
        # Wrap-around distance on a 365-day circle.
        diff = abs(doy - midpoint_doy)
        return min(diff, 365 - diff)

    def sort_key(c):
        d = datetime.fromisoformat(c["date"]).date()
        valid_pct = c.get("valid_pixel_pct", 100.0)
        artifact_score = c.get("snow_cloud_artifact_score", 0.0)
        return (
            c["cloud_cover"],          # ascending: lower cloud cover first
            -valid_pct,                # ascending on negative => higher valid pct first
            artifact_score,            # ascending: lower artifact score first
            doy_distance(d),           # ascending: closer to midpoint first
        )

    ranked = sorted(candidates, key=sort_key)
    return ranked[0]


def query_seasonal_candidates(client, geometry_geojson: dict, year: int, season: str) -> list[dict]:
    months = SEASONS[season]["months"]
    if season == "winter":
        # Dec of (year-1) through Feb of (year)
        start = f"{year - 1}-12-01"
        end_year = year
        end = f"{end_year}-03-01"
    else:
        start_month = months[0]
        end_month = months[-1]
        start = f"{year}-{start_month:02d}-01"
        # end day: last day of end_month (use next month minus a day, simplified)
        end_month_next = end_month + 1
        end_year = year
        if end_month_next > 12:
            end_month_next = 1
            end_year += 1
        end = f"{end_year}-{end_month_next:02d}-01"

    def _search():
        return client.search(
            collections=[SENTINEL2_COLLECTION],
            intersects=geometry_geojson,
            datetime=f"{start}/{end}",
            query={"eo:cloud_cover": {"lt": 50}},
            max_items=1000,
        )

    search = retry(_search, description=f"STAC seasonal search {year}-{season}", on_fail_return=None)
    if search is None:
        return []

    items = retry(lambda: list(search.items()), description="STAC seasonal item iteration", on_fail_return=[])
    out = []
    for item in items:
        try:
            dt = item.datetime
            cc = item.properties.get("eo:cloud_cover")
            if dt is None or cc is None:
                continue
            if cc >= CLOUD_COVER_THRESHOLD:
                continue
            out.append(
                {
                    "date": dt.date().isoformat(),
                    "cloud_cover": float(cc),
                    "item_id": item.id,
                    "item_href": item.self_href,
                    "_item": item,  # keep object for downloading in Phase 4
                }
            )
        except Exception as exc:  # noqa: BLE001
            LOG.debug("Skipping malformed seasonal STAC item: %s", exc)
    return out


def phase3_select_timestamps(client, glacier_row: dict, valid_years: list[int]) -> dict:
    """For a glacier with >=9 valid_years, pick the most recent 9 and, for each,
    select the best acquisition per season. Returns:
      { (year, season): acquisition_dict_or_None }
    A glacier is only "complete" if all 36 entries are non-None.
    """
    chosen_years = sorted(valid_years)[-REQUIRED_YEARS:]
    geom = glacier_row["geometry"]
    geom_geojson = geom.__geo_interface__

    schedule = {}
    for year in chosen_years:
        for season in SEASON_ORDER:
            candidates = query_seasonal_candidates(client, geom_geojson, year, season)
            best = select_best_seasonal_acquisition(candidates, season, year)
            # "If best acquisition fails, select next candidate" is handled
            # downstream during Phase 4 download (we keep the ranked list).
            schedule[(year, season)] = {
                "best": best,
                "ranked_candidates": sorted(
                    candidates,
                    key=lambda c: (c["cloud_cover"],),
                ) if candidates else [],
            }
    return schedule


# ==============================================================================
# Phase 4: Image processing
# ==============================================================================

def add_safety_margin(bounds: tuple[float, float, float, float], frac: float = SAFETY_MARGIN_FRACTION):
    minx, miny, maxx, maxy = bounds
    dx = (maxx - minx) * frac
    dy = (maxy - miny) * frac
    return (minx - dx, miny - dy, maxx + dx, maxy + dy)


def download_and_process_band(item, band: str, glacier_geom, dest_path: Path) -> bool:
    """Download one Sentinel-2 band asset, crop to glacier geometry (+margin),
    reproject/resample onto a common grid, resize to TILE_SIZE x TILE_SIZE, and
    write as GeoTIFF, preserving raw reflectance values (no normalization).
    Returns True on success.
    """
    if rasterio is None:
        raise RuntimeError("rasterio is required for image processing. pip install rasterio")

    signed_item = planetary_computer.sign(item)

    asset = signed_item.assets.get(band)
    if asset is None:
        LOG.warning("Item %s missing asset %s", signed_item.id, band)
        return False

    href = asset.href
    LOG.info("Band URL: %s", href)

    from rasterio.features import geometry_mask

    def glacier_occupancy_fraction(glacier_geom, transform, shape):

        mask = geometry_mask(
            [glacier_geom],
            transform=transform,
            invert=True,
            out_shape=shape
        )

        return mask.mean()

    def _open_and_process():
        with rasterio.open(href) as src:
            # Reproject glacier geometry to the raster CRS for cropping.
            geom_in_raster_crs = gpd.GeoSeries([glacier_geom], crs="EPSG:4326").to_crs(src.crs).iloc[0]
            minx, miny, maxx, maxy = geom_in_raster_crs.bounds
            minx, miny, maxx, maxy = add_safety_margin((minx, miny, maxx, maxy))

            window = window_from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            window = window.round_offsets().round_lengths()
            if window.width <= 0 or window.height <= 0:
                raise ValueError("Empty crop window (glacier outside raster extent)")

            data = src.read(
                1,
                window=window,
                out_shape=(TILE_SIZE, TILE_SIZE),
                resampling=Resampling.bilinear,
            )
            
            transform = src.window_transform(window)

            scale_x = window.width / TILE_SIZE
            scale_y = window.height / TILE_SIZE

            out_transform = transform * rasterio.Affine.scale(
                scale_x,
                scale_y
            )

            occupancy = glacier_occupancy_fraction(
                geom_in_raster_crs,
                out_transform,
                (TILE_SIZE, TILE_SIZE)
            )

            if occupancy < MIN_GLACIER_OCCUPANCY:
                raise ValueError(
                    f"Low glacier occupancy: {occupancy:.3f}"
                )

            profile = src.profile.copy()
            profile.update(
                {
                    "height": TILE_SIZE,
                    "width": TILE_SIZE,
                    "transform": out_transform,
                    "count": 1,
                    "dtype": data.dtype,
                    "driver": "GTiff",
                    "compress": "deflate",
                }
            )

            tmp_path = dest_path.with_suffix(".tmp.tif")
            with rasterio.open(tmp_path, "w", **profile) as dst:
                dst.write(data, 1)

            if not verify_geotiff(tmp_path):
                tmp_path.unlink(missing_ok=True)
                raise ValueError("Post-write verification failed")

            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(tmp_path), str(dest_path))  # atomic on same filesystem
            return True

    ok = retry(_open_and_process, description=f"download/process {band} for item {item.id}",
               on_fail_return=False)
    return bool(ok)


def verify_geotiff(path: Path) -> bool:
    """Integrity validation: file exists, correct dimensions, readable / not
    corrupt."""
    if rasterio is None:
        return path.exists() and path.stat().st_size > 0
    if not path.exists() or path.stat().st_size == 0:
        return False
    try:
        with rasterio.open(path) as src:
            if src.width != TILE_SIZE or src.height != TILE_SIZE:
                return False
            arr = src.read(1)
            if arr is None or arr.size == 0:
                return False
            if not np.isfinite(arr.astype("float64")).any():
                return False
        return True
    except Exception as exc:  # noqa: BLE001
        LOG.debug("Corrupt GeoTIFF %s: %s", path, exc)
        return False


def process_timestamp(
    state: StateDB,
    glacier_id: str,
    glacier_geom,
    region: str,
    split: str,
    year: int,
    season: str,
    ranked_candidates: list[dict],
    output_root: Path,
) -> bool:
    """Try each ranked candidate acquisition in order until one yields all 6
    bands successfully (atomic per-timestamp write). Updates state DB
    throughout. Returns True iff the timestamp is COMPLETE.
    """
    if state.bands_complete(glacier_id, year, season):
        state.set_timestamp_status(glacier_id, year, season, "COMPLETE")
        return True

    state.set_timestamp_status(glacier_id, year, season, "DOWNLOADING")

    final_band_dir_template = output_root / split / region / glacier_id  # band subdirs inside

    for candidate in ranked_candidates:
        item = candidate.get("_item")
        if item is None:
            continue
        date_str = candidate["date"]
        cloud_cover = candidate["cloud_cover"]
        filename = f"{year}_{season}_{date_str}.tif"

        tmp_dir = output_root / ".tmp" / glacier_id / f"{year}_{season}"
        tmp_dir.mkdir(parents=True, exist_ok=True)

        all_ok = True
        produced_paths = {}
        for band in BANDS:
            tmp_band_path = tmp_dir / f"{band}_{filename}"
            ok = download_and_process_band(item, band, glacier_geom, tmp_band_path)
            if not ok:
                all_ok = False
                state.set_band_status(glacier_id, year, season, band, "FAILED")
                break
            produced_paths[band] = tmp_band_path
            state.set_band_status(glacier_id, year, season, band, "DOWNLOADING", file_path=str(tmp_band_path))

        if not all_ok:
            # discard entire timestamp, try next candidate
            shutil.rmtree(tmp_dir, ignore_errors=True)
            LOG.info("Candidate %s failed for %s %s-%s; trying next candidate",
                      candidate.get("item_id"), glacier_id, year, season)
            continue

        # All bands succeeded for this candidate -> atomically move into place.
        final_ok = True
        final_paths = {}
        for band, tmp_band_path in produced_paths.items():
            final_band_dir = final_band_dir_template / band
            final_band_dir.mkdir(parents=True, exist_ok=True)
            final_path = final_band_dir / filename
            try:

                if not verify_geotiff(tmp_band_path):
                    final_ok = False
                    break
                shutil.move(str(tmp_band_path), str(final_path))
                final_paths[band] = final_path
            except Exception as exc:  # noqa: BLE001
                LOG.error("Failed to finalize band %s for %s %s-%s: %s", band, glacier_id, year, season, exc)
                final_ok = False
                break

        shutil.rmtree(tmp_dir, ignore_errors=True)

        if final_ok and len(final_paths) == len(BANDS):
            for band, p in final_paths.items():
                state.set_band_status(glacier_id, year, season, band, "COMPLETE", file_path=str(p))
            state.set_timestamp_status(glacier_id, year, season, "COMPLETE", acq_date=date_str, cloud_cover=cloud_cover)
            return True
        else:
            # roll back any partially finalized files for this candidate
            for p in final_paths.values():
                Path(p).unlink(missing_ok=True)
            for band in BANDS:
                state.set_band_status(glacier_id, year, season, band, "FAILED")
            continue

    # exhausted all candidates without success
    state.set_timestamp_status(glacier_id, year, season, "FAILED")
    return False


# ==============================================================================
# Phase 6: Metadata
# ==============================================================================

def write_glacier_metadata(output_root: Path, split: str, region: str, glacier_id: str,
                            db_area: float, years: list[int]):
    meta = {
        "glacier_id": glacier_id,
        "region": region,
        "db_area": db_area,
        "years": years,
        "num_years": len(years),
        "timestamps": REQUIRED_TIMESTAMPS,
        "bands": BANDS,
    }
    glacier_dir = output_root / split / region / glacier_id
    glacier_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = glacier_dir / "metadata.json.tmp"
    final_path = glacier_dir / "metadata.json"
    with open(tmp_path, "w") as f:
        json.dump(meta, f, indent=2)
    shutil.move(str(tmp_path), str(final_path))


def build_global_manifest(state: StateDB, output_root: Path):
    rows = state.manifest_rows()
    if not rows:
        LOG.warning("No completed records found; skipping global manifest write.")
        return
    df = pd.DataFrame(rows)
    df = df.rename(columns={"acq_date": "date"})
    df["timestamp"] = df["year"].astype(str) + "_" + df["season"]
    df = df[
        ["glacier_id", "region", "split", "db_area", "timestamp", "season", "date",
         "cloud_cover", "band", "file_path"]
    ]
    out_path = output_root / "dataset_manifest.parquet"
    tmp_path = output_root / "dataset_manifest.parquet.tmp"
    df.to_parquet(tmp_path, index=False)
    shutil.move(str(tmp_path), str(out_path))
    LOG.info("Wrote global manifest with %d rows to %s", len(df), out_path)


# ==============================================================================
# Phase 8: Quality-check reports
# ==============================================================================

def phase8_quality_reports(state: StateDB, region_stats: pd.DataFrame, split_manifest: pd.DataFrame,
                            output_root: Path):
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows = state.manifest_rows()
    df = pd.DataFrame(rows)

    # 1. Glacier count per region
    region_stats.to_csv(reports_dir / "01_glacier_count_per_region.csv", index=False)

    # 2. Split statistics
    split_stats = split_manifest.groupby("split").agg(
        glacier_count=("glacier_id", "nunique"), regions=("region", "nunique")
    ).reset_index()
    split_stats.to_csv(reports_dir / "02_split_statistics.csv", index=False)

    # 3. Area distribution
    area_stats = split_manifest["db_area"].describe().to_frame("db_area_km2")
    area_stats.to_csv(reports_dir / "03_area_distribution.csv")

    if not df.empty:
        print(df.columns.tolist())
        print(df.head())
        # 4. Cloud cover distribution
        cc = df.drop_duplicates(
    subset=["glacier_id", "year", "season"]
)["cloud_cover"]
        cc.describe().to_frame("cloud_cover_pct").to_csv(reports_dir / "04_cloud_cover_distribution.csv")

        # 5. Year distribution
        year_counts = df.drop_duplicates(subset=["glacier_id", "year", "season"])["year"]
        year_counts.value_counts().sort_index().to_csv(reports_dir / "05_year_distribution.csv")

        # 6. Missing season statistics (per glacier, which seasons are short of full coverage)
        per_glacier_ts = df.drop_duplicates(subset=["glacier_id", "year", "season"])
        ts_counts = per_glacier_ts.groupby("glacier_id").size().rename("completed_timestamps")
        missing = (REQUIRED_TIMESTAMPS - ts_counts).clip(lower=0).rename("missing_timestamps")
        missing.to_csv(reports_dir / "06_missing_season_statistics.csv")

        # 7. Download success rate (glacier-level)
        total_glaciers = split_manifest["glacier_id"].nunique()
        complete_glaciers = (ts_counts >= REQUIRED_TIMESTAMPS).sum()
        success_rate = pd.DataFrame(
            [{"total_glaciers": total_glaciers, "complete_glaciers": int(complete_glaciers),
              "success_rate_pct": round(100 * complete_glaciers / total_glaciers, 2) if total_glaciers else 0.0}]
        )
        success_rate.to_csv(reports_dir / "07_download_success_rate.csv", index=False)
    else:
        LOG.warning("No completed bands yet; reports 4-7 will be skipped/empty.")

    # 8. Storage usage
    total_bytes = sum(f.stat().st_size for f in output_root.rglob("*.tif") if f.is_file())
    storage = pd.DataFrame([{"total_size_gb": round(total_bytes / (1024 ** 3), 3)}])
    storage.to_csv(reports_dir / "08_storage_usage.csv", index=False)

    # 9. Final dataset summary
    summary = {
        "target_glaciers": TARGET_GLACIER_COUNT,
        "selected_glaciers": int(split_manifest["glacier_id"].nunique()),
        "regions": int(split_manifest["region"].nunique()),
        "bands": BANDS,
        "tile_size": TILE_SIZE,
        "required_years": REQUIRED_YEARS,
        "required_timestamps": REQUIRED_TIMESTAMPS,
        "generated_at": datetime.utcnow().isoformat(),
    }
    with open(reports_dir / "09_final_dataset_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    LOG.info("Phase 8 complete: reports written to %s", reports_dir)


# ==============================================================================
# Orchestration
# ==============================================================================

@dataclasses.dataclass
class PipelineConfig:
    output_dir: Path
    glims_path: Optional[Path] = None
    target_glaciers: int = TARGET_GLACIER_COUNT
    year_range: tuple = (2015, 2024)
    phases: tuple = (1, 2, 3, 4, 5, 6, 7, 8)
    max_glaciers_to_process: Optional[int] = None  # debugging knob
    pipeline_version: Optional[str] = None
    global_cache_path: Path = GLOBAL_CACHE_PATH


def build_pipeline_version(cfg: PipelineConfig) -> str:
    payload = {
        "schema": PIPELINE_CACHE_SCHEMA_VERSION,
        "year_range": tuple(cfg.year_range),
        "target_glaciers": cfg.target_glaciers,
        "min_area_km2": MIN_AREA_KM2,
        "max_area_km2": MAX_AREA_KM2,
        "required_years": REQUIRED_YEARS,
        "min_valid_years": MIN_VALID_YEARS,
        "cloud_cover_threshold": CLOUD_COVER_THRESHOLD,
        "sentinel2_collection": SENTINEL2_COLLECTION,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return f"{PIPELINE_CACHE_SCHEMA_VERSION}:{digest}"


def run_pipeline(cfg: PipelineConfig):
    global TARGET_GLACIER_COUNT
    TARGET_GLACIER_COUNT = cfg.target_glaciers

    output_root = Path(cfg.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    state = StateDB(output_root / "dataset_state.db")
    global_cache = GlobalGlacierCacheDB(cfg.global_cache_path)
    global_counts = global_cache.summary_counts()
    LOG.info("Loaded global glacier cache")
    LOG.info("Accepted glaciers cached: %d", global_counts["accepted"])
    LOG.info("Rejected glaciers cached: %d", global_counts["rejected"])
    pipeline_version = cfg.pipeline_version or build_pipeline_version(cfg)
    LOG.info("Pipeline version: %s", pipeline_version)
    cache_path = output_root / "stac_valid_year_cache.json"
    split_manifest_path = output_root / "split_manifest.csv"
    region_stats_path = output_root / "region_statistics.csv"

    accepted_df = None
    region_stats = None
    split_manifest = None

    

    try:
        # -------------------- Phase 1 --------------------
        if 1 in cfg.phases:
            if cfg.glims_path is None:
                raise ValueError("--glims-zip / --glims-path is required for Phase 1")
            gdf = load_glims_inventory(cfg.glims_path)
            client = get_stac_client()
            accepted_df, region_stats = phase1_filter_glaciers(
                gdf,
                client,
                cache_path,
                state,
                global_cache,
                pipeline_version,
                cfg.year_range,
            )
            accepted_df.to_csv(output_root / "phase1_accepted_glaciers.csv", index=False)
            region_stats.to_csv(region_stats_path, index=False)

            for _, row in accepted_df.iterrows():
                state.upsert_glacier(row["glacier_id"], row["region"], row["db_area"])
        else:
            accepted_path = output_root / "phase1_accepted_glaciers.csv"
            if accepted_path.exists():
                accepted_df = pd.read_csv(accepted_path)
                accepted_df["valid_years"] = accepted_df["valid_years"].apply(
                    lambda s: json.loads(s) if isinstance(s, str) else s
                )
            if region_stats_path.exists():
                region_stats = pd.read_csv(region_stats_path)

        # -------------------- Phase 2 --------------------
        if 2 in cfg.phases:
            if accepted_df is None or region_stats is None:
                raise RuntimeError("Phase 2 requires Phase 1 outputs; run Phase 1 first")
            split_manifest = phase2_region_split(accepted_df, region_stats, split_manifest_path)
            for _, row in split_manifest.iterrows():
                state.upsert_glacier(row["glacier_id"], row["region"], row["db_area"], split=row["split"])
        else:
            if split_manifest_path.exists():
                split_manifest = pd.read_csv(split_manifest_path)

        # -------------------- Phases 3-7 (per-glacier pipeline) --------------------
        if any(p in cfg.phases for p in (3, 4, 5, 6, 7)):
            if split_manifest is None or accepted_df is None:
                raise RuntimeError("Phases 3-7 require split_manifest + accepted glacier list")
            if gpd is None:
                raise RuntimeError("geopandas required for geometry lookups in Phases 3-7")

            gdf = load_glims_inventory(cfg.glims_path) if cfg.glims_path else None
            client = get_stac_client()

            merged = split_manifest.merge(
                accepted_df[["glacier_id", "valid_years"]], on="glacier_id", how="left"
            )

            glacier_geom_lookup = {}
            if gdf is not None:
                glacier_geom_lookup = dict(zip(gdf["glacier_id"], gdf["geometry"]))

            n_processed = 0
            for _, row in merged.iterrows():
                if cfg.max_glaciers_to_process and n_processed >= cfg.max_glaciers_to_process:
                    LOG.info("Reached debug limit of %d glaciers; stopping.", cfg.max_glaciers_to_process)
                    break

                glacier_id = row["glacier_id"]
                region = row["region"]
                split = row["split"]
                db_area = row["db_area"]
                valid_years = row.get("valid_years")
                if isinstance(valid_years, str):
                    valid_years = json.loads(valid_years)
                if not valid_years or len(valid_years) < MIN_VALID_YEARS:
                    LOG.warning("Glacier %s lacks sufficient valid_years metadata; skipping", glacier_id)
                    global_cache.mark_rejected(
                        glacier_id,
                        failed_phase=3,
                        failure_reason="Insufficient valid years",
                        region=region,
                        db_area=db_area,
                    )
                    continue

                current_status = state.get_glacier_status(glacier_id)
                if current_status == "COMPLETE":
                    LOG.debug("Glacier %s already COMPLETE; skipping", glacier_id)
                    global_cache.mark_accepted(
                        glacier_id,
                        region=region,
                        db_area=db_area,
                        valid_years=valid_years,
                        seasonal_observation_count=REQUIRED_TIMESTAMPS,
                    )
                    n_processed += 1
                    continue

                geom = glacier_geom_lookup.get(glacier_id)
                if geom is None:
                    LOG.warning("No geometry found for glacier %s; skipping", glacier_id)
                    state.set_glacier_status(glacier_id, "FAILED")
                    global_cache.mark_rejected(
                        glacier_id,
                        failed_phase=3,
                        failure_reason="Unknown",
                        region=region,
                        db_area=db_area,
                        valid_years=valid_years,
                    )
                    continue

                LOG.info("Processing glacier %s (region=%s, split=%s)", glacier_id, region, split)
                state.set_glacier_status(glacier_id, "DOWNLOADING")

                try:
                    schedule = phase3_select_timestamps(client, {"geometry": geom}, valid_years)

                    chosen_years = sorted({yr for (yr, _season) in schedule.keys()})
                    n_complete = 0
                    for (year, season), info in schedule.items():
                        ranked = info["ranked_candidates"]
                        # ensure we have full Item objects available for download
                        ok = process_timestamp(
                            state, glacier_id, geom, region, split, year, season, ranked, output_root
                        )
                        if ok:
                            n_complete += 1

                    if n_complete >= REQUIRED_TIMESTAMPS:
                        write_glacier_metadata(output_root, split, region, glacier_id, db_area, chosen_years)
                        state.set_glacier_status(glacier_id, "COMPLETE")
                        global_cache.mark_accepted(
                            glacier_id,
                            region=region,
                            db_area=db_area,
                            valid_years=valid_years,
                            seasonal_observation_count=n_complete,
                        )
                        LOG.info("Glacier %s COMPLETE (%d/%d timestamps)", glacier_id, n_complete, REQUIRED_TIMESTAMPS)
                    else:
                        # Per spec: incomplete glaciers are not saved as partial data.
                        glacier_dir = output_root / split / region / glacier_id
                        if glacier_dir.exists():
                            shutil.rmtree(glacier_dir, ignore_errors=True)
                        state.set_glacier_status(glacier_id, "FAILED")
                        global_cache.mark_rejected(
                            glacier_id,
                            failed_phase=4,
                            failure_reason="Incomplete seasonal coverage",
                            region=region,
                            db_area=db_area,
                            valid_years=valid_years,
                            seasonal_observation_count=n_complete,
                        )
                        LOG.warning("Glacier %s INCOMPLETE (%d/%d) -> discarded", glacier_id, n_complete, REQUIRED_TIMESTAMPS)

                except Exception as exc:  # noqa: BLE001
                    LOG.error("Unhandled error processing glacier %s: %s\n%s",
                              glacier_id, exc, traceback.format_exc())
                    state.set_glacier_status(glacier_id, "FAILED")
                    global_cache.mark_rejected(
                        glacier_id,
                        failed_phase=4,
                        failure_reason="Unknown",
                        region=region,
                        db_area=db_area,
                        valid_years=valid_years,
                    )

                n_processed += 1

            if 6 in cfg.phases:
                build_global_manifest(state, output_root)

        # -------------------- Phase 8 --------------------
        if 8 in cfg.phases:
            if split_manifest is None:
                split_manifest = pd.read_csv(split_manifest_path)
            if region_stats is None:
                region_stats = pd.read_csv(region_stats_path)
            phase8_quality_reports(state, region_stats, split_manifest, output_root)

    finally:
        LOG.info("Global cache updated")
        LOG.info("New accepted: %d", global_cache.new_accepted)
        LOG.info("New rejected: %d", global_cache.new_rejected)
        LOG.info("Skipped from cache: %d", global_cache.skipped_accepted + global_cache.skipped_rejected)
        global_cache.close()
        state.close()


# ==============================================================================
# CLI
# ==============================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Glacier SSL dataset creation pipeline")
    p.add_argument("--glims-zip", "--glims-path", dest="glims_path", type=str, default=None,
                    help="Path to GLIMS inventory file (shapefile/geopackage/geojson, may be zipped)")
    p.add_argument("--output-dir", type=str, required=True, help="Root output directory")
    p.add_argument("--target-glaciers", type=int, default=TARGET_GLACIER_COUNT)
    p.add_argument("--year-start", type=int, default=2015)
    p.add_argument("--year-end", type=int, default=2024)
    p.add_argument("--phases", type=str, default="1,2,3,4,5,6,7,8",
                    help="Comma-separated list of phases to run, e.g. '1,2'")
    p.add_argument("--max-glaciers", type=int, default=None,
                    help="Debug: cap the number of glaciers processed in phases 3-7")
    p.add_argument("--pipeline-version", type=str, default=None,
                    help="Override the Phase 1 evaluation cache version. Defaults to a hash of Phase 1 config.")
    p.add_argument("--global-cache-db", type=str, default=str(GLOBAL_CACHE_PATH),
                    help="Path to the persistent global glacier cache database.")
    p.add_argument("--resume", action="store_true",
                    help="No-op flag for clarity; the pipeline always resumes automatically "
                         "from dataset_state.db when re-run with the same --output-dir.")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    output_dir = Path(args.output_dir)
    global LOG
    LOG = setup_logging(output_dir)

    LOG.info("=" * 80)
    LOG.info("Glacier SSL dataset pipeline starting")
    LOG.info("Output dir: %s", output_dir)
    LOG.info("Phases requested: %s", args.phases)
    LOG.info("=" * 80)

    cfg = PipelineConfig(
        output_dir=output_dir,
        glims_path=Path(args.glims_path) if args.glims_path else None,
        target_glaciers=args.target_glaciers,
        year_range=(args.year_start, args.year_end),
        phases=tuple(int(x) for x in args.phases.split(",")),
        max_glaciers_to_process=args.max_glaciers,
        pipeline_version=args.pipeline_version,
        global_cache_path=Path(args.global_cache_db),
    )

    try:
        run_pipeline(cfg)
    except Exception as exc:  # noqa: BLE001
        LOG.critical("Pipeline failed with unhandled exception: %s\n%s", exc, traceback.format_exc())
        sys.exit(1)

    LOG.info("Pipeline run finished.")


if __name__ == "__main__":
    main()
