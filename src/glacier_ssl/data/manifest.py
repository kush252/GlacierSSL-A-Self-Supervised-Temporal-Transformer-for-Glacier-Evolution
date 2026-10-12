"""Phase 2 — Data manifest and sequence construction.

Turns the raw archive (SQLite catalog + on-disk GeoTIFFs) into per-glacier chronological
observation lists with full metadata, and records the glacier-identity split.

Design decisions (per build plan Phase 2 + Cross-cutting rules):
* One manifest row per *observation* (timestamp): glacier id, observation id, acquisition
  date, per-band file paths, quality metadata, spatial reference.
* Each glacier's COMPLETE valid observations assembled in chronological order by actual
  acquisition date. No fixed 1/4/9-year sequences are pre-baked — windows are sampled in
  Phase 4.
* Missing observations / irregular gaps are preserved as real gaps (we never synthesize
  timestamps, and the timeline is dated, not index-based).
* Split is by glacier identity, taken from the DB's region-based assignment, so later
  pair/interval sampling cannot leak across splits.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, asdict, field
from datetime import date, datetime
from pathlib import Path
from typing import Optional

from ..config import Config
from ..utils.logging_utils import get_logger

_SEASON_ORDER = {"DJF": 0, "MAM": 1, "JJA": 2, "SON": 3}


def _parse_date(s: Optional[str]) -> Optional[date]:
    if not s:
        return None
    for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(s[: len(fmt) + 4], fmt).date()
        except (ValueError, TypeError):
            continue
    try:
        return datetime.fromisoformat(s).date()
    except (ValueError, TypeError):
        return None


@dataclass
class Observation:
    glims_id: str
    observation_id: int          # DB timestamp id (stable original identifier)
    year: int
    season: str
    acquisition_date: str        # ISO date string (temporal truth)
    band_paths: dict[str, str]   # band_name -> path (relative to project root)
    scl_path: Optional[str]
    cloud_pct: Optional[float]
    crs: Optional[str] = None    # spatial reference, filled lazily if needed

    def date_obj(self) -> date:
        d = _parse_date(self.acquisition_date)
        if d is None:
            # Fallback to a representative date for the seasonal composite.
            month = {"DJF": 1, "MAM": 4, "JJA": 7, "SON": 10}[self.season]
            return date(self.year, month, 15)
        return d


@dataclass
class GlacierSequence:
    glims_id: str
    region: str
    split: str
    area_km2: Optional[float]
    centroid: tuple[Optional[float], Optional[float]]
    observations: list[Observation] = field(default_factory=list)

    def sort_chronologically(self) -> None:
        self.observations.sort(key=lambda o: (o.date_obj(),
                                              _SEASON_ORDER.get(o.season, 9)))

    def dates(self) -> list[date]:
        return [o.date_obj() for o in self.observations]


class ManifestBuilder:
    """Builds and serializes the dataset manifest from the SQLite catalog + disk."""

    def __init__(self, config: Config, project_root: Optional[Path] = None):
        self.cfg = config
        self.log = get_logger("manifest")
        # The project root is the parent of src/; DB file paths are relative to it.
        self.project_root = (project_root or Path(__file__).resolve().parents[3])
        self.bands = list(config.data.band_order)
        self.scl_band = config.data.scl_band

    def _resolve(self, rel: str) -> Path:
        return self.project_root / rel.replace("\\", "/")

    def build(self, require_on_disk: bool = True,
              min_observations: int = 2) -> dict[str, GlacierSequence]:
        db_path = self._resolve(str(self.cfg.data.sqlite_path))
        self.log.info("Reading catalog: %s", db_path)
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row

        glaciers: dict[str, GlacierSequence] = {}
        grows = conn.execute(
            "SELECT glims_id, region, split, status, area_km2, centroid_lon, centroid_lat "
            "FROM glaciers WHERE status='COMPLETE' AND split IS NOT NULL"
        ).fetchall()
        for g in grows:
            glaciers[g["glims_id"]] = GlacierSequence(
                glims_id=g["glims_id"], region=g["region"], split=g["split"],
                area_km2=g["area_km2"],
                centroid=(g["centroid_lon"], g["centroid_lat"]),
            )
        self.log.info("COMPLETE glaciers with split: %d", len(glaciers))

        # Pull all COMPLETE timestamps for those glaciers with their bands.
        ts_rows = conn.execute(
            "SELECT id, glims_id, year, season, acquisition_date, cloud_pct "
            "FROM timestamps WHERE validation_status='COMPLETE'"
        ).fetchall()

        kept, dropped_missing_band, dropped_disk = 0, 0, 0
        for ts in ts_rows:
            gid = ts["glims_id"]
            if gid not in glaciers:
                continue
            brows = conn.execute(
                "SELECT band_name, file_path, validation_status FROM bands "
                "WHERE timestamp_id=?", (ts["id"],)
            ).fetchall()
            by_name = {b["band_name"]: b for b in brows}
            # Require all six model bands present & validated.
            band_paths: dict[str, str] = {}
            ok = True
            for band in self.bands:
                b = by_name.get(band)
                if b is None or b["validation_status"] != "COMPLETE" or not b["file_path"]:
                    ok = False
                    break
                band_paths[band] = b["file_path"]
            if not ok:
                dropped_missing_band += 1
                continue
            scl = by_name.get(self.scl_band)
            scl_path = scl["file_path"] if scl and scl["file_path"] else None

            if require_on_disk:
                if not all(self._resolve(p).exists() for p in band_paths.values()):
                    dropped_disk += 1
                    continue

            obs = Observation(
                glims_id=gid, observation_id=ts["id"], year=ts["year"],
                season=ts["season"], acquisition_date=ts["acquisition_date"] or "",
                band_paths=band_paths, scl_path=scl_path, cloud_pct=ts["cloud_pct"],
            )
            glaciers[gid].observations.append(obs)
            kept += 1

        conn.close()

        # Chronological ordering + drop glaciers with too few observations.
        final: dict[str, GlacierSequence] = {}
        for gid, seq in glaciers.items():
            seq.sort_chronologically()
            if len(seq.observations) >= min_observations:
                final[gid] = seq
        self.log.info(
            "Observations kept=%d dropped(band)=%d dropped(disk)=%d | "
            "glaciers with >=%d obs: %d",
            kept, dropped_missing_band, dropped_disk, min_observations, len(final),
        )
        self._log_split_summary(final)
        return final

    def _log_split_summary(self, seqs: dict[str, GlacierSequence]) -> None:
        by_split: dict[str, int] = {}
        obs_by_split: dict[str, int] = {}
        for s in seqs.values():
            by_split[s.split] = by_split.get(s.split, 0) + 1
            obs_by_split[s.split] = obs_by_split.get(s.split, 0) + len(s.observations)
        for split in sorted(by_split):
            self.log.info("  split=%s glaciers=%d observations=%d",
                          split, by_split[split], obs_by_split[split])

    def save(self, seqs: dict[str, GlacierSequence], path: Optional[str] = None) -> Path:
        out = self._resolve(path or str(self.cfg.data.manifest_path))
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "band_order": self.bands,
            "scl_band": self.scl_band,
            "glaciers": {gid: _sequence_to_json(s) for gid, s in seqs.items()},
        }
        with open(out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, default=str)
        self.log.info("Wrote manifest: %s", out)
        return out


def _sequence_to_json(seq: GlacierSequence) -> dict:
    d = asdict(seq)
    d["centroid"] = list(seq.centroid)
    return d


def load_manifest(path: str | Path) -> dict[str, GlacierSequence]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    out: dict[str, GlacierSequence] = {}
    for gid, s in payload["glaciers"].items():
        obs = [Observation(**o) for o in s["observations"]]
        out[gid] = GlacierSequence(
            glims_id=s["glims_id"], region=s["region"], split=s["split"],
            area_km2=s.get("area_km2"),
            centroid=tuple(s.get("centroid", (None, None))),
            observations=obs,
        )
    return out


def split_glaciers(seqs: dict[str, GlacierSequence], split: str) -> dict[str, GlacierSequence]:
    """Filter by split, enforcing disjointness by glacier identity (no leakage)."""
    return {gid: s for gid, s in seqs.items() if s.split == split}
