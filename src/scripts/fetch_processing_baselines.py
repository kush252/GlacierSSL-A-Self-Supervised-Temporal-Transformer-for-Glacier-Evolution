"""Record each observation's Sentinel-2 processing baseline in the manifest.

Processing baseline >= 04.00 (ESA, from late 2021 / 25 Jan 2022, plus later reprocessing)
stores L2A reflectance with a +1000 DN offset (BOA_ADD_OFFSET = -1000). Mixing harmonised
and offset scenes in one time series puts an artificial brightness step into every
glacier's record at the switch-over -- about +1 standard deviation in SWIR on this dataset
-- which any temporal objective can read as "which side of 2022 is this image".

The acquisition date is only a proxy (late-December-2021 scenes already carry 04.00), so
the authoritative ``s2:processing_baseline`` is fetched from the Planetary Computer STAC
API, using the item id the download pipeline stored for every timestamp. Preprocessing
then removes the offset per scene (data.harmonize_baseline).

    python src/scripts/fetch_processing_baselines.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glacier_ssl.config import load_config
from glacier_ssl.utils.logging_utils import get_logger

SEARCH = "https://planetarycomputer.microsoft.com/api/stac/v1/search"
BATCH = 100


def fetch(ids: list[str], retries: int = 4) -> dict[str, str]:
    body = json.dumps({
        "collections": ["sentinel-2-l2a"], "ids": ids, "limit": len(ids),
        "fields": {"include": ["id", "properties.s2:processing_baseline"],
                   "exclude": ["assets", "links", "geometry", "bbox"]},
    }).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(SEARCH, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=60) as f:
                feats = json.load(f).get("features", [])
            return {ft["id"]: ft["properties"].get("s2:processing_baseline") for ft in feats}
        except Exception:                                   # noqa: BLE001
            time.sleep(2 ** attempt)
    return {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    log = get_logger("baseline")

    cfg = load_config(args.config)
    root = Path(__file__).resolve().parents[2]
    manifest_path = root / str(cfg.data.manifest_path)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    con = sqlite3.connect(str(root / str(cfg.data.sqlite_path)))
    item_of = dict(con.execute("SELECT id, stac_item_id FROM timestamps "
                               "WHERE stac_item_id IS NOT NULL").fetchall())
    obs = [o for g in payload["glaciers"].values() for o in g["observations"]]
    wanted = sorted({item_of[o["observation_id"]] for o in obs
                     if o["observation_id"] in item_of})
    log.info("%d observations, %d distinct STAC items to query", len(obs), len(wanted))

    batches = [wanted[i:i + BATCH] for i in range(0, len(wanted), BATCH)]
    found: dict[str, str] = {}
    with ThreadPoolExecutor(args.workers) as ex:
        for k, res in enumerate(ex.map(fetch, batches)):
            found.update(res)
            if (k + 1) % 50 == 0:
                log.info("  %d/%d batches", k + 1, len(batches))

    n_set = 0
    for o in obs:
        b = found.get(item_of.get(o["observation_id"], ""))
        o["processing_baseline"] = b
        n_set += b is not None
    missing = len(obs) - n_set
    backup = manifest_path.with_suffix(".json.pre_baseline.bak")
    if not backup.exists():
        shutil.copy2(manifest_path, backup)
    manifest_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    counts = Counter(o["processing_baseline"] for o in obs)
    log.info("baseline set for %d/%d observations (%d missing -> date fallback)",
             n_set, len(obs), missing)
    for b, n in sorted(counts.items(), key=lambda kv: str(kv[0])):
        log.info("  baseline %s: %d", b, n)
    log.info("manifest updated: %s (backup: %s)", manifest_path, backup.name)


if __name__ == "__main__":
    main()
