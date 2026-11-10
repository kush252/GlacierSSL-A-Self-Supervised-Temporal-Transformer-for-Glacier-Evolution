"""Glacier metadata: outline, split, dataset scenes (from the SSL catalog), GLIMS analyses."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from functools import lru_cache

from shapely import wkt as shapely_wkt


@dataclass
class GlacierInfo:
    glims_id: str
    region: str
    split: str
    area_km2: float
    outline_wgs84: object                      # shapely geometry, EPSG:4326
    dataset_scenes: list = field(default_factory=list)  # dicts: year, season, date, item_id


def load_manifest(cfg) -> dict:
    with open(cfg.paths.manifest, "r", encoding="utf-8") as f:
        m = json.load(f)
    g = m["glaciers"]
    return g if isinstance(g, dict) else {x["glims_id"]: x for x in g}


def load_glacier(cfg, glims_id: str) -> GlacierInfo:
    man = load_manifest(cfg).get(glims_id, {})
    con = sqlite3.connect(cfg.paths.sqlite)
    try:
        row = con.execute(
            "SELECT region, split, area_km2, geometry_wkt FROM glaciers WHERE glims_id=?", (glims_id,)
        ).fetchone()
        if row is None:
            raise KeyError(f"{glims_id} not in catalog")
        scenes = [
            {"year": y, "season": s, "date": d, "item_id": i}
            for (y, s, d, i) in con.execute(
                "SELECT year, season, acquisition_date, stac_item_id FROM timestamps "
                "WHERE glims_id=? AND stac_item_id IS NOT NULL ORDER BY acquisition_date",
                (glims_id,),
            )
        ]
    finally:
        con.close()
    return GlacierInfo(
        glims_id=glims_id,
        region=man.get("region", row[0]),
        split=man.get("split", row[1]),
        area_km2=row[2],
        outline_wgs84=shapely_wkt.loads(row[3]),
        dataset_scenes=scenes,
    )


@lru_cache(maxsize=1)
def _glims_table(path: str):
    import geopandas as gpd

    df = gpd.read_file(path)
    return df[df["line_type"] == "glac_bound"]


def glims_analyses(cfg, glims_id: str):
    """All GLIMS outlines (different analysts / dates) of this glacier, or None if no cache."""
    if not cfg.paths.glims_cache.exists():
        return None
    df = _glims_table(str(cfg.paths.glims_cache))
    return df[df["glac_id"] == glims_id].sort_values("src_date")


def glims_children(cfg, glims_id: str):
    """GLIMS outlines whose parent_id is this glacier (possible fragmentation products)."""
    path = cfg.paths.glims_cache.with_name("glims_children.gpkg")
    if not path.exists():
        return None
    df = _glims_table(str(path))
    return df[df["parent_id"] == glims_id]
