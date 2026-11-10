"""Native-resolution, pixel-aligned Sentinel-2 L2A chips (and Copernicus DEM) per glacier.

Why not reuse the SSL tiles: those are masked to the static GLIMS outline (everything outside is
0, so advance and seasonal snow around the glacier are invisible) and resampled to 512x512 with
non-square pixels. Ground truth is measured here on a fixed 10 m UTM grid covering the outline
plus a buffer, read straight from the same Planetary Computer catalogue (same item ids where the
dataset scene is used), so every epoch is pixel-aligned and areas are exact.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import planetary_computer as pc
import pyproj
import pystac
import pystac_client
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine, from_origin
from rasterio.vrt import WarpedVRT
from shapely.ops import transform as shp_transform

PC_STAC = "https://planetarycomputer.microsoft.com/api/stac/v1"
S2_COLLECTION = "sentinel-2-l2a"
BANDS = ["B02", "B03", "B04", "B08", "B11", "SCL"]
GDAL_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    GDAL_HTTP_MAX_RETRY="4",
    GDAL_HTTP_RETRY_DELAY="2",
    VSI_CACHE="TRUE",
)


@dataclass
class Grid:
    epsg: int
    transform: Affine
    width: int
    height: int
    px: float

    def to_json(self):
        return {"epsg": self.epsg, "transform": list(self.transform)[:6], "width": self.width,
                "height": self.height, "px": self.px}

    @classmethod
    def from_json(cls, d):
        return cls(d["epsg"], Affine(*d["transform"]), d["width"], d["height"], d["px"])


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def to_crs(geom_wgs84, epsg: int):
    tr = pyproj.Transformer.from_crs(4326, epsg, always_xy=True).transform
    return shp_transform(tr, geom_wgs84)


def make_grid(outline_wgs84, px: float, buffer_m: float) -> tuple[Grid, object]:
    c = outline_wgs84.centroid
    epsg = utm_epsg(c.x, c.y)
    geom = to_crs(outline_wgs84, epsg)
    minx, miny, maxx, maxy = geom.buffer(buffer_m).bounds
    minx, miny = np.floor(minx / px) * px, np.floor(miny / px) * px
    maxx, maxy = np.ceil(maxx / px) * px, np.ceil(maxy / px) * px
    grid = Grid(epsg, from_origin(minx, maxy, px, px), int(round((maxx - minx) / px)),
                int(round((maxy - miny) / px)), px)
    return grid, geom


def _client():
    return pystac_client.Client.open(PC_STAC, modifier=pc.sign_inplace)


def search_candidates(outline_wgs84, year: int, cfg) -> list[dict]:
    """Lowest-cloud end-of-ablation scenes of one year intersecting the outline."""
    lo, hi = cfg.scenes.window
    for attempt in range(3):
        try:
            items = list(_client().search(
                collections=[S2_COLLECTION], intersects=outline_wgs84.__geo_interface__,
                datetime=f"{year}-{lo}/{year}-{hi}",
                query={"eo:cloud_cover": {"lt": cfg.scenes.max_tile_cloud_pct}},
            ).items())
            break
        except Exception:  # transient API errors
            if attempt == 2:
                raise
            time.sleep(3 * (attempt + 1))
    items.sort(key=lambda it: (it.properties.get("eo:cloud_cover", 100), it.id))
    out, seen_dates = [], set()
    for it in items:
        d = it.datetime.date().isoformat()
        if d in seen_dates:          # same overpass on two MGRS tiles: keep the clearer one
            continue
        seen_dates.add(d)
        out.append({"item_id": it.id, "date": d, "tile_cloud": it.properties.get("eo:cloud_cover"),
                    "source": "search"})
        if len(out) >= cfg.scenes.max_candidates_per_year:
            break
    return out


def _read_to_grid(href: str, grid: Grid, resampling: Resampling, dtype) -> np.ndarray:
    with rasterio.Env(**GDAL_ENV):
        with rasterio.open(href) as src:
            with WarpedVRT(src, crs=f"EPSG:{grid.epsg}", transform=grid.transform, width=grid.width,
                           height=grid.height, resampling=resampling, src_nodata=0, nodata=0) as vrt:
                return vrt.read(1).astype(dtype)


def fetch_scene(item_id: str, grid: Grid, cache_dir: Path) -> Path:
    """Read the 6 bands of one item onto the glacier grid; cached as compressed npz."""
    out = cache_dir / "scenes" / f"{item_id}.npz"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    last_err = None
    for attempt in range(3):
        try:
            item = pc.sign(pystac.Item.from_file(f"{PC_STAC}/collections/{S2_COLLECTION}/items/{item_id}"))
            arrays = {}
            for b in BANDS:
                rs = Resampling.nearest if b == "SCL" else Resampling.bilinear
                arrays[b] = _read_to_grid(item.assets[b].href, grid, rs, np.uint8 if b == "SCL" else np.uint16)
            np.savez_compressed(
                out, **arrays,
                baseline=np.array(item.properties.get("s2:processing_baseline", "00.00")),
                datetime=np.array(item.datetime.isoformat()),
            )
            return out
        except Exception as e:  # network hiccups, expired SAS token
            last_err = e
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"failed to fetch {item_id}: {last_err}")


def fetch_dem(outline_wgs84, grid: Grid, cache_dir: Path) -> np.ndarray:
    """Copernicus GLO-30 DEM resampled onto the glacier grid (float32, NaN = missing)."""
    out = cache_dir / "dem.npy"
    if out.exists():
        return np.load(out)
    items = list(_client().search(collections=["cop-dem-glo-30"],
                                  intersects=outline_wgs84.buffer(0.02).__geo_interface__).items())
    dem = np.full((grid.height, grid.width), np.nan, np.float32)
    for it in items:
        with rasterio.Env(**GDAL_ENV), rasterio.open(it.assets["data"].href) as src:
            with WarpedVRT(src, crs=f"EPSG:{grid.epsg}", transform=grid.transform, width=grid.width,
                           height=grid.height, resampling=Resampling.bilinear,
                           nodata=np.nan, dtype="float32") as vrt:
                a = vrt.read(1)
        dem = np.where(np.isnan(dem) & np.isfinite(a), a, dem)
    np.save(out, dem)
    return dem


def gather_candidates(info, cfg) -> list[dict]:
    """Searched end-of-season scenes + (optionally) the dataset's own JJA/SON scenes."""
    y0, y1 = cfg.scenes.years
    cands = []
    for y in range(y0, y1 + 1):
        cands += [dict(c, year=y) for c in search_candidates(info.outline_wgs84, y, cfg)]
    if cfg.scenes.include_dataset_scenes:
        have = {c["item_id"] for c in cands}
        lo, hi = cfg.scenes.window
        for s in info.dataset_scenes:
            if lo <= s["date"][5:10] <= hi and y0 <= int(s["date"][:4]) <= y1 and s["item_id"] not in have:
                cands.append({"item_id": s["item_id"], "date": s["date"], "year": int(s["date"][:4]),
                              "tile_cloud": None, "source": f"dataset_{s['season']}"})
    return sorted(cands, key=lambda c: c["date"])


def acquire_glacier(info, cfg, workers: int = 8, log=print) -> dict:
    """Grid + DEM + all candidate scenes for one glacier, cached under cache/<glims_id>/."""
    cache_dir = Path(cfg.paths.cache) / info.glims_id
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "acquisition.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        grid = Grid.from_json(meta["grid"])
    else:
        grid, _ = make_grid(info.outline_wgs84, cfg.grid.pixel_m, cfg.grid.buffer_m)
        meta = {"grid": grid.to_json(), "candidates": gather_candidates(info, cfg)}
        meta_path.write_text(json.dumps(meta, indent=1))
    log(f"  {info.glims_id}: grid {grid.width}x{grid.height} @ {grid.px} m, "
        f"{len(meta['candidates'])} candidate scenes")
    dem = fetch_dem(info.outline_wgs84, grid, cache_dir)

    def job(c):
        try:
            return c["item_id"], str(fetch_scene(c["item_id"], grid, cache_dir)), None
        except Exception as e:
            return c["item_id"], None, str(e)

    with ThreadPoolExecutor(workers) as ex:
        results = {i: (p, e) for i, p, e in ex.map(job, meta["candidates"])}
    for c in meta["candidates"]:
        c["path"], c["fetch_error"] = results[c["item_id"]]
    return {"grid": grid, "dem": dem, "candidates": meta["candidates"], "cache_dir": cache_dir}
