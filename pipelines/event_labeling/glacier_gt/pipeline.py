"""Per-glacier ground-truth build: acquire -> measure -> select epochs -> lineage -> windows."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from rasterio.features import rasterize

from .acquire import Grid, acquire_glacier, to_crs
from .catalog import glims_analyses, glims_children, load_glacier
from .labels import detect_splits, label_window
from .measure import Measurement, build_regions, measure_scene, select_epochs


class GlacierGT:
    """Everything measured for one glacier; labels any window of its annual epochs."""

    def __init__(self, info, cfg, grid, dem, regions, measurements, epochs, masks):
        self.info, self.cfg, self.grid, self.dem, self.regions = info, cfg, grid, dem, regions
        self.measurements, self.epochs, self.masks = measurements, epochs, masks
        self.glims = glims_analyses(cfg, info.glims_id)
        self.children = glims_children(cfg, info.glims_id)
        ok = [e for e in epochs if e.status == "ok"]
        yrs = [e.year for e in ok]
        self.splits_ratio = detect_splits(yrs, [masks[y]["components"] for y in yrs],
                                          [masks[y]["areas"] for y in yrs], cfg)
        self.splits_ndsi = detect_splits(yrs, [masks[y]["components_ndsi"] for y in yrs],
                                         [masks[y]["areas_ndsi"] for y in yrs], cfg)

    @property
    def years(self):
        return [e.year for e in self.epochs]

    def label(self, y0: int, y1: int) -> dict:
        r = label_window(self.epochs, y0, y1, self.splits_ratio, self.splits_ndsi, self.cfg,
                         self.glims, self.children)
        r.update(glims_id=self.info.glims_id, region=self.info.region, split=self.info.split,
                 window_id=f"{self.info.glims_id}_{y0}_{y1}")
        return r

    def label_dates(self, start: str, end: str) -> dict:
        """Label for an SSL window given by acquisition dates (seasonal timestamps).

        Uses the annual end-of-season epochs whose dates fall inside [start, end]. Sub-annual
        windows get 'Insufficient': change is not resolvable below the annual epoch spacing.
        """
        inside = [e.year for e in self.epochs if start <= e.date <= end]
        if len(inside) < 2:
            return {"primary_label": "Insufficient", "auto_tier": "-", "reason": "fewer than 2 annual epochs",
                    "start_date": start, "end_date": end, "glims_id": self.info.glims_id}
        r = self.label(min(inside), max(inside))
        r.update(start_date=start, end_date=end)
        return r

    def glims_validation(self) -> list[dict]:
        """Measurement accuracy vs independent GLIMS outlines from the same year."""
        if self.glims is None or self.glims.empty:
            return []
        rows = []
        ok = {e.year: e for e in self.epochs if e.status == "ok"}
        df = self.glims.assign(yr=self.glims["src_date"].str[:4].astype(int))
        for (aid, sdate), g in df[df.yr.isin(ok)].groupby(["anlys_id", "src_date"]):
            geom = to_crs(g.geometry.union_all(), self.grid.epsg)
            ref = rasterize([(geom, 1)], out_shape=(self.grid.height, self.grid.width),
                            transform=self.grid.transform, dtype="uint8").astype(bool)
            e = ok[int(sdate[:4])]
            body = self.masks[e.year]["body"]
            inter, union = (ref & body).sum(), (ref | body).sum()
            rows.append({
                "glims_id": self.info.glims_id, "anlys_id": int(aid), "glims_date": sdate[:10],
                "s2_date": e.date, "glims_area_km2": round(float(g["area"].sum()), 4),
                "glims_raster_km2": round(ref.sum() * self.grid.px**2 / 1e6, 4),
                "s2_area_km2": round(e.area_km2, 4), "s2_sigma_km2": round(e.sigma_km2, 4),
                "iou": round(inter / union, 3) if union else None,
                "analysts": g["analysts"].iloc[0], "method": (g["proc_desc"].iloc[0] or "")[:120],
            })
        return rows

    def save(self, out_dir: Path):
        out_dir.mkdir(parents=True, exist_ok=True)
        rec = {
            "glims_id": self.info.glims_id, "region": self.info.region, "split": self.info.split,
            "grid": self.grid.to_json(),
            "epochs": [asdict(e) for e in self.epochs],
            "candidates": [asdict(m) for m in self.measurements],
            "splits_ratio": self.splits_ratio, "splits_ndsi": self.splits_ndsi,
            "glims_validation": self.glims_validation(),
        }
        (out_dir / "ground_truth.json").write_text(json.dumps(rec, indent=1, default=float), encoding="utf-8")
        np.savez_compressed(out_dir / "masks.npz", **{
            f"{y}_{k}": v for y, m in self.masks.items() for k, v in m.items() if isinstance(v, np.ndarray)})
        return rec


def build_glacier(glims_id: str, cfg, log=print) -> GlacierGT:
    info = load_glacier(cfg, glims_id)
    acq = acquire_glacier(info, cfg, log=log)
    grid: Grid = acq["grid"]
    outline_utm = to_crs(info.outline_wgs84, grid.epsg)
    regions = build_regions(outline_utm, grid, acq["dem"], cfg)
    measurements, all_masks = [], {}
    for c in acq["candidates"]:
        m, masks = measure_scene(c, regions, acq["dem"], grid, cfg)
        measurements.append(m)
        if masks is not None:
            all_masks[m.item_id] = masks
    epochs = select_epochs(measurements, cfg)
    masks = {e.year: all_masks[e.item_id] for e in epochs if e.item_id in all_masks}
    st = {}
    for e in epochs:
        st[e.status.split(":")[0]] = st.get(e.status.split(":")[0], 0) + 1
    log(f"  {glims_id}: epochs {st}")
    return GlacierGT(info, cfg, grid, acq["dem"], regions, measurements, epochs, masks)
