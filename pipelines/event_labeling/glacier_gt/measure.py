"""Per-epoch measurements: the ground-truth layer every label is derived from.

For each candidate scene: reflectance -> two independent ice masks -> QC -> glacier ice body
(inside the outline + connected ice beyond it) -> area with uncertainty, significant
components, terminus metrics. Then one epoch per year is selected (least seasonal snow among
QC-passing scenes, the standard inventory practice) and flagged if still snow-affected.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rasterio.features import rasterize
from scipy import ndimage
from skimage.measure import perimeter as sk_perimeter
from skimage.morphology import disk

EIGHT = np.ones((3, 3), bool)


# ----------------------------------------------------------------------------- regions
@dataclass
class Regions:
    outline: np.ndarray      # GLIMS outline (reference extent)
    R: np.ndarray            # analysis region: outline dilated by digitising tolerance
    ADV: np.ndarray          # R plus advance buffer (ice here must connect to R-ice)
    ENV: np.ndarray          # QC envelope
    RING: np.ndarray         # off-glacier ring for seasonal-snow detection
    TZ: np.ndarray           # terminus zone: lowest part of the outline by elevation


def _dilate(mask, metres, px):
    r = int(round(metres / px))
    return ndimage.binary_dilation(mask, structure=disk(r)) if r > 0 else mask.copy()


def build_regions(outline_utm, grid, dem, cfg) -> Regions:
    px = grid.px
    outline = rasterize([(outline_utm, 1)], out_shape=(grid.height, grid.width),
                        transform=grid.transform, all_touched=False, dtype="uint8").astype(bool)
    R = _dilate(outline, cfg.geometry.region_dilate_m, px)
    ADV = _dilate(outline, cfg.geometry.advance_dilate_m, px)
    zo = dem[outline]
    if np.isfinite(zo).any():
        low = dem <= np.nanquantile(zo, cfg.geometry.advance_zone_quantile)
        ADV = R | (ADV & low)
    else:
        ADV = R.copy()
    ENV = _dilate(outline, cfg.qc.envelope_m, px)
    RING = _dilate(outline, cfg.qc.ring_outer_m, px) & ~_dilate(outline, cfg.qc.ring_inner_m, px)
    z = dem[outline]
    if np.isfinite(z).any():
        thr = np.nanquantile(z, cfg.geometry.terminus_zone_quantile)
        TZ = outline & (dem <= thr)
    else:
        TZ = np.zeros_like(outline)
    return Regions(outline, R, ADV, ENV, RING, TZ)


# ----------------------------------------------------------------------------- spectra
def load_reflectance(path):
    d = np.load(path)
    baseline = float(str(d["baseline"]) or 0)
    # Processing baseline >= 04.00 (Jan 2022 onward, and ESA reprocessing) adds a +1000 DN offset.
    off = 1000.0 if baseline >= 4.0 else 0.0
    nodata = np.zeros(d["B02"].shape, bool)
    refl = {}
    for b in ("B02", "B03", "B04", "B08", "B11"):
        a = d[b].astype(np.float32)
        nodata |= a == 0
        refl[b] = (a - off) / 10000.0
    return refl, d["SCL"], nodata, str(d["datetime"]), baseline


def ice_masks(refl, cfg):
    m = cfg.mask
    red, swir, blue, green, nir = refl["B04"], refl["B11"], refl["B02"], refl["B03"], refl["B08"]
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = red / np.maximum(swir, 1e-4)
        ndsi = (green - swir) / np.maximum(green + swir, 1e-4)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndwi = (green - nir) / np.maximum(green + nir, 1e-4)
    water = (ndwi > m.water_ndwi_min) & (nir < m.water_nir_max)
    ice_ratio = (ratio > m.ratio_red_swir) & (blue > m.blue_min) & ~water
    ice_ndsi = (ndsi > m.ndsi_min) & (nir > m.ndsi_nir_min)
    return ice_ratio, ice_ndsi


# ----------------------------------------------------------------------------- ice body
def ice_body(ice, regions: Regions):
    """Ice inside R, plus ice in the advance buffer that is connected to it."""
    in_adv = ice & regions.ADV
    lab, _ = ndimage.label(in_adv, structure=EIGHT)
    keep = np.unique(lab[in_adv & regions.R])
    keep = keep[keep > 0]
    return np.isin(lab, keep) & in_adv


def components(body, px, cfg):
    """Significant components (>= MMU) of the ice body; narrow gaps closed first."""
    k = cfg.geometry.closing_px
    closed = (ndimage.binary_closing(body, structure=disk(k)) | body) if k > 0 else body
    lab, _ = ndimage.label(closed, structure=EIGHT)
    lab = np.where(body, lab, 0)          # areas counted on real ice pixels only
    mmu_px = cfg.geometry.mmu_km2 * 1e6 / px**2
    out = np.zeros_like(lab, dtype=np.int32)
    areas = []
    nid = 0
    for i, cnt in enumerate(np.bincount(lab.ravel())[1:], start=1):
        if cnt >= mmu_px:
            nid += 1
            out[lab == i] = nid
            areas.append(cnt * px**2 / 1e6)
    return out, areas


def area_sigma(body, px, unknown_area_km2, area_km2, cfg):
    perim_m = sk_perimeter(body) * px
    sigma_buf = perim_m * cfg.uncertainty.buffer_px * px / 1e6
    sigma = float(np.hypot(sigma_buf, 0.5 * unknown_area_km2))
    return max(sigma, cfg.uncertainty.rel_floor * area_km2), perim_m / 1000.0


@dataclass
class Measurement:
    item_id: str
    date: str
    year: int
    source: str
    baseline: float
    path: str = ""
    nodata_frac: float = np.nan
    unknown_frac: float = np.nan
    shadow_frac: float = np.nan
    ring_snow_frac: float = np.nan
    area_km2: float = np.nan
    sigma_km2: float = np.nan
    perimeter_km: float = np.nan
    area_ndsi_km2: float = np.nan
    sigma_ndsi_km2: float = np.nan
    area_scl_km2: float = np.nan
    beyond_outline_km2: float = np.nan
    outline_coverage: float = np.nan     # fraction of GLIMS outline covered by mapped ice
    mask_iou: float = np.nan             # ratio-mask vs NDSI-mask agreement on the ice body
    n_components: int = 0
    component_areas: list = field(default_factory=list)
    z_terminus_m: float = np.nan
    tz_ice_frac: float = np.nan
    qc_pass: bool = False
    status: str = "unevaluated"          # ok | seasonal_snow | cloud | nodata | fetch_error | not_selected
    selected: bool = False


def measure_scene(cand, regions: Regions, dem, grid, cfg):
    """Returns (Measurement, masks dict) for one candidate scene."""
    px = grid.px
    m = Measurement(item_id=cand["item_id"], date=cand["date"], year=int(cand["year"]),
                    source=cand["source"], baseline=0.0, path=cand.get("path") or "")
    if not cand.get("path"):
        m.status = "fetch_error"
        return m, None
    refl, scl, nodata, dt, baseline = load_reflectance(cand["path"])
    m.baseline, m.date = baseline, dt[:10]
    ice_r, ice_n = ice_masks(refl, cfg)
    ice_r &= ~nodata
    ice_n &= ~nodata
    unknown = np.isin(scl, cfg.mask.scl_unknown_classes) & ~ice_r & ~nodata
    shadow = (refl["B08"] < 0.06) & (refl["B04"] < 0.06) & ~nodata

    m.nodata_frac = float(nodata[regions.ENV].mean())
    m.unknown_frac = float(unknown[regions.R].mean())
    ring_valid = regions.RING & ~nodata & ~unknown
    m.ring_snow_frac = float(ice_r[ring_valid].mean()) if ring_valid.sum() > 50 else np.nan
    if m.nodata_frac > cfg.qc.max_nodata_frac:
        m.status = "nodata"
        return m, None
    if m.unknown_frac > cfg.qc.max_unknown_frac:
        m.status = "cloud"
        return m, None
    m.shadow_frac = float(shadow[regions.R].mean())
    if m.shadow_frac > cfg.qc.max_shadow_frac:
        m.status = "shadow"
        return m, None

    pxa = px**2 / 1e6
    body = ice_body(ice_r, regions)
    body_n = ice_body(ice_n, regions)
    m.area_km2 = float(body.sum() * pxa)
    unknown_km2 = float((unknown & regions.R).sum() * pxa)
    m.sigma_km2, m.perimeter_km = area_sigma(body, px, unknown_km2, m.area_km2, cfg)
    m.area_ndsi_km2 = float(body_n.sum() * pxa)
    m.sigma_ndsi_km2, _ = area_sigma(body_n, px, unknown_km2, m.area_ndsi_km2, cfg)
    m.area_scl_km2 = float(((scl == 11) & regions.R).sum() * pxa)
    m.beyond_outline_km2 = float((body & ~regions.R).sum() * pxa)
    m.outline_coverage = float(body[regions.outline].mean()) if regions.outline.any() else np.nan
    union = (body | body_n).sum()
    m.mask_iou = float((body & body_n).sum() / union) if union else 1.0
    comp, areas = components(body, px, cfg)
    m.n_components, m.component_areas = len(areas), [round(a, 5) for a in areas]
    main = comp == (int(np.argmax(areas)) + 1) if areas else body   # terminus of the main body
    z = dem[main & np.isfinite(dem)]
    m.z_terminus_m = float(np.quantile(z, cfg.geometry.terminus_quantile)) if z.size > 20 else np.nan
    m.tz_ice_frac = float(body[regions.TZ].mean()) if regions.TZ.any() else np.nan
    m.qc_pass = True
    m.status = "qc_pass"
    comp_n, areas_n = components(body_n, px, cfg)
    masks = {"body": body, "body_ndsi": body_n, "components": comp, "areas": areas,
             "components_ndsi": comp_n, "areas_ndsi": areas_n, "unknown": unknown & regions.ENV}
    return m, masks


def select_epochs(measurements: list[Measurement], cfg) -> list[Measurement]:
    """One epoch per year: the QC-passing scene with the least off-glacier snow.

    Seasonal-snow baseline = the cleanest ring fraction this glacier ever shows (neighbouring
    permanent ice lives in that baseline), so the test is self-calibrating per glacier.
    """
    passed = [m for m in measurements if m.qc_pass]
    best_per_year = {}
    for m in passed:
        if np.isfinite(m.ring_snow_frac):
            best_per_year[m.year] = min(best_per_year.get(m.year, 9.0), m.ring_snow_frac)
    base = (float(np.quantile(list(best_per_year.values()), cfg.qc.seasonal_snow_baseline_quantile))
            if best_per_year else np.nan)
    years = sorted({m.year for m in measurements})
    epochs = []
    for y in years:
        pool = [m for m in passed if m.year == y]
        if not pool:
            # keep a placeholder so the figure/tables show *why* the year is missing
            reasons = sorted({m.status for m in measurements if m.year == y})
            epochs.append(Measurement(item_id="", date=f"{y}-09-01", year=y, source="none", baseline=0,
                                      status="no_valid_scene:" + "|".join(reasons)))
            continue

        def key(m):
            r = m.ring_snow_frac if np.isfinite(m.ring_snow_frac) else 9.0
            return (round(r, 3), m.area_km2, -int(m.date.replace("-", "")))

        best = min(pool, key=key)
        best.selected = True
        if np.isfinite(base) and np.isfinite(best.ring_snow_frac) and \
                best.ring_snow_frac > base + cfg.qc.seasonal_snow_margin:
            best.status = "seasonal_snow"
        else:
            best.status = "ok"
        epochs.append(best)
    for m in measurements:
        if m.qc_pass and not m.selected:
            m.status = "not_selected"
    return epochs
