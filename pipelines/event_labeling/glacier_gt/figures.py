"""Evidence panels: the ground truth each label was derived from, as a paper-ready figure.

Row 1  false-colour chips (SWIR1-NIR-Green) of every epoch in the window; GLIMS outline dashed,
       mapped ice body outlined, significant components in fixed categorical order, rejected
       epochs marked with the reason.
Row 2  area series (both masks, +-1 sigma, Theil-Sen fit) | significant components |
       terminus elevation | change map first->last valid epoch.
Each measure has its own axis (no dual axes).
"""
from __future__ import annotations

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from .labels import decimal_year
from .measure import load_reflectance

# reference palette (dataviz skill, light mode)
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
LOSS, GAIN, KEEP = "#eb6834", "#2a78d6", "#d9d8d3"     # diverging poles + neutral

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": MUTED, "ytick.color": MUTED,
    "text.color": INK, "axes.spines.top": False, "axes.spines.right": False, "font.size": 8,
    "axes.titlesize": 8.5, "axes.titleweight": "bold", "axes.grid": True, "grid.color": "#ecebe7",
    "grid.linewidth": 0.6, "lines.linewidth": 2,
})


def false_colour(path):
    refl, _, nodata, _, _ = load_reflectance(path)
    rgb = np.stack([refl["B11"], refl["B08"], refl["B03"]], -1)
    rgb = np.clip(rgb / 0.6, 0, 1) ** 0.8          # fixed stretch: comparable across epochs
    rgb[nodata] = 1.0
    return rgb


def _contour(ax, mask, color, lw=0.9, ls="-"):
    if mask is not None and mask.any() and not mask.all():
        ax.contour(mask.astype(float), levels=[0.5], colors=[color], linewidths=lw, linestyles=ls)


def _crop_box(regions, pad=15):
    ys, xs = np.where(regions.ADV)
    if ys.size == 0:
        return slice(None), slice(None)
    return (slice(max(0, ys.min() - pad), ys.max() + pad), slice(max(0, xs.min() - pad), xs.max() + pad))


def window_figure(glacier, epochs, masks, regions, rec, out_base, title_extra=""):
    y0, y1 = rec["start_year"], rec["end_year"]
    win = [e for e in epochs if y0 <= e.year <= y1]
    ncol_chips = min(len(win), 9)
    nrow_chips = math.ceil(len(win) / ncol_chips)
    fig = plt.figure(figsize=(max(11, 1.45 * ncol_chips), 2.0 * nrow_chips + 3.3), layout="constrained")
    gs = fig.add_gridspec(nrow_chips + 1, 1, height_ratios=[1.0] * nrow_chips + [1.55])
    sy, sx = _crop_box(regions)

    # ------------------------------------------------------------ chips
    for i, e in enumerate(win):
        row = i // ncol_chips
        sub = gs[row].subgridspec(1, ncol_chips, wspace=0.03)
        ax = fig.add_subplot(sub[i % ncol_chips])
        ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(True), s.set_color(MUTED), s.set_linewidth(0.6)
        m = masks.get(e.year)
        if e.item_id and getattr(e, "path", None):
            ax.imshow(false_colour(e.path)[sy, sx], interpolation="nearest")
            _contour(ax, regions.outline[sy, sx], "white", 0.8, "--")
            if m is not None and e.status == "ok":
                comp = m["components"][sy, sx]
                for c in range(1, min(comp.max(), 8) + 1):
                    _contour(ax, comp == c, CAT[(c - 1) % 8], 1.1)
        else:
            ax.set_facecolor("#f0efec")
        ok = e.status == "ok"
        label = f"{e.date}" if ok else f"{e.year}  ✗ {e.status.split(':')[0]}"
        ax.set_title(label, fontsize=7, fontweight="bold" if ok else "normal", color=INK if ok else MUTED)
        if ok:
            ax.text(0.02, 0.02, f"{e.area_km2:.3f} km²\n{e.n_components} comp.", transform=ax.transAxes,
                    fontsize=6.5, color=INK, va="bottom",
                    bbox=dict(boxstyle="round,pad=0.2", fc=SURFACE, ec="none", alpha=0.85))

    # ------------------------------------------------------------ plots
    bot = gs[-1].subgridspec(1, 4, width_ratios=[2.2, 1, 1, 1.25], wspace=0.08)
    ok_all = [e for e in epochs if e.status == "ok"]
    t_all = np.array([decimal_year(e.date) for e in ok_all])
    shade = dict(color="#ecebe7", zorder=0)

    ax = fig.add_subplot(bot[0])
    ax.axvspan(y0, y1 + 1, **shade)
    for key, sk, col, name in (("area_km2", "sigma_km2", CAT[0], "red/SWIR mask"),
                               ("area_ndsi_km2", "sigma_ndsi_km2", CAT[1], "NDSI mask")):
        a = np.array([getattr(e, key) for e in ok_all])
        s = np.array([getattr(e, sk) for e in ok_all])
        ax.errorbar(t_all, a, yerr=s, color=col, marker="o", ms=4.5, lw=1.6, capsize=2, elinewidth=1,
                    label=name, mec=SURFACE, mew=1)
    bad = [e for e in epochs if e.status != "ok" and np.isfinite(e.area_km2)]
    if bad:
        ax.scatter([decimal_year(e.date) for e in bad], [e.area_km2 for e in bad], marker="x", s=24,
                   color=MUTED, label="excluded epoch (QC)", zorder=3)
    ar = rec["evidence"].get("area_ratio")
    vw = [e for e in ok_all if y0 <= e.year <= y1]
    if ar and len(vw) >= 2:
        tw = np.array([decimal_year(e.date) for e in vw])
        aw = np.array([e.area_km2 for e in vw])
        mid = np.median(aw - ar["slope_per_yr"] * tw)
        ax.plot(tw, mid + ar["slope_per_yr"] * tw, color=INK2, lw=1.2, ls="--", label="Theil–Sen (window)")
    ax.set_title("Glacier ice area (km²), ±1σ")
    ax.legend(fontsize=6.5, frameon=False, loc="best")

    ax = fig.add_subplot(bot[1])
    ax.axvspan(y0, y1 + 1, **shade)
    ax.step(t_all, [e.n_components for e in ok_all], where="post", color=CAT[0])
    ax.plot(t_all, [e.n_components for e in ok_all], "o", color=CAT[0], ms=4.5, mec=SURFACE, mew=1)
    for s in rec["evidence"].get("splits_ratio", []):
        ax.axvline(s["year"] + 0.6, color=CAT[7] if s["confirmed"] else MUTED, lw=1, ls=":")
    ax.set_title("Components ≥ MMU")
    ax.yaxis.get_major_locator().set_params(integer=True)

    ax = fig.add_subplot(bot[2])
    ax.axvspan(y0, y1 + 1, **shade)
    z = [(t, e.z_terminus_m) for t, e in zip(t_all, ok_all) if np.isfinite(e.z_terminus_m)]
    if z:
        ax.plot(*zip(*z), "o-", color=CAT[0], ms=4.5, mec=SURFACE, mew=1)
    ax.set_title("Terminus elevation (m)")

    ax = fig.add_subplot(bot[3])
    ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    if len(vw) >= 2 and vw[0].year in masks and vw[-1].year in masks:
        a, b = masks[vw[0].year]["body"][sy, sx], masks[vw[-1].year]["body"][sy, sx]
        cm = np.zeros(a.shape, np.uint8)
        cm[a & b], cm[a & ~b], cm[~a & b] = 1, 2, 3
        ax.imshow(cm, cmap=ListedColormap([SURFACE, KEEP, LOSS, GAIN]), vmin=0, vmax=3, interpolation="nearest")
        _contour(ax, regions.outline[sy, sx], INK2, 0.7, "--")
        ax.legend(handles=[Patch(color=KEEP, label="ice both"), Patch(color=LOSS, label=f"lost by {vw[-1].year}"),
                           Patch(color=GAIN, label=f"new by {vw[-1].year}"),
                           Line2D([], [], color=INK2, ls="--", lw=0.8, label="GLIMS outline")],
                  fontsize=6, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 0.0), ncol=2)
        ax.set_title(f"Change {vw[0].year} → {vw[-1].year}")

    ev = rec["evidence"]
    def fmt(line, unit="km²"):
        if not line:
            return "n/a"
        p = f", MK p={line['mk_p']}" if line.get("mk_p") is not None else ""
        return f"Δ={line['delta']:+.3f} {unit} (SNR {line['snr']}{p}) → {['−', '0', '+'][line['direction'] + 1]}"
    head = (f"{glacier.region} · {glacier.glims_id} · {y0}–{y1} ({rec['n_valid']}/{rec['n_epochs']} valid)   "
            f"LABEL: {rec['primary_label']}   tier {rec['auto_tier']}   agree={'+'.join(rec['agree']) or '–'}"
            f"{'   against=' + '+'.join(rec['against']) if rec['against'] else ''}{title_extra}")
    sub = (f"area(ratio) {fmt(ev.get('area_ratio'))}   area(NDSI) {fmt(ev.get('area_ndsi'))}   "
           f"terminus {fmt(ev.get('terminus'), 'm')}   flags: {', '.join(rec['flags']) or 'none'}")
    fig.suptitle(head + "\n" + sub, fontsize=8, x=0.01, ha="left")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out_base}.png", dpi=170)
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)
    return f"{out_base}.png"


def class_gallery(examples, out_base):
    """One row per class: first / middle / last valid chip of the best example + change map.

    examples: list of dicts {label, glacier, rec, epochs, masks, regions}
    """
    if not examples:
        return None
    fig, axes = plt.subplots(len(examples), 4, figsize=(8.5, 2.2 * len(examples)), layout="constrained",
                             squeeze=False)
    for r, ex in enumerate(examples):
        rec, regions = ex["rec"], ex["regions"]
        sy, sx = _crop_box(regions)
        vw = [e for e in ex["epochs"] if e.status == "ok" and rec["start_year"] <= e.year <= rec["end_year"]]
        picks = [vw[0], vw[len(vw) // 2], vw[-1]]
        for c, e in enumerate(picks):
            ax = axes[r, c]
            ax.imshow(false_colour(e.path)[sy, sx], interpolation="nearest")
            _contour(ax, regions.outline[sy, sx], "white", 0.8, "--")
            comp = ex["masks"][e.year]["components"][sy, sx]
            for k in range(1, min(comp.max(), 8) + 1):
                _contour(ax, comp == k, CAT[(k - 1) % 8], 1.1)
            ax.set_title(f"{e.date} · {e.area_km2:.3f} km²", fontsize=7)
        a, b = ex["masks"][vw[0].year]["body"][sy, sx], ex["masks"][vw[-1].year]["body"][sy, sx]
        cm = np.zeros(a.shape, np.uint8)
        cm[a & b], cm[a & ~b], cm[~a & b] = 1, 2, 3
        axes[r, 3].imshow(cm, cmap=ListedColormap([SURFACE, KEEP, LOSS, GAIN]), vmin=0, vmax=3,
                          interpolation="nearest")
        axes[r, 3].set_title("lost / new / kept", fontsize=7)
        axes[r, 0].set_ylabel(f"{ex['label']}\n(tier {rec['auto_tier']})", fontsize=9, fontweight="bold", color=INK)
        for ax in axes[r]:
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
    fig.suptitle("What each event class looks like (Sentinel-2 SWIR1-NIR-Green; dashed = GLIMS outline)",
                 fontsize=9, x=0.01, ha="left")
    fig.savefig(f"{out_base}.png", dpi=170)
    fig.savefig(f"{out_base}.pdf")
    plt.close(fig)
    return f"{out_base}.png"
