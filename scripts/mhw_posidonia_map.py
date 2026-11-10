"""
Recreation of the Copernicus/C3S "Where can Posidonia oceanica meadows be found?"
map style (European State of the Climate 2025, Fig. 20.1).

Real data:      Natural Earth coastlines (50m for the main map, 10m for the inset).
Synthetic data: a year of daily sea-surface temperature anomalies for 2025, run
                through the Hobday et al. (2018) marine heatwave rules, plus
                synthetic Posidonia locations and meadow polygons.

To use real data instead:
  * SST: C3S/ESA CCI SST v3 or NOAA OISST daily files -> build a 1991-2020 daily
    climatology and 90th-percentile threshold (11-day window), then feed the
    2025 anomalies and thresholds to max_heatwave_category().
  * Seagrass: UNEP-WCMC "Global Distribution of Seagrasses" (polygons + points),
    filtered to Posidonia oceanica.
"""
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Circle, Rectangle
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import box

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ne")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("mhw_posidonia_2025_synthetic.png")
rng = np.random.default_rng(20)

# ----------------------------------------------------------------------------
# Style
# ----------------------------------------------------------------------------
for f in Path.home().joinpath(".fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Barlow Condensed"

NAVY, GREY_TXT, COAST, DOT = "#2c3e5c", "#8a909c", "#3a3f4a", "#1d2557"
CATS = ["No heatwave", "Moderate", "Strong", "Severe", "Extreme"]
COLORS = ["#ffffff", "#fef3db", "#fbb77a", "#ed6242", "#b11f24"]

VIEW = (-10.5, 42.5, 30.0, 47.6)         # lon/lat window
ASPECT = 1 / np.cos(np.radians(38))      # equirectangular, true scale at 38°N

# ----------------------------------------------------------------------------
# 1. Mediterranean sea mask on a 0.1° grid
# ----------------------------------------------------------------------------
RES = 0.1
LON0, LON1, LAT0, LAT1 = -6.0, 36.5, 30.0, 46.0
W, H = int((LON1 - LON0) / RES), int((LAT1 - LAT0) / RES)
lons = LON0 + (np.arange(W) + 0.5) * RES
lats = LAT1 - (np.arange(H) + 0.5) * RES
LON, LAT = np.meshgrid(lons, lats)

land50 = gpd.read_file(DATA / "ne_50m_land.shp").clip(box(-30, 20, 60, 60))
land_r = rasterize(((g, 1) for g in land50.geometry), out_shape=(H, W),
                   transform=from_origin(LON0, LAT1, RES, RES), dtype="uint8").astype(bool)
sea = ~land_r
sea &= ~(LON < -5.6)                      # cut at Gibraltar: no Atlantic
sea &= ~((LON > 26.3) & (LAT > 40.3))     # cut at the Dardanelles: no Marmara / Black Sea
sea &= ~((LON < 0) & (LAT > 42))          # no Bay of Biscay
lab, _ = ndimage.label(sea)
med = lab == lab[np.argmin(abs(lats - 35)), np.argmin(abs(lons - 18))]   # drop inland lakes

# ----------------------------------------------------------------------------
# 2. Synthetic daily SST anomalies for 2025 + Hobday categories
# ----------------------------------------------------------------------------
def smooth_noise(sigma):
    n = ndimage.gaussian_filter(rng.standard_normal((H, W)), sigma)
    return n / n.std()


days = np.arange(365)
# Threshold above climatology (p90 - clim): larger in the shallow, variable north
delta = 1.0 + 0.35 * np.clip((LAT - 36) / 8, 0, 1) + 0.15 * smooth_noise(20)

# Spatial "hot spots" that peak at different times of the year
events = [  # lon, lat, size(deg), peak day, duration(days), amplitude (°C)
    (3.5, 39.5, 3.0, 175, 45, 3.6),   # Balearic / Algerian basin, late June
    (5.5, 37.5, 2.0, 185, 30, 3.4),   # off Algeria
    (12.0, 41.0, 3.0, 195, 40, 2.6),  # Tyrrhenian
    (4.5, 42.6, 1.5, 180, 25, 2.8),   # Gulf of Lion
    (16.0, 42.5, 2.5, 200, 35, 2.2),  # Adriatic
    (19.0, 36.0, 3.0, 215, 30, 2.1),  # Ionian
    (25.0, 38.5, 2.0, 225, 35, 2.4),  # Aegean
    (24.5, 34.5, 3.0, 230, 40, 3.2),  # Cretan Passage
    (30.5, 33.0, 3.0, 240, 45, 3.4),  # Levantine
    (33.5, 35.5, 2.0, 250, 30, 3.0),  # around Cyprus
    (14.0, 34.5, 2.5, 160, 25, 2.6),  # Gulf of Gabès / Sicily Channel
]
AMP_SCALE = 0.62
anom = np.zeros((365, H, W), dtype=np.float32)
state = np.zeros((H, W))
for t in days:
    state = 0.92 * state + 0.39 * smooth_noise(6)          # AR(1) weather noise
    field = 0.40 * state + 0.30                             # 2025 runs warm overall
    for lo, la, s, peak, dur, amp in events:
        tt = np.exp(-0.5 * ((t - peak) / (dur / 2.355)) ** 2)
        if tt > 1e-3:
            d2 = ((LON - lo) * np.cos(np.radians(la))) ** 2 + (LAT - la) ** 2
            field = field + AMP_SCALE * amp * tt * np.exp(-d2 / (2 * s ** 2)) * (1 + 0.3 * smooth_noise(4))
    anom[t] = field


def max_heatwave_category(anom, delta, min_days=5):
    """Hobday et al. 2018: a heatwave is >= min_days above the 90th percentile.
    Category = floor((SST - clim) / (p90 - clim)), capped at 4 (Extreme)."""
    above = anom > delta
    fwd = np.zeros_like(anom, dtype=np.int16)
    bwd = np.zeros_like(anom, dtype=np.int16)
    for t in range(len(anom)):
        fwd[t] = np.where(above[t], (fwd[t - 1] if t else 0) + 1, 0)
    for t in range(len(anom) - 1, -1, -1):
        bwd[t] = np.where(above[t], (bwd[t + 1] if t < len(anom) - 1 else 0) + 1, 0)
    in_event = above & (fwd + bwd - 1 >= min_days)
    ratio = np.where(in_event, anom / delta, 0)
    return np.clip(np.floor(ratio.max(axis=0)), 0, 4).astype(int)


cat = max_heatwave_category(anom, delta)
cat = ndimage.median_filter(cat, size=3)                     # tidy single-pixel speckle
cat_m = np.ma.masked_where(~med, cat)
share = {c: (cat[med] == i).mean() * 100 for i, c in enumerate(CATS)}

# ----------------------------------------------------------------------------
# 3. Synthetic Posidonia: dots along northern/western coasts, meadows for the inset
# ----------------------------------------------------------------------------
coast_sea = med & ndimage.binary_dilation(land_r, iterations=1)
yy, xx = np.nonzero(coast_sea)
plon, plat = lons[xx], lats[yy]
# Posidonia is rare along the hot, turbid SE Levant / Nile delta coasts
w = np.where((plon > 27) & (plat < 33.5), 0.08, 1.0) * np.where(plon < 16, 1.6, 1.0)
idx = rng.choice(len(plon), size=70, replace=False, p=w / w.sum())
# thin points that are too close together, like a curated point layer
keep = []
for i in idx:
    if all(np.hypot(plon[i] - plon[j], plat[i] - plat[j]) > 0.7 for j in keep):
        keep.append(i)
dots_lon, dots_lat = plon[keep], plat[keep]

INSET = (9.75, 11.35, 36.65, 37.45)       # Gulf of Tunis / Cap Bon window
INSET_C = ((INSET[0] + INSET[1]) / 2, (INSET[2] + INSET[3]) / 2)
fr = 0.0025
iw, ih = int((INSET[1] - INSET[0]) / fr), int((INSET[3] - INSET[2]) / fr)
land10 = gpd.read_file(DATA / "ne_10m_land.shp").clip(box(INSET[0] - 1, INSET[2] - 1, INSET[1] + 1, INSET[3] + 1))
iland = rasterize(((g, 1) for g in land10.geometry), out_shape=(ih, iw),
                  transform=from_origin(INSET[0], INSET[3], fr, fr), dtype="uint8").astype(bool)
dist_km = ndimage.distance_transform_edt(~iland) * fr * 111
n = ndimage.gaussian_filter(rng.standard_normal((ih, iw)), 6)
n /= n.std()
meadow = (~iland) & (dist_km < 3.0 + 2.2 * n)   # ragged shallow band
ilon = INSET[0] + (np.arange(iw) + 0.5) * fr
ilat = INSET[3] - (np.arange(ih) + 0.5) * fr

# ----------------------------------------------------------------------------
# 4. Render
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(7.6, 5.4), dpi=300)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.02, 0.12, 0.96, 0.745])
ax.set_anchor("N")
ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3])
ax.set_aspect(ASPECT); ax.axis("off")

cmap = ListedColormap(COLORS)
norm = BoundaryNorm(np.arange(-0.5, 5.5), cmap.N)
ax.pcolormesh(lons, lats, cat_m, cmap=cmap, norm=norm, shading="nearest", zorder=1, rasterized=True)
land50.clip(box(*[VIEW[0] - 1, VIEW[2] - 1, VIEW[1] + 1, VIEW[3] + 1])).boundary.plot(
    ax=ax, color=COAST, linewidth=0.3, zorder=2)

ax.scatter(dots_lon, dots_lat, s=16, color=DOT, edgecolor="white", linewidth=0.5, zorder=4)

# ocean labels
for txt, x, y in [("Atlantic Ocean", -7.5, 44.6), ("Black Sea", 34.5, 43.2)]:
    ax.text(x, y, txt, fontsize=5.5, style="italic", color=GREY_TXT, ha="center", zorder=5)


def rich_text(x, y, parts, size, ha="left", va="baseline", transform=None, linespacing=1.25):
    """Draw [(text, {props}), ...] as one run, with '\n' starting a new line."""
    transform = transform or fig.transFigure
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    lines, cur = [], []
    for txt, kw in parts:
        segs = txt.split("\n")
        for k, s in enumerate(segs):
            if k:
                lines.append(cur); cur = []
            if s:
                cur.append((s, kw))
    lines.append(cur)
    inv = transform.inverted()
    for li, line in enumerate(lines):
        # measure
        widths, artists = [], []
        for s, kw in line:
            a = fig.text(0, 0, s, fontsize=size, transform=fig.transFigure, **kw)
            widths.append(a.get_window_extent(rend).width)
            artists.append(a)
        total = sum(widths)
        x0_disp, y_disp = transform.transform((x, y))
        y_disp -= li * size * linespacing * fig.dpi / 72
        start = {"left": x0_disp, "center": x0_disp - total / 2, "right": x0_disp - total}[ha]
        for a, wdt in zip(artists, widths):
            a.set_position(fig.transFigure.inverted().transform((start, y_disp)))
            a.set_va(va)
            start += wdt


# title + subtitle (species name in italics)
rich_text(0.5, 0.945, [("Where can ", dict(fontweight="bold", color=NAVY)),
                       ("Posidonia oceanica", dict(fontweight="bold", style="italic", color=NAVY)),
                       (" meadows be found?", dict(fontweight="bold", color=NAVY))], 12, ha="center")
rich_text(0.5, 0.912, [("Posidonia oceanica", dict(style="italic", color=GREY_TXT)),
                      (" locations alongside the highest marine heatwave category experienced in the "
                       "Mediterranean Sea during 2025 (synthetic data)", dict(color=GREY_TXT))], 7.2, ha="center")

# callout: blue dots (elbow leader line to the northernmost Adriatic dot)
adr = np.argmax(np.where((dots_lon > 12) & (dots_lon < 20), dots_lat, -99))
tx, ty = 15.6, 46.6
ax.plot([dots_lon[adr], dots_lon[adr], tx - 0.3], [dots_lat[adr] + 0.35, ty + 0.15, ty + 0.15],
        color=COAST, lw=0.5, zorder=3)
rich_text(tx, ty, [("Blue dots", dict(fontweight="bold", color=NAVY)), (": known locations\n", dict(color=NAVY)),
                   ("where ", dict(color=NAVY)), ("Posidonia oceanica", dict(style="italic", color=NAVY)),
                   (" occurs.", dict(color=NAVY))], 6.6, transform=ax.transData)

# magnifier: small circle on the map, big circle inset below-left
r_small = 0.75
ax.add_patch(Circle(INSET_C, r_small, transform=ax.transData, fill=False, ec=COAST, lw=0.5, zorder=6))
# inset axes in figure coords
cx, cy, R = 0.365, 0.2, 0.085                       # centre + radius (fraction of fig width)
rw, rh = 2 * R, 2 * R * fig.get_figwidth() / fig.get_figheight()
iax = fig.add_axes([cx - R, cy - rh / 2, rw, rh])
iax.set_xlim(INSET_C[0] - 0.8, INSET_C[0] + 0.8)
iax.set_ylim(INSET_C[1] - 0.8 / ASPECT, INSET_C[1] + 0.8 / ASPECT)
iax.set_aspect(ASPECT); iax.axis("off")
clip = Circle((0.5, 0.5), 0.5, transform=iax.transAxes)
iax.add_patch(Circle((0.5, 0.5), 0.5, transform=iax.transAxes, fc="white", ec="none", zorder=0))
m = iax.pcolormesh(ilon, ilat, np.ma.masked_where(~meadow, meadow), cmap=ListedColormap([DOT]),
                   shading="nearest", zorder=1, rasterized=True)
cl = land10.boundary.plot(ax=iax, color=COAST, linewidth=0.45, zorder=2)
for art in iax.collections + iax.patches[1:]:
    art.set_clip_path(clip)
iax.add_patch(Circle((0.5, 0.5), 0.5, transform=iax.transAxes, fill=False, ec=COAST, lw=0.6, zorder=5))

# elbow leader: small circle -> down -> big circle top
p_small = fig.transFigure.inverted().transform(ax.transData.transform((INSET_C[0], INSET_C[1] - r_small / ASPECT * 1.0)))
p_big_top = (cx, cy + rh / 2)
fig.add_artist(plt.Line2D([p_small[0], p_small[0], cx, cx], [p_small[1], (p_small[1] + p_big_top[1]) / 2, (p_small[1] + p_big_top[1]) / 2, p_big_top[1]],
                          transform=fig.transFigure, color=COAST, lw=0.5))

rich_text(cx - R - 0.012, cy + 0.02, [("Blue areas", dict(fontweight="bold", color=NAVY)), (":\n", dict(color=NAVY)),
                                      ("where ", dict(color=NAVY)), ("Posidonia oceanica\n", dict(style="italic", color=NAVY)),
                                      ("form beds or meadows.", dict(color=NAVY))], 6.6, ha="right")

# legend
lx, ly, lw_, lh = 0.6, 0.175, 0.056, 0.019
fig.text(lx + 2.5 * lw_, ly + lh + 0.012, "Marine heatwave category", ha="center", fontsize=7.2,
         fontweight="bold", color=NAVY)
for k, (lab_, col) in enumerate(zip(CATS, COLORS)):
    fig.patches.append(Rectangle((lx + k * lw_ + 0.0015, ly), lw_ - 0.003, lh, transform=fig.transFigure,
                                 fc=col, ec=COAST, lw=0.4))
    fig.text(lx + (k + 0.5) * lw_, ly - 0.008, lab_, ha="center", va="top", fontsize=5.6, color="#3d4757")

fig.text(0.5, 0.048, "Data: synthetic SST anomalies & seagrass • Coastlines: Natural Earth • Reference period: 1991–2020",
         ha="center", fontsize=6.8, color=GREY_TXT, fontweight="medium")
fig.text(0.5, 0.024, "Style recreation of Copernicus C3S European State of the Climate, Fig. 20.1 • "
         "Heatwave categories after Hobday et al. (2018) • Made with geopandas + matplotlib",
         ha="center", fontsize=5.8, color="#a0a6b0")

fig.savefig(OUT, dpi=300, facecolor="white")
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
print("category shares (% of sea):", {k: round(v, 1) for k, v in share.items()}, "dots:", len(keep))
