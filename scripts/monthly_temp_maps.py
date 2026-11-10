"""
Recreation of the Copernicus/C3S "Temperature variations through 2025" small-
multiples figure (E-OBS tab, European State of the Climate 2025).

Real data:      Natural Earth coastlines.
Synthetic data: twelve monthly surface air temperature anomaly fields for 2025,
                shaped after the published panels.

To use real data instead: E-OBS daily mean temperature (tg, 0.1°) -> monthly
means -> per-pixel percentiles of 1991-2020 for each calendar month; categories
"Coolest/Warmest" = lowest/highest value in the 1979-2025 record for that month.
Regrid to the same EPSG:3035 grid used here (e.g. with rioxarray.reproject) and
pass the 12 category arrays to draw_panel().
"""
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import box

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ne")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("monthly_temp_2025_synthetic.png")
rng = np.random.default_rng(12)

for f in Path.home().joinpath(".fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams.update({"font.family": "Barlow Condensed", "hatch.color": "#9c9c9c", "hatch.linewidth": 0.45})

NAVY, TICK = "#3c4b68", "#3d4757"
CATS = ["Coolest", "Much cooler\nthan average", "Cooler\nthan average", "Near\naverage",
        "Warmer\nthan average", "Much warmer\nthan average", "Warmest"]
COLORS = ["#2a71b2", "#6bacd1", "#c2ddec", "#f9f9f9", "#fbccb4", "#e48066", "#ba2832"]
NODATA_GREY, HATCH_BG = "#a9a9a9", "#dcdcdc"

# ----------------------------------------------------------------------------
# 1. Projected 10 km grid (EPSG:3035) and masks
# ----------------------------------------------------------------------------
CRS = "EPSG:3035"
XMIN, XMAX, YMIN, YMAX = 2.50e6, 6.90e6, 1.40e6, 5.20e6
CELL = 10_000
W, H = int((XMAX - XMIN) / CELL), int((YMAX - YMIN) / CELL)
xs = XMIN + (np.arange(W) + 0.5) * CELL
ys = YMAX - (np.arange(H) + 0.5) * CELL
X, Y = np.meshgrid(xs, ys)
LON, LAT = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(X, Y)

land = gpd.read_file(DATA / "ne_50m_land.shp").clip(box(-35, 20, 75, 82)).to_crs(CRS)
is_land = rasterize(((g, 1) for g in land.geometry), out_shape=(H, W),
                    transform=from_origin(XMIN, YMAX, CELL, CELL), dtype="uint8").astype(bool)

LON_EDGE = 40.0                                   # data-domain edge (a meridian)
outside = LON > LON_EDGE                          # hatched block, land and sea alike
iceland = is_land & (LON < -12.5) & (LAT > 62.5)  # shown as solid grey (no data)
valid = is_land & ~outside & ~iceland


def smooth_noise(sigma):
    n = ndimage.gaussian_filter(rng.standard_normal((H, W)), sigma)
    return n / n.std()


# ----------------------------------------------------------------------------
# 2. Synthetic monthly anomaly fields (z-scores vs 1991-2020)
# ----------------------------------------------------------------------------
MONTHS = {  # base, [(lon, lat, sigma_deg, amplitude), ...]
    "January":   (0.55, [(36, 53, 7, 1.8), (22, 50, 7, 1.0), (-5, 40, 4, -0.7), (4, 32, 4, 1.8), (-3, 55, 3, -0.6)]),
    "February":  (0.15, [(33, 43, 6, -1.2), (20, 67, 5, 1.0), (-4, 40, 4, 0.7), (10, 48, 6, -0.3)]),
    "March":     (1.00, [(38, 53, 6, 2.0), (22, 52, 6, 0.8), (-4, 40, 4, -1.2), (6, 33, 3, 2.0)]),
    "April":     (0.70, [(35, 42, 4, -1.3), (2, 48, 6, 0.6), (30, 56, 5, -0.6), (15, 62, 4, 0.5)]),
    "May":       (0.10, [(21, 52, 6, -1.8), (-6, 54, 4, 1.2), (30, 45, 4, -0.6), (-2, 38, 4, 0.5)]),
    "June":      (0.40, [(0, 42, 7, 1.8), (14, 46, 6, 1.0), (32, 60, 6, -0.6), (38, 40, 2.5, -2.0), (-5, 53, 3, 1.2)]),
    "July":      (0.40, [(22, 65, 6, 1.7), (35, 50, 6, 1.0), (4, 48, 5, -0.6), (30, 38, 4, 1.2)]),
    "August":    (0.40, [(-3, 40, 5, 1.6), (26, 56, 6, -1.1), (32, 39, 4, 1.3), (15, 60, 4, 0.8)]),
    "September": (0.50, [(26, 60, 9, 1.4), (-5, 40, 4, -0.8), (10, 45, 5, -0.4), (36, 46, 5, 0.8)]),
    "October":   (0.30, [(36, 51, 6, 1.3), (15, 43, 4, -0.9), (20, 67, 4, 1.1), (0, 47, 5, -0.3)]),
    "November":  (0.30, [(35, 48, 7, 1.9), (10, 62, 5, -0.8), (32, 39, 4, 1.2), (-2, 44, 5, -0.3)]),
    "December":  (0.70, [(20, 50, 8, 1.2), (-6, 40, 3, -0.8), (32, 58, 6, 0.6), (30, 40, 4, 0.9)]),
}
edges = [-1.95, -1.2816, -0.4307, 0.4307, 1.2816, 1.95]   # percentile bins; |z|>1.95 ≈ record


def month_field(base, blobs):
    z = np.full((H, W), base)
    for lo, la, s, a in blobs:
        d2 = ((LON - lo) * np.cos(np.radians(la))) ** 2 + (LAT - la) ** 2
        z += a * np.exp(-d2 / (2 * s ** 2))
    # relief-like texture: stations + terrain make E-OBS fields locally patchy
    z += 0.40 * smooth_noise(12) + 0.25 * smooth_noise(3) + 0.12 * smooth_noise(0.8)
    return np.digitize(z, edges)


cats = {m: month_field(*v) for m, v in MONTHS.items()}

# ----------------------------------------------------------------------------
# 3. Render the 3 x 4 small multiples
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(8.4, 7.75), dpi=300)
fig.patch.set_facecolor("white")
fig.text(0.5, 0.965, "Temperature variations through 2025", ha="center", va="center",
         fontsize=15, fontweight="bold", color=NAVY)
fig.text(0.5, 0.932, "Anomalies and extremes in monthly surface air temperature in 2025 (synthetic data)",
         ha="center", va="center", fontsize=11, color="#8a909c")

cmap = ListedColormap(COLORS)
norm = BoundaryNorm(np.arange(-0.5, 7.5), cmap.N)
extent = (XMIN, XMAX, YMIN, YMAX)
coast = land.cx[XMIN - 3e5:XMAX + 3e5, YMIN - 3e5:YMAX + 3e5].boundary

pw = 0.215                                   # panel width (fig fraction)
ph = pw * (YMAX - YMIN) / (XMAX - XMIN) * fig.get_figwidth() / fig.get_figheight()
gx, gy = 0.025, 0.04
x0 = (1 - 4 * pw - 3 * gx) / 2
ytop = 0.885


def draw_panel(ax, cat, title):
    ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX); ax.set_aspect("equal"); ax.axis("off")
    # hatched no-data block
    ax.contourf(xs, ys, outside.astype(float), levels=[0.5, 1.5], colors=[HATCH_BG], hatches=["//////"], zorder=1)
    ax.imshow(np.ma.masked_where(~valid, cat), cmap=cmap, norm=norm, extent=extent,
              interpolation="nearest", zorder=2)
    ax.imshow(np.ma.masked_where(~iceland, np.ones_like(cat)), cmap=ListedColormap([NODATA_GREY]),
              extent=extent, interpolation="nearest", zorder=2)
    # coastline only where it isn't inside the hatched block
    c = coast.clip(box(XMIN, YMIN, XMAX, YMAX))
    c.plot(ax=ax, color="black", lw=0.25, zorder=3)
    ax.contourf(xs, ys, outside.astype(float), levels=[0.5, 1.5], colors=[HATCH_BG], hatches=["//////"], zorder=4)
    ax.set_title(title, fontsize=10, fontweight="bold", color=NAVY, pad=3)


for k, (m, cat) in enumerate(cats.items()):
    r, c = divmod(k, 4)
    ax = fig.add_axes([x0 + c * (pw + gx), ytop - (r + 1) * ph - r * gy, pw, ph])
    draw_panel(ax, cat, m)

# legend
lw_, lh = 0.098, 0.02
lx0, ly = 0.5 - 3.5 * lw_, ytop - 3 * ph - 2 * gy - 0.06
for k, (lab, col) in enumerate(zip(CATS, COLORS)):
    fig.patches.append(Rectangle((lx0 + k * lw_ + 0.0015, ly), lw_ - 0.003, lh, transform=fig.transFigure,
                                 fc=col, ec="none"))
    fig.text(lx0 + (k + 0.5) * lw_, ly - 0.01, lab, ha="center", va="top", fontsize=9.5, color=TICK,
             linespacing=1.05)

fig.text(0.5, 0.036, "Data: synthetic monthly anomalies (2025) • Coastlines: Natural Earth • Reference period: 1991–2020",
         ha="center", fontsize=9.5, color="#8a909c")
fig.text(0.5, 0.014, "Style recreation of Copernicus C3S European State of the Climate (E-OBS tab) • Made with geopandas + matplotlib",
         ha="center", fontsize=7.8, color="#a0a6b0")

fig.savefig(OUT, dpi=300, facecolor="white")
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
print({m: round((c[valid] >= 4).mean() * 100) for m, c in cats.items()}, "% of land warmer than average")
