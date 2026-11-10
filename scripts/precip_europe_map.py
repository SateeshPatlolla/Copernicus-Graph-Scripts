"""
Recreation of the Copernicus/C3S "West-east swathe of below-average
precipitation across Europe in 2025" map (European State of the Climate 2025).

Real data:      Natural Earth coastlines.
Synthetic data: the 2025 annual precipitation anomaly field (on an ERA5-like
                0.25° grid, so it keeps ERA5's blocky look), and analysis-region
                boxes approximating the ones in the figure.

To use real data instead: ERA5 monthly total precipitation (0.25°) -> 2025 annual
total -> per-pixel percentile vs 1991-2020; "Driest/Wettest" = lowest/highest in
1979-2025. Pass that category array in place of `cat` below.
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
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("precip_europe_2025_synthetic.png")
rng = np.random.default_rng(31)

for f in Path.home().joinpath(".fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Barlow Condensed"

NAVY, TICK = "#2e3a59", "#3d4757"
CATS = ["Driest", "Much drier\nthan average", "Drier\nthan average", "Near\naverage",
        "Wetter\nthan average", "Much wetter\nthan average", "Wettest"]
COLORS = ["#8c510a", "#c2974b", "#f0d897", "#f5f5f5", "#bedaea", "#67a9cf", "#2166ac"]

# ----------------------------------------------------------------------------
# 1. ERA5-like 0.25° land grid
# ----------------------------------------------------------------------------
RES = 0.25
LON0, LON1, LAT0, LAT1 = -80.0, 80.0, 24.0, 86.0
W, H = int((LON1 - LON0) / RES), int((LAT1 - LAT0) / RES)
lons = LON0 + (np.arange(W) + 0.5) * RES
lats = LAT1 - (np.arange(H) + 0.5) * RES
LON, LAT = np.meshgrid(lons, lats)

land = gpd.read_file(DATA / "ne_50m_land.shp").clip(box(-100, 15, 100, 90))
is_land = rasterize(((g, 1) for g in land.geometry), out_shape=(H, W),
                    transform=from_origin(LON0, LAT1, RES, RES), dtype="uint8", all_touched=False).astype(bool)


def smooth_noise(sigma):
    n = ndimage.gaussian_filter(rng.standard_normal((H, W)), sigma)
    return n / n.std()


# ----------------------------------------------------------------------------
# 2. Synthetic 2025 precipitation anomaly (z-score; negative = drier)
# ----------------------------------------------------------------------------
blobs = [  # lon, lat, sigma(deg), amplitude
    # the dry swathe, west to east
    (-1, 53, 2.0, -1.0), (4, 50, 4.0, -1.3), (12, 50, 4.0, -1.2), (10, 56, 1.8, -1.4),
    (23, 51, 4.0, -0.8), (33, 48, 5.0, -1.8), (36, 40, 4.5, -1.7), (45, 43, 4.0, -1.6),
    (57, 47, 6.0, -1.6), (27, 62, 3.0, -0.8), (-18, 65, 1.6, -1.0), (34, 66, 2.2, -1.8),
    # wetter areas
    (-7, 40, 3.0, 1.9), (-2, 38, 2.5, 0.6), (8, 63, 3.0, 1.4), (17, 68, 2.5, 1.0),
    (24, 56, 2.5, 1.2), (50, 61, 7.0, 1.1), (55, 66, 3.5, 1.7), (-42, 72, 6.0, 1.0),
    (-65, 70, 4.0, 1.6), (12, 42, 3.0, 0.5), (3, 33, 5.0, 0.4), (40, 55, 3.0, 0.6),
]
z = np.zeros((H, W))
for lo, la, s, a in blobs:
    a = a * 0.72 if a < 0 else a                    # keep "Driest" to patches, as in the original
    d2 = ((LON - lo) * np.cos(np.radians(la))) ** 2 + (LAT - la) ** 2
    z += a * np.exp(-d2 / (2 * s ** 2))
z += 0.55 * smooth_noise(6) + 0.35 * smooth_noise(2) + 0.15 * smooth_noise(0.7)
cat = np.digitize(z, [-1.95, -1.2816, -0.4307, 0.4307, 1.2816, 1.95])
cat_m = np.ma.masked_where(~is_land, cat)

# ----------------------------------------------------------------------------
# 3. Projection + analysis regions
# ----------------------------------------------------------------------------
PROJ = "+proj=laea +lat_0=52 +lon_0=10 +datum=WGS84 +units=m"
fwd = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True)
XMIN, XMAX, YMIN, YMAX = -3.46e6, 3.00e6, -2.14e6, 3.62e6
XE, YE = fwd.transform(*np.meshgrid(np.linspace(LON0, LON1, W + 1), np.linspace(LAT1, LAT0, H + 1)))

REGIONS = {  # code: (name, lon_min, lon_max, lat_min, lat_max) — approximations of the figure's boxes
    "NWE": ("Northwestern Europe", -12, 10, 48, 61),
    "CE": ("Central Europe", 2, 24, 45.6, 55),
    "EE": ("Eastern Europe", 24.6, 50, 45.6, 60),
    "SC": ("Scandinavia", 4, 32, 55, 71),
    "MD": ("Mediterranean", -10, 40, 30, 45),
}


def box_line(lo0, lo1, la0, la1, n=120):
    """A lon/lat rectangle densified so it curves correctly in the projection."""
    lo = np.r_[np.linspace(lo0, lo1, n), np.full(n, lo1), np.linspace(lo1, lo0, n), np.full(n, lo0)]
    la = np.r_[np.full(n, la1), np.linspace(la1, la0, n), np.full(n, la0), np.linspace(la0, la1, n)]
    return fwd.transform(lo, la)


# ----------------------------------------------------------------------------
# 4. Render
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(7.4, 7.15), dpi=300)
fig.patch.set_facecolor("white")
fig.text(0.5, 0.965, "West-east swathe of below-average precipitation across Europe in 2025",
         ha="center", va="center", fontsize=14.5, fontweight="bold", color=NAVY)
fig.text(0.5, 0.933, "Anomalies and extremes in annual precipitation in 2025 (synthetic data)",
         ha="center", va="center", fontsize=11, color="#8a909c")

mw = 0.80
mh = mw * (YMAX - YMIN) / (XMAX - XMIN) * fig.get_figwidth() / fig.get_figheight()
ax = fig.add_axes([(1 - mw) / 2 + 0.06, 0.905 - mh, mw - 0.02, mh])
ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX); ax.set_aspect("equal"); ax.axis("off")

cmap = ListedColormap(COLORS)
ax.pcolormesh(XE, YE, cat_m, cmap=cmap, norm=BoundaryNorm(np.arange(-0.5, 7.5), 7),
              shading="flat", zorder=1, rasterized=True)
coast = gpd.read_file(DATA / "ne_50m_land.shp").boundary.clip(box(-99, 16, 99, 89.5))  # no clip-edge lines
coast.to_crs(PROJ).plot(ax=ax, color="black", lw=0.35, zorder=2)
ax.set_xlabel(""); ax.set_ylabel("")

for code, (_, lo0, lo1, la0, la1) in REGIONS.items():
    x, y = box_line(lo0, lo1, la0, la1)
    ax.plot(x, y, color="#6b6b6b", lw=0.5, ls=(0, (3, 2)), zorder=3)
    # label just inside the top-left corner, rotated to follow the top edge
    p0 = np.array(fwd.transform(lo0 + 0.6, la1 - 1.6))
    p1 = np.array(fwd.transform(lo0 + 4.0, la1 - 1.6))
    ang = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(*p0, code, fontsize=8.5, fontweight="bold", style="italic", color=NAVY,
            rotation=ang, rotation_mode="anchor", ha="left", va="center", zorder=5)

# region key, upper left
for k, (code, (name, *_)) in enumerate(REGIONS.items()):
    yk = 0.60 - k * 0.032                                   # axes coords, over the open Atlantic
    ax.text(-0.04, yk, f"{code}:", transform=ax.transAxes, fontsize=9.5, fontweight="bold", color=NAVY, va="center")
    ax.text(0.025, yk, name, transform=ax.transAxes, fontsize=9.5,
            color=NAVY, va="center")

# category legend
lw_, lh, ly = 0.124, 0.02, 0.915 - mh - 0.04
lx0 = 0.5 - 3.5 * lw_ + 0.005
for k, (lab, col) in enumerate(zip(CATS, COLORS)):
    fig.patches.append(Rectangle((lx0 + k * lw_ + 0.0015, ly), lw_ - 0.003, lh, transform=fig.transFigure,
                                 fc=col, ec="none"))
    fig.text(lx0 + (k + 0.5) * lw_, ly - 0.008, lab, ha="center", va="top", fontsize=9.5, color=TICK,
             linespacing=1.05)

fig.text(0.5, ly - 0.082, "Data: synthetic precipitation anomalies (2025) • Coastlines: Natural Earth • "
         "Reference period: 1991–2020 • Region boxes approximate",
         ha="center", fontsize=9, color="#8a909c")

fig.savefig(OUT, dpi=300, facecolor="white")
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
eu = is_land & (LON > -10) & (LON < 40) & (LAT > 36) & (LAT < 70)
print("Europe land drier than average: %.0f%%" % ((cat[eu] <= 2).mean() * 100))
