"""
Recreation of the Copernicus/C3S "Record-high sea surface temperature over
European seas in 2025" figure style (European State of the Climate 2025).

Real data:      Natural Earth coastlines.
Synthetic data: annual SST anomaly series (1982-2025) and the 2025 anomaly
                field, sea-ice edge and an approximate WMO RA VI boundary.

To use real data instead: C3S/ESA CCI SST v3 daily L4 files -> annual means ->
anomalies vs 1991-2020. Area-weight (cos lat) over the RA VI ocean mask and the
Mediterranean for the bar charts; rank each pixel's 2025 value within its own
1982-2025 history for the map categories.
"""
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage
from shapely.geometry import box

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ne")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("sst_europe_2025_synthetic.png")
rng = np.random.default_rng(7)

for f in Path.home().joinpath(".fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Barlow Condensed"

NAVY, TICK, GRID = "#3c4b68", "#4a5262", "#ececec"
BLUE, RED, RED_25 = "#699bc9", "#ce686f", "#ba2832"
CATS = ["Coolest", "Much cooler\nthan\naverage", "Cooler\nthan\naverage", "Near\naverage",
        "Warmer\nthan\naverage", "Much warmer\nthan\naverage", "Warmest"]
COLORS = ["#2a71b2", "#6bacd1", "#c2ddec", "#f9f9f9", "#fbccb4", "#e48066", "#ba2832"]

# ----------------------------------------------------------------------------
# 1. Synthetic annual anomaly series (relative to 1991-2020)
# ----------------------------------------------------------------------------
years = np.arange(1982, 2026)


def series(start, end, noise, last):
    s = np.interp(years, [1982, 1998, 2008, 2022], [start, -0.15, 0.10, end])
    s = s + rng.normal(0, noise, years.size)
    ref = (years >= 1991) & (years <= 2020)
    s -= s[ref].mean()
    for y, v in last.items():
        s[years == y] = v
    return s


europe = series(-0.55, 0.30, 0.09, {2022: 0.40, 2023: 0.55, 2024: 0.58, 2025: 0.65})
med = series(-0.60, 0.45, 0.20, {2022: 0.66, 2023: 0.87, 2024: 1.18, 2025: 1.03})

# ----------------------------------------------------------------------------
# 2. Synthetic 2025 SST anomaly field (z-scores of the 2025 annual mean)
# ----------------------------------------------------------------------------
RES = 0.1
LON0, LON1, LAT0, LAT1 = -75.0, 65.0, 22.0, 88.0
W, H = int((LON1 - LON0) / RES), int((LAT1 - LAT0) / RES)
lons = LON0 + (np.arange(W) + 0.5) * RES
lats = LAT1 - (np.arange(H) + 0.5) * RES
LON, LAT = np.meshgrid(lons, lats)

land = gpd.read_file(DATA / "ne_50m_land.shp").clip(box(-100, 15, 90, 90))
is_land = rasterize(((g, 1) for g in land.geometry), out_shape=(H, W),
                    transform=from_origin(LON0, LAT1, RES, RES), dtype="uint8").astype(bool)
water = ~is_land


def smooth_noise(sigma):
    n = ndimage.gaussian_filter(rng.standard_normal((H, W)), sigma)
    return n / n.std()


z = np.full((H, W), 0.80)                       # 2025: warm almost everywhere
blobs = [  # lon, lat, sigma(deg), amplitude
    (4, 58, 5, 1.6),     # North Sea / Skagerrak
    (10, 66, 5, 1.3),    # Norwegian coast
    (40, 73, 9, 1.5),    # Barents Sea
    (20, 58, 4, 1.4),    # Baltic
    (-12, 46, 6, 1.4),   # Bay of Biscay / west of France
    (-22, 50, 5, 0.8),   # mid-latitude Atlantic patch
    (5, 39, 4, 1.6),     # western Mediterranean
    (18, 36, 4, 1.0),    # central Mediterranean
    (30, 34, 3, 1.0),    # Levantine
    (35, 43, 4, 1.0),    # Black Sea
    (-18, 72, 6, -2.2),  # Greenland Sea: near average / cooler
    (-38, 60, 6, -1.6),  # Irminger / southern Greenland
    (-30, 54, 5, -1.0),  # subpolar gyre
    (-55, 58, 7, -1.2),  # Labrador Sea
]
for lo, la, s, a in blobs:
    d2 = ((LON - lo) * np.cos(np.radians(la))) ** 2 + (LAT - la) ** 2
    z += a * np.exp(-d2 / (2 * s ** 2))
z += 0.55 * smooth_noise(14) + 0.30 * smooth_noise(5) + 0.10 * smooth_noise(1.5)

edges = [-1.95, -1.2816, -0.4307, 0.4307, 1.2816, 1.95]   # caption-style percentile bins
cat = np.digitize(z, edges)

# Sea ice 2025: Arctic pack plus the East Greenland band
green = rasterize(((g, 1) for g in land.clip(box(-75, 59, -25.5, 85)).geometry), out_shape=(H, W),
                  transform=from_origin(LON0, LAT1, RES, RES), dtype="uint8").astype(bool)
d_green_km = ndimage.distance_transform_edt(~green, sampling=[RES * 111, RES * 111 * np.cos(np.radians(72))])
ice = ((LAT > 80.5 - 1.5 * np.clip((LON - 20) / 30, 0, 1)) |
       ((LON < -5) & (LAT > 65) & (d_green_km < 60 + 18 * (LAT - 65))))
ice = ndimage.gaussian_filter(ice.astype(float), 5) + 0.10 * smooth_noise(5) > 0.5
ice &= water

cat_m = np.ma.masked_where(~water | ice, cat)

# ----------------------------------------------------------------------------
# 3. Projection (Lambert azimuthal equal-area centred on the North-East Atlantic)
# ----------------------------------------------------------------------------
PROJ = "+proj=laea +lat_0=63 +lon_0=-5 +datum=WGS84 +units=m"
fwd = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True)
# corner grid for pcolormesh
lon_e = np.linspace(LON0, LON1, W + 1)
lat_e = np.linspace(LAT1, LAT0, H + 1)
LONE, LATE = np.meshgrid(lon_e, lat_e)
XE, YE = fwd.transform(LONE, LATE)
XC, YC = fwd.transform(LON, LAT)
land_p = land.to_crs(PROJ)

# view window chosen to match the original's crop (Greenland & Svalbard top, Black Sea cut at right)
XMIN, XMAX, YMIN, YMAX = -2.55e6, 3.05e6, -2.95e6, 2.05e6
MAP_ASPECT = (YMAX - YMIN) / (XMAX - XMIN)

# Approximate WMO RA VI boundary segments (illustrative, not official)
ra6 = [
    [(-8, 86), (-18, 78), (-28, 70), (-36, 63), (-42, 57), (-48, 50), (-55, 42), (-60, 30)],
    [(-35, 35.0), (-6, 35.9), (5, 37.6), (11, 37.6), (16, 35.5), (25, 34.0), (34, 32.0), (40, 31)],
]

# ----------------------------------------------------------------------------
# 4. Render
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(9.6, 5.4), dpi=300)
fig.patch.set_facecolor("white")
fig.text(0.5, 0.955, "Record-high sea surface temperature over European seas in 2025",
         ha="center", va="center", fontsize=16, fontweight="bold", color=NAVY)


def bars(rect, data, title):
    ax = fig.add_axes(rect)
    cols = [RED_25 if y == 2025 else (RED if v >= 0 else BLUE) for y, v in zip(years, data)]
    ax.bar(years, data, width=0.72, color=cols, zorder=2)
    ax.set_ylim(-1.0, 1.5)
    ax.set_xlim(1981.2, 2025.8)
    ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    ax.set_yticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0", "1.5°C"])
    ax.set_xticks([1982, 1990, 2000, 2010, 2020, 2025])
    ax.set_xticklabels(["1982", "1990", "2000", "2010", "2020", "2025"])
    ax.get_xticklabels()[-1].set_fontweight("bold")
    ax.tick_params(length=0, labelsize=9, colors=TICK, pad=3)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=11.5, fontweight="semibold", color=NAVY, pad=10)
    return ax


bars([0.105, 0.585, 0.355, 0.245], europe, "SST anomalies for WMO RA VI (Europe) region")
bars([0.105, 0.215, 0.355, 0.245], med, "SST anomalies for the Mediterranean Sea")

# map
mx0, my0, mw = 0.495, 0.235, 0.37
mh = mw * MAP_ASPECT * fig.get_figwidth() / fig.get_figheight()
ax = fig.add_axes([mx0, my0, mw, mh])
ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX)
ax.set_aspect("equal")
ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Anomalies and extremes in annual SST in 2025", fontsize=11.5, fontweight="semibold",
             color=NAVY, pad=10)

cmap = ListedColormap(COLORS)
norm = BoundaryNorm(np.arange(-0.5, 7.5), cmap.N)
ax.pcolormesh(XE, YE, cat_m, cmap=cmap, norm=norm, shading="flat", zorder=1, rasterized=True)
land_p.cx[XMIN - 5e5:XMAX + 5e5, YMIN - 5e5:YMAX + 5e5].boundary.plot(ax=ax, color="black", lw=0.45, zorder=3)
ax.set_xlabel(""); ax.set_ylabel("")
ax.contour(XC, YC, ice.astype(float), levels=[0.5], colors="#555555", linewidths=0.7,
           linestyles=[(0, (1, 1.6))], zorder=4)
for seg in ra6:
    lo, la = zip(*seg)
    lo_d = np.interp(np.linspace(0, len(lo) - 1, 200), np.arange(len(lo)), lo)
    la_d = np.interp(np.linspace(0, len(la) - 1, 200), np.arange(len(la)), la)
    x, y = fwd.transform(lo_d, la_d)
    ax.plot(x, y, color="#7a7f88", lw=0.7, ls=(0, (4, 2.5)), zorder=4)

# small legend for the lines, to the right of the map
lx = mx0 + mw + 0.012
for k, (ls, txt) in enumerate([((0, (4, 2)), "WMO RA VI\nregion"), ((0, (1, 1.6)), "Sea ice edge\nin 2025")]):
    yy = my0 + mh * (0.66 - k * 0.11)
    fig.add_artist(Line2D([lx, lx + 0.022], [yy, yy], transform=fig.transFigure, color="black", lw=1.0, ls=ls))
    fig.text(lx + 0.03, yy + 0.008, txt, fontsize=8, color=TICK, va="top", linespacing=1.05)

# category legend under the map
lw_, lh, ly = mw / 7, 0.022, my0 - 0.045
for k, (lab, col) in enumerate(zip(CATS, COLORS)):
    fig.patches.append(Rectangle((mx0 + k * lw_ + 0.002, ly), lw_ - 0.004, lh, transform=fig.transFigure,
                                 fc=col, ec="none"))
    fig.text(mx0 + (k + 0.5) * lw_, ly - 0.012, lab, ha="center", va="top", fontsize=8.6, color=TICK,
             linespacing=1.0)

fig.text(0.5, 0.045, "Data: synthetic SST anomalies (1982–2025) • Coastlines: Natural Earth • "
         "Reference period: 1991–2020 • RA VI boundary approximate",
         ha="center", fontsize=8.6, color="#8a909c")
fig.text(0.5, 0.018, "Style recreation of Copernicus C3S European State of the Climate • Made with geopandas + matplotlib",
         ha="center", fontsize=7.2, color="#a0a6b0")

fig.savefig(OUT, dpi=300, facecolor="white")
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
vis = water & ~ice
print("map shares:", {c.replace(chr(10), ' '): round((cat[vis] == i).mean() * 100, 1) for i, c in enumerate(CATS)})
