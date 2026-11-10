"""
Recreation of the Copernicus/C3S "Ice discharge remained at an accelerated rate
in 2025" Greenland map (European State of the Climate 2025, Fig. 15.3).

Real data:      Natural Earth 10m Greenland outline.
Synthetic data: ice-sheet extent (outline eroded by a variable ice-free margin),
                surface topography, ice-flow velocity and drainage basins. The
                velocity comes from a simple flow model: ice drains downhill from
                a central dome, and speed grows with the upstream area feeding
                each point, so outlet glaciers emerge as branching streams.
Annotations:    discharge totals and anomalies are copied from the original figure.

To use real data instead: ENVEO/ESA CCI or MEaSUREs Greenland velocity mosaics
(GeoTIFF, EPSG:3413) for the raster, the GEUS/PROMICE discharge dataset for the
numbers, and Mouginot & Rignot (2019) basin polygons for the dashed boundaries.
"""
import heapq
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap, LogNorm, LightSource
from matplotlib.patches import FancyBboxPatch
from pyproj import Transformer
from rasterio.features import rasterize
from rasterio.transform import from_origin
from scipy import ndimage

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ne")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("greenland_ice_discharge_synthetic.png")
rng = np.random.default_rng(15)

for f in Path.home().joinpath(".fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Barlow Condensed"
NAVY, GREY_TXT, LINE = "#2e3a59", "#8a909c", "#2b3550"

# ----------------------------------------------------------------------------
# 1. Greenland on a 3 km polar-stereographic grid (EPSG:3413)
# ----------------------------------------------------------------------------
CRS = "EPSG:3413"
land = gpd.read_file(DATA / "ne_10m_land.shp").explode(index_parts=False)
rp = land.representative_point()
gl = land[(rp.x > -74) & (rp.x < -10) & (rp.y > 59.5) & ~((rp.x > -25) & (rp.y < 67))].to_crs(CRS)
gl = gl[gl.geometry.centroid.x > -700e3]                       # drop Canadian islands
XMIN, XMAX, YMIN, YMAX = -1060e3, 1180e3, -3390e3, -640e3
CELL = 3000
W, H = int((XMAX - XMIN) / CELL), int((YMAX - YMIN) / CELL)
tr = from_origin(XMIN, YMAX, CELL, CELL)
is_land = rasterize(((g, 1) for g in gl.geometry), out_shape=(H, W), transform=tr, dtype="uint8").astype(bool)
xs = XMIN + (np.arange(W) + 0.5) * CELL
ys = YMAX - (np.arange(H) + 0.5) * CELL
X, Y = np.meshgrid(xs, ys)
LON, LAT = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(X, Y)


def smooth_noise(sigma):
    n = ndimage.gaussian_filter(rng.standard_normal((H, W)), sigma)
    return n / n.std()


# Ice sheet = main landmass minus an ice-free margin of varying width
lab, _ = ndimage.label(is_land)
main = lab == np.bincount(lab.ravel())[1:].argmax() + 1
d_coast_km = ndimage.distance_transform_edt(main) * CELL / 1000
margin_km = (25 + 22 * smooth_noise(18) + 10 * smooth_noise(5)
             + 30 * np.exp(-((LAT - 82.3) / 1.2) ** 2)                 # ice-free far-north fringe
             + 35 * np.exp(-(((LON + 50) / 4) ** 2 + ((LAT - 67) / 2.5) ** 2))  # Kangerlussuaq area
             - 18 * np.exp(-(((LON + 38) / 5) ** 2 + ((LAT - 64) / 3) ** 2)))   # narrow SE
ice = main & (d_coast_km > np.clip(margin_km, 4, None))
ice = ndimage.binary_opening(ice, iterations=2)
lab_i, _ = ndimage.label(ice)
ice = lab_i == np.bincount(lab_i.ravel())[1:].argmax() + 1

# ----------------------------------------------------------------------------
# 2. Ice surface + flow routing (priority flood from the ice margin inward)
# ----------------------------------------------------------------------------
d_edge = ndimage.distance_transform_edt(ice) * CELL / 1000          # km from ice margin
r = np.clip(d_edge / d_edge.max(), 0, 1)
surf = 3000 * (1 - (1 - r) ** 2)                                      # dome, finite slope at the margin
# bedrock valleys: strong relief near the margin funnels ice into outlet glaciers
near = np.exp(-d_edge / 120)
surf += (220 * smooth_noise(9) + 120 * smooth_noise(4)) * (0.35 + near) + 20 * smooth_noise(1.5)
# Northeast Greenland Ice Stream: a long trough from the interior to the NE coast
p0 = np.array(Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(-38.5, 75.3))
p1 = np.array(Transformer.from_crs("EPSG:4326", CRS, always_xy=True).transform(-23.5, 79.4))
t = np.clip(((X - p0[0]) * (p1 - p0)[0] + (Y - p0[1]) * (p1 - p0)[1]) / np.sum((p1 - p0) ** 2), 0, 1)
dist_line = np.hypot(X - (p0[0] + t * (p1 - p0)[0]), Y - (p0[1] + t * (p1 - p0)[1])) / 1000
surf -= 260 * np.exp(-(dist_line / 22) ** 2) * (0.4 + 0.6 * t)

N = H * W
e = surf.ravel()
ice_f = ice.ravel()
rec = np.full(N, -1)
closed = ~ice_f.copy()
edge = ice & ndimage.binary_dilation(~ice, structure=np.ones((3, 3)))
# Outlet glaciers: margin cells where the ice reaches (nearly) to the sea, thinned to ~40 km
# spacing. Only these drain the ice sheet, so catchments converge into fast outlet streams.
cand = np.flatnonzero((edge & (d_coast_km < np.where(LAT > 78, 30, 14))).ravel())
w_out = np.exp(rng.normal(0, 0.6, cand.size))
outlets = []
for i in cand[np.argsort(-w_out)]:
    if all(np.hypot(X.flat[i] - X.flat[j], Y.flat[i] - Y.flat[j]) > 34e3 for j in outlets):
        outlets.append(i)
negis = np.argmin(np.where(edge, np.hypot(X - p1[0], Y - p1[1]), np.inf))
outlets.append(int(negis))
heap = []
for i in outlets:
    heapq.heappush(heap, (e[i], int(i)))
    closed[i] = True
order = []
nb = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
while heap:
    z, i = heapq.heappop(heap)
    order.append(i)
    r, c = divmod(i, W)
    for dr, dc in nb:
        rr, cc = r + dr, c + dc
        if 0 <= rr < H and 0 <= cc < W:
            j = rr * W + cc
            if not closed[j]:
                closed[j] = True
                rec[j] = i
                heapq.heappush(heap, (e[j], j))
order = np.array(order)

acc = np.where(ice_f, 1.0, 0.0)
for i in order[::-1]:
    if rec[i] >= 0:
        acc[rec[i]] += acc[i]
root = np.arange(N)
for i in order:
    if rec[i] >= 0:
        root[i] = root[rec[i]]

# velocity (m/day): grows with upstream area, plus a slow sheet-flow background
area_km2 = acc.reshape(H, W) * (CELL / 1000) ** 2
near = np.exp(-d_edge / 110)                                   # ice speeds up close to the margin
vel = 0.004 + 6e-4 * area_km2 ** 0.95 * (0.03 + 0.97 * near)
vel += 0.5 * np.exp(-(dist_line / 16) ** 2) * (0.2 + t) * ice           # NE Greenland Ice Stream
vel *= np.exp(0.25 * smooth_noise(3))
logv = ndimage.gaussian_filter(ndimage.maximum_filter(np.log10(vel), size=3), 1.0)  # widen 1-px streams
vel = np.where(ice, 10 ** logv, np.nan)

# ----------------------------------------------------------------------------
# 3. Drainage basins from the outlet each cell drains to
# ----------------------------------------------------------------------------
olon, olat = LON.ravel()[root], LAT.ravel()[root]
east = olon > -41.5 + 6 * np.clip((olat - 76) / 4, 0, 1)
basin = np.select(
    [olat >= 79.6, east & (olat >= 75.3), east & (olat >= 68.5), east,
     ~east & (olat >= 73.6), ~east & (olat >= 68.3)],
    [0, 1, 2, 3, 4, 5], default=6).reshape(H, W)            # N, NE, CE, SE, NW, CW, SW
basin = ndimage.median_filter(np.where(ice, basin, -1), size=5)
NAMES = ["North", "North\nEast", "Central\nEast", "South\nEast", "North\nWest", "Central\nWest", "South\nWest"]

# ----------------------------------------------------------------------------
# 4. Terrain for the ice-free margin (hillshade of synthetic relief)
# ----------------------------------------------------------------------------
dem = 400 * smooth_noise(1.2) + 600 * smooth_noise(4) + 900 * np.clip(d_coast_km / 60, 0, 1)
shade = LightSource(azdeg=315, altdeg=40).hillshade(dem, vert_exag=0.02, dx=CELL, dy=CELL)
margin = main & ~ice | (is_land & ~main)
grey = np.where(margin, 0.80 + 0.17 * shade, np.nan)

# ----------------------------------------------------------------------------
# 5. Render
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(6.6, 7.4), dpi=300)
fig.patch.set_facecolor("white")
fig.text(0.5, 0.965, "Ice discharge remained at an accelerated rate in 2025", ha="center", va="center",
         fontsize=16, fontweight="bold", color=NAVY)
fig.text(0.5, 0.915, "Ice flow velocity, ice discharge and relative ice discharge anomalies\n"
         "for the 2025 hydrological year (synthetic velocity field)", ha="center", va="center",
         fontsize=10.5, color=GREY_TXT, linespacing=1.2)

ax = fig.add_axes([0.03, 0.075, 0.94, 0.80])
ax.set_xlim(XMIN, XMAX); ax.set_ylim(YMIN, YMAX); ax.set_aspect("equal"); ax.axis("off")
ext = (XMIN, XMAX, YMIN, YMAX)

ax.imshow(grey, cmap="gray", vmin=0, vmax=1, extent=ext, interpolation="bilinear", zorder=1)
cmap = LinearSegmentedColormap.from_list("icevel", ["#ffffff", "#e6eff6", "#cddbe6", "#afcbdb", "#84b3c9",
                                                    "#579ac1", "#3277a9", "#195993", "#0e2e6b"])
norm = LogNorm(0.01, 10)
ax.imshow(vel, cmap=cmap, norm=norm, extent=ext, interpolation="bilinear", zorder=2)
ax.contour(xs, ys, ice.astype(float), levels=[0.5], colors="#1e1e1e", linewidths=0.6, zorder=3)
ax.contour(xs, ys, is_land.astype(float), levels=[0.5], colors="#9a9a9a", linewidths=0.3, zorder=3)

# dashed basin divides: zero-line between each touching pair of basins
soft = [ndimage.gaussian_filter((basin == k).astype(float), 2.0) for k in range(7)]
for a in range(7):
    for b in range(a + 1, 7):
        both = (soft[a] > 0.05) & (soft[b] > 0.05) & ice
        if both.sum() < 20:
            continue
        f = np.where(ndimage.binary_dilation(both, iterations=2), soft[a] - soft[b], np.nan)
        ax.contour(xs, ys, f, levels=[0], colors="#555555", linewidths=0.55,
                   linestyles=[(0, (3, 2.2))], zorder=4)

# basin labels with leader lines
VALUES = {0: ("26 Gt", "+5%"), 1: ("30 Gt", "+22%"), 2: ("77 Gt", "+4%"), 3: ("138 Gt", "+0%"),
          4: ("115 Gt", "+17%"), 5: ("84 Gt", "+5%"), 6: ("19 Gt", "0%")}
SIDE = {0: "left", 1: "right", 2: "right", 3: "right", 4: "left", 5: "left", 6: "left"}
NUDGE = {0: (-60e3, 60e3), 1: (-40e3, 0), 2: (40e3, 0), 3: (0, -60e3), 4: (0, 0), 5: (0, 0), 6: (40e3, -80e3)}
x_left, x_right = XMIN + 30e3, XMAX - 30e3


def rich(x, y, parts, va, size=10.5):
    """Left-aligned run of differently-styled text pieces, in data coordinates."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    xd, yd = ax.transData.transform((x, y))
    for txt, kw in parts:
        a_ = ax.text(0, 0, txt, fontsize=size, color=NAVY, va=va, zorder=8, **kw)
        a_.set_position(ax.transData.inverted().transform((xd, yd)))
        xd += a_.get_window_extent(rend).width
for k in range(7):
    yy, xx = np.nonzero(basin == k)
    cx, cy = xs[int(np.median(xx))] + NUDGE[k][0], ys[int(np.median(yy))] + NUDGE[k][1]
    ax.plot(cx, cy, "o", ms=3.5, color=LINE, zorder=8)
    ax.text(cx + 22e3, cy - 8e3, NAMES[k], fontsize=10, color=NAVY, va="top" if "\n" in NAMES[k] else "center", ha="left", zorder=8, linespacing=1.0)
    gt, an = VALUES[k]
    if SIDE[k] == "left":
        ax.plot([x_left, cx], [cy, cy], color=LINE, lw=0.6, zorder=7)
        if k == 0:
            rich(x_left, cy + 14e3, [("Total discharge: ", {}), (gt, {"fontweight": "bold"})], "bottom")
            rich(x_left, cy - 14e3, [("Anomaly: ", {}), (an, {"fontweight": "bold"})], "top")
        else:
            ax.text(x_left, cy + 14e3, gt, fontsize=10.5, color=NAVY, va="bottom", fontweight="bold")
            ax.text(x_left, cy - 14e3, an, fontsize=10.5, color=NAVY, va="top", fontweight="bold")
    else:
        ax.plot([cx, x_right], [cy, cy], color=LINE, lw=0.6, zorder=7)
        ax.text(x_right, cy + 14e3, gt, fontsize=10.5, color=NAVY, va="bottom", ha="right", fontweight="bold")
        ax.text(x_right, cy - 14e3, an, fontsize=10.5, color=NAVY, va="top", ha="right", fontweight="bold")

# "Entire ice sheet" box
bx, by = 0.69, 0.745
fig.patches.append(FancyBboxPatch((bx, by), 0.235, 0.105, boxstyle="square,pad=0", transform=fig.transFigure,
                                  fc="#e6effb", ec="none", zorder=0))
fig.text(bx + 0.018, by + 0.08, "Entire ice sheet", fontsize=12, fontweight="bold", color=NAVY, va="center")
fig.text(bx + 0.018, by + 0.048, "Total discharge: ", fontsize=10.5, color=NAVY, va="center")
fig.text(bx + 0.153, by + 0.048, "489 Gt", fontsize=10.5, color=NAVY, va="center", fontweight="bold")
fig.text(bx + 0.018, by + 0.02, "Anomaly: ", fontsize=10.5, color=NAVY, va="center")
fig.text(bx + 0.093, by + 0.02, "+7%", fontsize=10.5, color=NAVY, va="center", fontweight="bold")

# colorbar
cax = fig.add_axes([0.60, 0.115, 0.25, 0.018])
grad = np.logspace(-2, 1, 256)[None, :]
cax.imshow(grad, cmap=cmap, norm=norm, aspect="auto", extent=(0, 1, 0, 1))
cax.set_xticks([0.0, 1 / 3, 2 / 3, 1.0])
cax.set_xticklabels(["<0.01", "0.1", "1.0", ">10 m/day"])
cax.set_yticks([])
cax.tick_params(length=0, labelsize=9.5, colors="#3d4757", pad=3)
for s in cax.spines.values():
    s.set_edgecolor("#555555"); s.set_linewidth(0.5)
cax.set_title("Ice velocity", fontsize=10.5, fontweight="bold", color=NAVY, pad=4)

fig.text(0.07, 0.03, "Velocity field & basins: synthetic (flow-model) • Outline: Natural Earth • "
         "Discharge values: from the original C3S figure • Reference period: 1991–2020",
         fontsize=8.6, color=GREY_TXT)

fig.savefig(OUT, dpi=300, facecolor="white")
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
print("ice cells:", ice.sum(), "outlets:", len(outlets), "max vel:", np.nanmax(vel), "p99:", np.nanpercentile(vel, 99))
