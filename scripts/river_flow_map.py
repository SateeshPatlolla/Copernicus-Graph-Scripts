"""
Recreation of the Copernicus/C3S "Annual river flow anomalies 2025" map style.

Real data:      Natural Earth land polygons (coastlines).
Synthetic data: the river network (derived from a synthetic terrain with a
                priority-flood D8 drainage model, so it looks dendritic like
                EFAS' LISFLOOD network) and the 2025 flow anomalies (a smooth
                field shaped like the published pattern: dry east, wet Iberia
                and western Norway).

To use a real network instead, swap build_synthetic_network() for e.g.
HydroRIVERS (https://www.hydrosheds.org/products/hydrorivers), filter
UPLAND_SKM > 1000, and join your own anomaly categories onto it.
"""
import heapq
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle
from matplotlib import font_manager
from pyproj import Transformer
from scipy import ndimage
from shapely.geometry import box

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ne")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("river_flow_2025_synthetic.png")
rng = np.random.default_rng(2025)

# ----------------------------------------------------------------------------
# 1. Projection and extents (ETRS89-LAEA Europe, EPSG:3035)
# ----------------------------------------------------------------------------
CRS = "EPSG:3035"
VIEW = (2.45e6, 7.45e6, 1.30e6, 5.58e6)    # xmin, xmax, ymin, ymax shown
DOMAIN = VIEW
LON_MAX = 44.0     # model domain edge: a meridian, so it slants in LAEA like EFAS's
CELL = 5_000       # 5 km grid
MIN_AREA_KM2 = 600 # original uses 1000 km²; a little lower reads denser at this size

land = gpd.read_file(DATA / "ne_50m_land.shp").clip(box(-26, 25, 110, 73)).to_crs(CRS)  # no Greenland
land = land.clip(box(VIEW[0], VIEW[2], VIEW[1], VIEW[3]))

# ----------------------------------------------------------------------------
# 2. Synthetic terrain + drainage network
# ----------------------------------------------------------------------------
def rasterize_land(gdf, xmin, xmax, ymin, ymax, cell):
    from rasterio.features import rasterize
    from rasterio.transform import from_origin
    W = int((xmax - xmin) / cell)
    H = int((ymax - ymin) / cell)
    tr = from_origin(xmin, ymax, cell, cell)
    m = rasterize(((g, 1) for g in gdf.geometry), out_shape=(H, W), transform=tr, dtype="uint8")
    return m.astype(bool), H, W


def fractal_noise(shape, scales_px, weights):
    out = np.zeros(shape)
    for s, w in zip(scales_px, weights):
        n = ndimage.gaussian_filter(rng.standard_normal(shape), s, mode="wrap")
        out += w * n / n.std()
    return out


def build_synthetic_network():
    xmin, xmax, ymin, ymax = DOMAIN
    is_land, H, W = rasterize_land(land, xmin, xmax, ymin, ymax, CELL)
    # drop specks so tiny islets don't seed noise
    real_land = ndimage.binary_opening(is_land, iterations=1)
    rows, cols = np.divmod(np.arange(H * W), W)
    x = xmin + (cols + 0.5) * CELL
    y = ymax - (rows + 0.5) * CELL
    lon_c, _ = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y)
    lon2d, lat2d = (a.reshape(H, W) for a in Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y))
    in_domain = (lon2d <= LON_MAX) & ~((lon2d > 26) & (lat2d < 36.6))  # stop at Turkey's southern border
    is_land = real_land & in_domain                          # rivers only inside the domain

    dist = ndimage.distance_transform_edt(real_land)         # cells to the real coast
    noise = fractal_noise((H, W), [1.2, 4, 12, 40], [0.35, 0.8, 1.2, 1.8])
    elev = dist * 0.10 + noise * 4.0                         # gentle coast-down slope + relief

    # Priority-flood (Barnes et al. 2014): grows the drainage tree inland from
    # the sea, which removes pits and assigns every land cell a receiver.
    N = H * W
    rec = np.full(N, -1, dtype=np.int64)
    closed = ~is_land.ravel().copy()
    e = elev.ravel()
    heap = []
    land_idx = np.flatnonzero(is_land.ravel())
    sea_adj = ndimage.binary_dilation(~is_land, structure=np.ones((3, 3))) & is_land
    sea_adj[[0, -1], :] |= is_land[[0, -1], :]               # domain edges drain out
    sea_adj[:, [0, -1]] |= is_land[:, [0, -1]]
    for i in np.flatnonzero(sea_adj.ravel()):
        heapq.heappush(heap, (e[i], int(i)))
        closed[i] = True
    order = []
    nbrs = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    while heap:
        z, i = heapq.heappop(heap)
        order.append(i)
        r, c = divmod(i, W)
        for dr, dc in nbrs:
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W:
                j = rr * W + cc
                if not closed[j]:
                    closed[j] = True
                    rec[j] = i
                    heapq.heappush(heap, (e[j], j))  # raw key: depressions drain along terrain, no flat-fill hatching
    order = np.array(order)

    # flow accumulation: walk the tree from the headwaters down
    acc = np.zeros(N)
    acc[land_idx] = 1.0
    for i in order[::-1]:
        if rec[i] >= 0:
            acc[rec[i]] += acc[i]
    area_km2 = acc * (CELL / 1000) ** 2

    return dict(H=H, W=W, rec=rec, order=order, acc=acc, area=area_km2, x=x, y=y,
                is_land=is_land.ravel())


net = build_synthetic_network()

# ----------------------------------------------------------------------------
# 3. Synthetic 2025 anomaly field (in standard-normal "z" units)
# ----------------------------------------------------------------------------
to_ll = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True)
lon, lat = to_ll.transform(net["x"], net["y"])

blobs = [  # lon, lat, sigma(deg), amplitude
    (31, 49, 8.0, -1.7),   # Ukraine / western Russia: very dry
    (34, 39, 6.0, -2.0),   # Anatolia: record low
    (25, 45, 3.5, -1.0),   # Lower Danube / Romania
    (21, 53, 5.0, -0.6),   # Poland / Baltics
    (13, 49, 4.0, -0.4),   # Central Europe
    (-5, 40, 4.5, +2.2),   # Iberia: wet
    (1, 44, 2.5, +0.6),    # SW France
    (7, 61, 2.8, +1.8),    # western Norway
    (15, 66, 3.0, +0.9),   # northern Scandinavia, mixed
    (-19, 65, 3.0, +1.1),  # Iceland
    (-4, 56.5, 2.0, +0.9), # Scotland
    (9, 46, 2.0, +1.0),    # Alps / Po
    (20, 40, 2.5, +0.6),   # western Balkans
]
z = np.full_like(lon, -0.55)
for lo, la, s, a in blobs:
    d2 = ((lon - lo) * np.cos(np.radians(la))) ** 2 + (lat - la) ** 2
    z += a * np.exp(-d2 / (2 * s ** 2))
smooth = fractal_noise((net["H"], net["W"]), [12, 30], [0.6, 0.8]).ravel()
z += 0.55 * smooth + 0.35 * rng.standard_normal(z.size)

# River flow integrates its basin: a river's anomaly is the area-weighted
# mean of the field over everything upstream of it.
zsum = np.where(net["is_land"], z, 0.0)
for i in net["order"][::-1]:
    r = net["rec"][i]
    if r >= 0:
        zsum[r] += zsum[i]
z_river = np.divide(zsum, net["acc"], out=np.zeros_like(zsum), where=net["acc"] > 0)
z_river = z_river / np.nanstd(z_river[net["area"] >= MIN_AREA_KM2]) * 1.15

# Categories, as defined in the figure caption
CATS = ["Lowest", "Much lower\nthan average", "Lower\nthan average", "Near\naverage",
        "Higher\nthan average", "Much higher\nthan average", "Highest"]
COLORS = ["#af1f29", "#d7644d", "#f3ae8e", "#ffffff", "#55a0aa", "#37728e", "#193449"]
edges = [-np.inf, -1.95, -1.2816, -0.4307, 0.4307, 1.2816, 1.95, np.inf]  # extremes ~ rank 1 of 34 yrs
cat = np.digitize(z_river, edges[1:-1])

# ----------------------------------------------------------------------------
# 4. Trace the network into smooth polylines, split at confluences and
#    wherever the category changes, then Chaikin-smooth away the grid steps
# ----------------------------------------------------------------------------
rec, area = net["rec"], net["area"]
chan = (area >= MIN_AREA_KM2) & net["is_land"]
n_in = np.zeros(rec.size, dtype=int)
up_of = np.full(rec.size, -1)
ci = np.flatnonzero(chan & (rec >= 0))
np.add.at(n_in, rec[ci], 1)
up_of[rec[ci]] = ci
start = chan & ((n_in != 1) | (cat != cat[np.maximum(up_of, 0)]))


def chaikin(p, n=3):
    for _ in range(n):
        q = 0.75 * p[:-1] + 0.25 * p[1:]
        r = 0.25 * p[:-1] + 0.75 * p[1:]
        mid = np.empty((2 * len(q), 2)); mid[0::2] = q; mid[1::2] = r
        p = np.vstack([p[:1], mid, p[-1:]])
    return p


lines, l_cat, l_area = [], [], []
for s0 in np.flatnonzero(start):
    path = [s0]
    cur = s0
    while True:
        nxt = rec[cur]
        if nxt < 0 or not chan[nxt]:
            break
        path.append(nxt)
        if start[nxt]:
            break
        cur = nxt
    if len(path) < 2:
        continue
    pts = np.column_stack([net["x"][path], net["y"][path]])
    lines.append(chaikin(pts))
    l_cat.append(cat[s0])
    l_area.append(area[path[:-1]].mean())

l_cat = np.array(l_cat)
logA = np.log10(np.array(l_area))
width = 0.28 + 1.35 * ((logA - logA.min()) / (logA.max() - logA.min())) ** 1.4

# share of river cells (≈ river length) below average
share_below = (cat[chan] <= 2).mean() * 100

# ----------------------------------------------------------------------------
# 5. Render
# ----------------------------------------------------------------------------
for f in Path.home().joinpath(".fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
FONT = "Barlow Condensed" if any("Barlow" in f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
plt.rcParams["font.family"] = FONT

NAVY, GREY_TXT, LAND = "#2c3e5c", "#7a8290", "#c9c9c9"

fig = plt.figure(figsize=(7.2, 7.2), dpi=300)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.02, 0.075, 0.96, 0.78])
ax.set_xlim(VIEW[0], VIEW[1]); ax.set_ylim(VIEW[2], VIEW[3])
ax.set_aspect("equal"); ax.axis("off")

land.plot(ax=ax, color=LAND, linewidth=0, zorder=1)

# draw small rivers first so main stems sit on top
o = np.argsort(width)
lc = LineCollection([lines[i] for i in o], colors=np.array(COLORS)[l_cat[o]], linewidths=width[o],
                    capstyle="round", joinstyle="round", zorder=2)
ax.add_collection(lc)

# title + subtitle
fig.text(0.5, 0.965, f"Annual river flow was below average in {share_below:.0f}% of European rivers in 2025",
         ha="center", va="center", fontsize=12.5, fontweight="bold", color=NAVY)
fig.text(0.5, 0.94, "Anomalies and extremes in annual average river flow in 2025 (synthetic data)",
         ha="center", va="center", fontsize=9.5, color=GREY_TXT, fontweight="medium")

# legend: a segmented bar with labels underneath
lx0, lw, ly, lh = 0.12, 0.76 / 7, 0.905, 0.011
for k, (lab, col) in enumerate(zip(CATS, COLORS)):
    fig.patches.append(Rectangle((lx0 + k * lw + 0.002, ly), lw - 0.004, lh, transform=fig.transFigure,
                                 facecolor=col, edgecolor="none"))
    fig.text(lx0 + (k + 0.5) * lw, ly - 0.007, lab, ha="center", va="top", fontsize=6.6,
             color="#3d4757", linespacing=1.0)

fig.text(0.5, 0.05, "Data: synthetic network & anomalies • Coastlines: Natural Earth • Reference period: 1992–2020",
         ha="center", va="center", fontsize=7.5, color=GREY_TXT, fontweight="semibold")
fig.text(0.5, 0.028, "Style recreation of Copernicus C3S European State of the Climate, Fig. 11.2 • Made with geopandas + matplotlib",
         ha="center", va="center", fontsize=6.5, color="#a0a6b0")

fig.savefig(OUT, dpi=300, facecolor="white")
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
print(f"lines={len(lines):,}  below-average share={share_below:.1f}%  -> {OUT}")
