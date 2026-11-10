"""
Recreation of the Copernicus/C3S "Contrasting conditions for potential power
generation from renewables" figure (European State of the Climate 2025).

Real data:      Natural Earth countries for the locator globes.
Synthetic data: monthly 2025 anomalies (%) and 10th-90th percentile bands for
                onshore wind, solar PV and run-of-river hydropower, shaped
                after the published panels.

To use real data instead: C3S "Climate and energy indicators for Europe"
(capacity factors / generation potential, monthly, by country). Average the
countries of each region, express 2025 as % anomaly vs 1991-2020 per calendar
month, and take the 10th/90th percentiles of the 1991-2020 anomalies for the band.
"""
import sys
from pathlib import Path

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Circle
from scipy.interpolate import CubicSpline
from shapely.geometry import box

DATA = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("ne")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("renewables_2025_synthetic.png")

for f in Path.home().joinpath(".fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Barlow Condensed"

NAVY, GREY_TXT, TICK, BAND, EDGE = "#2e3a59", "#7a8190", "#a3a3a3", "#eeefef", "#4a4a4a"
STYLE = {  # positive colour, negative colour, y-limits, y-ticks
    "Onshore wind": ("#635799", "#f38846", (-48, 32), [-45, -30, -15, 0, 15, 30]),
    "Solar PV": ("#fcd04d", "#9a8686", (-21, 16), [-20, -15, -10, -5, 0, 5, 10, 15]),
    "Run-of-river hydropower": ("#6ea0cb", "#c6c5c5", (-32, 46), [-30, -20, -10, 0, 10, 20, 30, 40]),
}

# ----------------------------------------------------------------------------
# 1. Synthetic monthly data: anomaly (%), p10, p90 for Jan..Dec 2025
# ----------------------------------------------------------------------------
D = {
    ("NW", "Onshore wind"): dict(
        v=[0, -33, -26, -12, 0, 6, -4, -12, 15, 14, -2, -15],
        lo=[-30, -30, -29, -26, -19, -16, -16, -19, -23, -30, -31, -31],
        hi=[29, 27, 20, 14, 10, 11, 13, 15, 19, 25, 24, 20]),
    ("NW", "Solar PV"): dict(
        v=[-1, 11, 6, 7, 8.5, 2, 7.5, -1, -4.5, -4, 3, 7],
        lo=[-13, -12, -10, -9, -8, -8, -8, -9, -10, -12, -14, -17],
        hi=[14, 12, 11, 9, 9, 8, 8, 8, 8, 10, 12, 15]),
    ("NW", "Run-of-river hydropower"): dict(
        v=[12, 5, 2, 1, -5, -10, -17, -17, 1, -6, -3, -1],
        lo=[-15, -20, -17, -15, -16, -18, -20, -20, -15, -12, -12, -15],
        hi=[15, 17, 16, 15, 13, 8, 6, 8, 10, 26, 18, 18]),
    ("IB", "Onshore wind"): dict(
        v=[30, -44, 24, -8, -28, -31, 8, -12, 8, -15, 16, -18],
        lo=[-30, -46, -36, -30, -25, -21, -21, -25, -28, -31, -36, -40],
        hi=[32, 25, 21, 16, 13, 13, 15, 16, 20, 23, 28, 32]),
    ("IB", "Solar PV"): dict(
        v=[-3, -19, -8, 1, 0, 0, 0.5, 2.5, 2, 1, 1, -17],
        lo=[-14, -20, -14, -10, -8, -6, -6, -6, -8, -10, -12, -17],
        hi=[15, 12, 12, 8, 6, 5, 5, 5, 6, 8, 10, 14]),
    ("IB", "Run-of-river hydropower"): dict(
        v=[-10, 12, 42, 40, 28, 18, 13, 6, -8, -30, -12, 5],
        lo=[-30, -30, -26, -21, -16, -13, -13, -16, -21, -31, -31, -30],
        hi=[35, 41, 36, 31, 26, 21, 16, 15, 16, 16, 21, 26]),
}
REGIONS = {
    "NW": ("Northwestern and\ncentral Europe",
           ["GBR", "IRL", "FRA", "BEL", "NLD", "LUX", "DEU", "CHE", "AUT", "DNK", "CZE"], (6, 50)),
    "IB": ("Iberian Peninsula", ["ESP", "PRT"], (-3, 40)),
}

months = np.arange(12)
tt = np.linspace(0, 11, 600)


def smooth(vals):
    return CubicSpline(months, vals, bc_type="natural")(tt)


# ----------------------------------------------------------------------------
# 2. Figure
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(10.2, 6.6), dpi=300)
fig.patch.set_facecolor("white")
fig.text(0.53, 0.955, "Contrasting conditions for potential power generation from renewables",
         ha="center", fontsize=16, fontweight="bold", color=NAVY)
fig.text(0.53, 0.915, "Monthly anomalies in the potential for renewable power generation for two regions "
         "in Europe (synthetic data)", ha="center", fontsize=11, color=GREY_TXT)

cols = list(STYLE)
px0, pw, gx = 0.205, 0.225, 0.04
rows_y = {"NW": 0.515, "IB": 0.13}
ph = 0.31
axes = {}

for reg, y0 in rows_y.items():
    for c, var in enumerate(cols):
        pos, neg, ylim, yt = STYLE[var]
        d = D[(reg, var)]
        ax = fig.add_axes([px0 + c * (pw + gx), y0, pw, ph])
        axes[(reg, var)] = ax
        v, lo, hi = smooth(d["v"]), smooth(d["lo"]), smooth(d["hi"])
        ax.fill_between(tt, lo, hi, color=BAND, lw=0, zorder=1)
        ax.fill_between(tt, 0, v, where=v >= 0, interpolate=True, color=pos, lw=0, zorder=2)
        ax.fill_between(tt, 0, v, where=v < 0, interpolate=True, color=neg, lw=0, zorder=2)
        ax.plot(tt, v, color=EDGE, lw=0.6, zorder=3)
        ax.set_xlim(-0.05, 11.05)
        ax.set_ylim(*ylim)
        ax.set_yticks(yt)
        ax.set_yticklabels([f"{t}%" if t == yt[-1] else str(t) for t in yt])
        ax.set_xticks([0, 2, 4, 6, 8, 10])
        ax.set_xticklabels(["Jan\n2025", "Mar", "May", "Jul", "Sep", "Nov"], va="top")
        ax.tick_params(length=0, labelsize=8.5, colors=TICK, pad=3)
        for s in ax.spines.values():
            s.set_visible(False)
        if reg == "NW":
            ax.set_title(var, fontsize=13, fontweight="bold", color=NAVY, pad=10)


def callout(ax, xy, text_xy, text, ha="left"):
    """Dot on the curve + elbow leader line to a text label (data coords)."""
    ax.plot(*xy, "o", ms=3.2, color="#222", zorder=5)
    ax.plot([xy[0], xy[0], text_xy[0] - (0.15 if ha == "left" else -0.15)],
            [xy[1], text_xy[1], text_xy[1]], color="#222", lw=0.55, zorder=5)
    ax.text(*text_xy, text, fontsize=8.3, color="#2b2b2b", va="center", ha=ha, zorder=6, linespacing=1.15)


callout(axes[("NW", "Onshore wind")], (1.9, -18), (2.3, -30), "Low wind conditions\nin early 2025")
axes[("NW", "Onshore wind")].text(0.7, 12, "10th–90th\npercentile range", fontsize=8.3, style="italic",
                                  color=GREY_TXT, va="center", linespacing=1.1)
a = axes[("NW", "Solar PV")]
a.plot(4.1, 3.5, "o", ms=3.2, color="#222", zorder=5)
a.plot([4.1, 4.1], [3.5, -3.2], color="#222", lw=0.55, zorder=5)
a.text(4.1, -6.0, "More sunshine than average\nduring spring and summer", fontsize=8.3, color="#2b2b2b",
       ha="center", va="center", linespacing=1.15, zorder=6)
callout(axes[("IB", "Run-of-river hydropower")], (3.6, 13), (4.6, 30),
        "Increased potential due to\nwetter-than-average conditions")

# ----------------------------------------------------------------------------
# 3. Locator globes (orthographic, clipped to a circle)
# ----------------------------------------------------------------------------
countries = gpd.read_file(DATA / "ne_50m_admin_0_countries.shp")
code = "ADM0_A3" if "ADM0_A3" in countries.columns else "ISO_A3"
for reg, (label, isos, (lon0, lat0)) in REGIONS.items():
    proj = f"+proj=ortho +lat_0={lat0} +lon_0={lon0} +datum=WGS84 +units=m"
    window = box(lon0 - 35, lat0 - 25, lon0 + 35, lat0 + 25)
    cc = countries.clip(window).to_crs(proj)
    cc = cc[cc.is_valid & ~cc.is_empty]
    R = 1.45e6                                       # radius of the globe window (m)
    gax = fig.add_axes([0.03, rows_y[reg] + 0.01, 0.15, 0.15 * fig.get_figwidth() / fig.get_figheight()])
    gax.set_xlim(-R, R); gax.set_ylim(-R, R); gax.set_aspect("equal"); gax.axis("off")
    clip = Circle((0, 0), R, transform=gax.transData)
    gax.add_patch(Circle((0, 0), R, fc="white", ec="none", zorder=0))
    sel = cc[code].isin(isos)
    cc[~sel].plot(ax=gax, color="#dce2e7", edgecolor="white", lw=0.3, zorder=1)
    cc[sel].plot(ax=gax, color="#344163", edgecolor="#8a93a8", lw=0.25, zorder=2)
    for art in gax.collections:
        art.set_clip_path(clip)
    gax.add_patch(Circle((0, 0), R, fill=False, ec="#9a9a9a", lw=0.8, zorder=5))
    fig.text(0.105, rows_y[reg] + ph + 0.005, label, ha="center", va="top", fontsize=11, style="italic",
             color=NAVY, linespacing=1.15)

fig.text(0.04, 0.018, "Data: synthetic monthly anomalies • Countries: Natural Earth • Reference period: 1991–2020 • "
         "Style recreation of Copernicus C3S European State of the Climate", fontsize=9, color=GREY_TXT)

fig.savefig(OUT, dpi=300, facecolor="white")
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
print("saved", OUT)
