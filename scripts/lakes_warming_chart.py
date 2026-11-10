"""
Recreation of the Copernicus/C3S "European lakes are warming faster than the
global average" chart (European State of the Climate 2025).

Data: annual values digitized by eye from the published chart (approximate,
not the real series). For real data, use the ESA CCI Lakes / C3S lake surface
water temperature product: average July-September LSWT per lake, take anomalies
vs 1995-2020 per lake, then average across the 265 European / 1912 global lakes.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from scipy.interpolate import PchipInterpolator  # same idea as D3 curveMonotoneX: no overshoot

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("lakes_warming_2025.png")

for f in Path.home().joinpath(".fonts").glob("*.ttf"):
    font_manager.fontManager.addfont(str(f))
plt.rcParams["font.family"] = "Barlow Condensed"

NAVY, GREY_TXT, TICK = "#2e3a59", "#7a8190", "#3d4757"
EU, GLOBE, GLOBE_TXT = "#004c78", "#e8a800", "#a46a00"
FILL, BAND, WARM_TXT = "#edbbb3", "#fbf3f4", "#d86953"

years = np.arange(1995, 2026)
europe = np.array([-0.87, -1.02, -0.21, -1.24, -0.13, -0.57, 0.17, 0.27, 0.35, -0.23, 0.10, 0.38,
                   -0.30, -0.64, -0.18, 0.50, 0.35, -0.22, 0.18, 0.53, 0.20, 0.33, -0.61, 0.73,
                   -0.20, 0.07, 0.08, 0.39, 0.66, 0.75, 0.33])
globe = np.array([-0.57, -0.53, -0.40, -0.02, -0.35, -0.20, 0.21, -0.04, 0.17, -0.30, 0.07, 0.17,
                  -0.12, -0.23, -0.12, 0.17, 0.12, 0.25, 0.21, 0.10, 0.13, 0.45, 0.10, -0.01,
                  -0.09, 0.12, 0.03, 0.34, 0.48, 0.50, 0.35])

tt = np.linspace(years[0], years[-1], 1500)
eu_s = PchipInterpolator(years, europe)(tt)
gl_s = PchipInterpolator(years, globe)(tt)
eu_fit = np.polyfit(years, europe, 1)
gl_fit = np.polyfit(years, globe, 1)

fig = plt.figure(figsize=(9.6, 4.6), dpi=300)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0.065, 0.14, 0.835, 0.60])


def rich_line(y, parts, size):
    """Centered line of text made of differently styled pieces."""
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    arts = [fig.text(0, y, t, fontsize=size, va="center", **kw) for t, kw in parts]
    widths = [a.get_window_extent(rend).width for a in arts]
    x = fig.bbox.width / 2 - sum(widths) / 2
    for a, w in zip(arts, widths):
        a.set_x(x / fig.bbox.width)
        x += w


fig.text(0.5, 0.925, "European lakes are warming faster than the global average", ha="center",
         va="center", fontsize=16, fontweight="bold", color=NAVY)
fig.text(0.5, 0.868, "Warm season (July to September) lake surface water temperature anomalies for",
         ha="center", va="center", fontsize=11, color=GREY_TXT)
rich_line(0.828, [("265 lakes in ", dict(color=GREY_TXT)), ("Europe", dict(color=EU, fontweight="bold")),
                  (" and 1912 lakes ", dict(color=GREY_TXT)), ("globally", dict(color=GLOBE, fontweight="bold"))], 11)

# background: "warmer than average" band, gridlines, zero line
ax.axhspan(0, 1.0, color=BAND, zorder=0, lw=0)
for yv in [-1.0, -0.5, 0.5]:
    ax.axhline(yv, color="#efefef", lw=0.8, zorder=0)
ax.axhline(0, color="#aaaaaa", lw=0.8, zorder=1)

# Europe above zero filled
ax.fill_between(tt, 0, eu_s, where=eu_s > 0, interpolate=True, color=FILL, lw=0, zorder=2)

# trends
ax.plot(years, np.polyval(eu_fit, years), color=EU, lw=0.8, ls=(0, (4, 2.5)), zorder=3)
ax.plot(years, np.polyval(gl_fit, years), color=GLOBE, lw=0.8, ls=(0, (4, 2.5)), zorder=3)

# series
ax.plot(tt, gl_s, color=GLOBE, lw=1.1, zorder=4)
ax.plot(tt, eu_s, color=EU, lw=1.8, zorder=5)
ax.plot(years[-1], europe[-1], "o", ms=5.5, color=EU, zorder=6)

# end labels and annotation
ax.text(2025.5, europe[-1] + 0.075, "Europe", color=EU, fontsize=11, va="center", fontweight="medium")
ax.text(2025.5, globe[-1] - 0.10, "Globe", color=GLOBE_TXT, fontsize=11, va="center", fontweight="medium")
ax.text(1994.6, 0.93, "Warmer than\n1995–2020 average", color=WARM_TXT, fontsize=10, style="italic",
        va="top", linespacing=1.2)

ax.set_xlim(1994.5, 2025.4)
ax.set_ylim(-1.38, 1.0)
ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
ax.set_yticklabels(["-1.0", "-0.5", "0.0", "0.5", "1.0°C"])
ax.set_xticks([1995, 2000, 2005, 2010, 2015, 2020, 2025])
ax.tick_params(length=0, labelsize=11, colors=TICK, pad=6)
ax.yaxis.set_tick_params(pad=4)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_clip_on(False)

trend_eu, trend_gl = eu_fit[0] * 10, gl_fit[0] * 10
fig.text(0.04, 0.035, f"Data: values digitized approximately from the published C3S chart • Reference period: "
         f"1995–2020 • Trend: Europe {trend_eu:+.2f} °C/decade, globe {trend_gl:+.2f} °C/decade",
         fontsize=9.5, color=GREY_TXT)

fig.savefig(OUT, dpi=300, facecolor="white")
fig.savefig(OUT.with_suffix(".svg"), facecolor="white")
print(f"trend Europe {trend_eu:.3f} °C/dec, globe {trend_gl:.3f} °C/dec")
