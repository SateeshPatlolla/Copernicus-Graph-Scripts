# Climate figure recreations (Python)

Recreations of eight figures from the Copernicus C3S *European State of the Climate 2025* report, built with
**geopandas + matplotlib** to show how much of their editorial look can be reproduced in code.

Coastlines and country shapes are real (Natural Earth). The climate data is **synthetic**: generated to
resemble the published patterns, not the real observations. The lake chart uses values read by eye off the
published chart, and the Greenland map copies its discharge numbers from the original. Each script's
docstring explains how to swap in the real datasets.

| Figure | Script | Output | Techniques |
|---|---|---|---|
| River flow anomalies | `river_flow_map.py` | `river_flow_2025_synthetic.png` | Synthetic terrain → priority-flood drainage network, basin-averaged anomalies, Chaikin-smoothed polylines, EPSG:3035 |
| Marine heatwaves & Posidonia | `mhw_posidonia_map.py` | `mhw_posidonia_2025_synthetic.png` | Daily SST simulation, Hobday et al. (2018) heatwave categories, circular magnifier inset, mixed bold/italic text |
| Sea surface temperature | `sst_europe_map.py` | `sst_europe_2025_synthetic.png` | Anomaly bar charts + LAEA map, sea-ice edge contour, dashed region boundary |
| Monthly temperature | `monthly_temp_maps.py` | `monthly_temp_2025_synthetic.png` | 3×4 small multiples, hatched no-data block, shared legend |
| Renewable energy potential | `renewables_anomalies.py` | `renewables_2025_synthetic.png` | Spline area charts with percentile bands, orthographic locator globes, callouts |
| Precipitation | `precip_europe_map.py` | `precip_europe_2025_synthetic.png` | ERA5-style 0.25° grid projected cell by cell, curved region boxes, projection fitted to landmarks |
| Lake temperatures | `lakes_warming_chart.py` | `lakes_warming_2025.png` | Monotone (PCHIP) interpolation like D3 `curveMonotoneX`, trend lines, colour-keyed subtitle |
| Greenland ice discharge | `greenland_ice_discharge.py` | `greenland_ice_discharge_synthetic.png` | Ice-flow model routed to outlet glaciers, log velocity scale, model-derived drainage basins, EPSG:3413 |

Each figure is saved as a 300 dpi PNG and an SVG in `output/`.

## Run

```bash
pip install -r requirements.txt
./fetch_data.sh                      # Natural Earth layers into ./ne, Barlow Condensed font into ~/.fonts
python scripts/river_flow_map.py ne output/river_flow_2025_synthetic.png
python scripts/lakes_warming_chart.py output/lakes_warming_2025.png     # needs no map data
```

Every map script takes `<natural-earth-dir> <output.png>`; each run takes a few seconds.

## Credits

Original figures: Copernicus Climate Change Service (C3S) / ECMWF, *European State of the Climate 2025*.
These are independent style recreations with synthetic data and are not affiliated with C3S or ECMWF.
Coastlines: [Natural Earth](https://www.naturalearthdata.com/) (public domain). Font: Barlow Condensed (SIL OFL).
