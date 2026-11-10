#!/usr/bin/env bash
# Downloads the Natural Earth layers and the Barlow Condensed font the scripts use.
set -euo pipefail
mkdir -p ne ~/.fonts
base=https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master
for e in shp shx dbf prj; do
  curl -sSfL -o ne/ne_50m_land.$e                 $base/50m_physical/ne_50m_land.$e
  curl -sSfL -o ne/ne_10m_land.$e                 $base/10m_physical/ne_10m_land.$e
  curl -sSfL -o ne/ne_50m_admin_0_countries.$e    $base/50m_cultural/ne_50m_admin_0_countries.$e
done
for w in Regular Medium SemiBold Bold Italic MediumItalic BoldItalic; do
  curl -sSfL -o ~/.fonts/BarlowCondensed-$w.ttf \
    https://raw.githubusercontent.com/google/fonts/main/ofl/barlowcondensed/BarlowCondensed-$w.ttf
done
rm -rf ~/.cache/matplotlib
echo "Done. Run e.g.: python scripts/river_flow_map.py ne output/river_flow_2025_synthetic.png"
