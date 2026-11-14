---
id: apollo17-height-maps
title: Apollo 17 height maps (LROC NAC DTM)
type: research
status: current
updated: 2026-11-15
summary: The two NASA height maps of the Apollo 17 site, their file layout, projection and local-frame formula, and where they live (not in git).
---

# Apollo 17 height maps (LROC NAC DTM)

Public products of the LRO Camera team (Arizona State University), data set
`LRO-L-LROC-5-RDR-V1.0`, volume `LROLRC_2001`. Each product is a `.TIF` with a PDS3 `.LBL` label.
**Not committed to git** (user rule). Location `assets/terrain/apollo17/` (git-ignored); get
them with `python scripts/download_height_maps.py` (`--with-5m` adds the 5 m product).
Confirmed download addresses (2026-11-15, sizes match the local copies):

- https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/APOLLO17_2M/NAC_DTM_APOLLO17_2M.TIF (and `.LBL`)
- https://pds.lroc.im-ldi.com/data/LRO-L-LROC-5-RDR-V1.0/LROLRC_2001/DATA/SDP/NAC_DTM/APOLLO17/NAC_DTM_APOLLO17.TIF (and `.LBL`)

On the RTX 5060 machine the files in `assets/terrain/apollo17/` are hard links to the copies in
`~/Desktop/MEO/apollo_lander_modeling/models/`.

| Product | Pixel | Lines × samples | Raw data starts at byte | Size |
| --- | --- | --- | --- | --- |
| `NAC_DTM_APOLLO17_2M` (v1.9) | 2 m | 19,760 × 12,049 | 158,751 | 952 MB |
| `NAC_DTM_APOLLO17` | 5 m | 11,600 × 9,985 | 93,471 | 463 MB |

## File format (checked 2026-11-14)

- Uncompressed little-endian float32, one band, row 0 = north. A numpy `memmap` at the offset
  above returns exactly what `rasterio` returns, so **no extra package is needed**.
  (The label's `^IMAGE` byte is 1-based: label value minus 1.)
- No-data value −3.40282266e38 (treat anything below −1e6 as no data).
- 2 m map: 71 % of its box has data; 5 m map: 85 %. They agree to about 0.09 m rms where they overlap.

## Projection and local frame

- Equirectangular, centre latitude 20°, centre longitude 180°, sphere radius 1,737.4 km.
- Map coordinates: x = R·(lon − 180°)·cos 20°, y = R·lat (radians). GeoTIFF bounds of the 2 m
  map: left −4,265,914 m, top 629,952 m.
- Apollo 17 LM position: **20.19091° N, 30.77168° E** (Haase et al. 2019, Earth and Space
  Science 6, doi:10.1029/2018EA000408, Table 3). The sibling folder used 20.1908, 30.7717.
- Local east = (x − x_site)·cos(lat_site)/cos 20°, north = y − y_site; z = height minus the 2 m
  map's value at the touchdown point.

- used by: [[measured-terrain-surface]]
- first worked with in: [[apollo-lander-modeling-folder]]
