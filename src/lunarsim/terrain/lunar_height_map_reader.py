# SPDX-License-Identifier: BSD-3-Clause
"""Read an LROC NAC digital terrain model (a PDS3 ``.LBL`` label plus its ``.TIF`` data file).

The ``.TIF`` files of these products are uncompressed: after a header, the label points to a
plain block of little-endian 32-bit floats, one row after the other (row 0 is the north edge).
So numpy reads them with a memory map; no GeoTIFF library is needed, and only the part that is
sampled is read from disk.

Local frame: x east, y north, z up, in metres, with the origin at a chosen site (latitude,
longitude). The products use an equirectangular projection on a sphere. Near the site,

    map x = map x of the site + east * cos(projection centre latitude) / cos(site latitude)
    map y = map y of the site + north

(the second line is exact on the sphere; the first is exact along the site's parallel and good
to a few parts per million over a few kilometres).
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy

# Values below this are the products' "no data" marker (-3.4028226550889045e+38).
NO_DATA_BELOW = -1.0e30


class HeightMapError(RuntimeError):
    """The height-map files are missing, unreadable, or have no data where it is needed."""


def parse_planetary_data_system_label(text: str) -> dict[str, str]:
    """Return ``KEY -> value text`` for every ``KEY = value`` line of a PDS3 label.

    Keys inside OBJECT blocks are stored with the object name in front, for example
    ``IMAGE_MAP_PROJECTION.MAP_SCALE``, and also without it if the plain key is not taken yet.
    Values keep their text (quotes and units removed later by :func:`label_number`).
    """
    entries: dict[str, str] = {}
    objects: list[str] = []
    key = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line == "END":
            continue
        match = re.match(r"^([\^A-Z0-9_:]+)\s*=\s*(.*)$", line)
        if not match:
            if key is not None:  # continuation of a multi-line value
                entries[key] += " " + line
            continue
        name, value = match.group(1), match.group(2).strip()
        if name == "OBJECT":
            objects.append(value)
            key = None
            continue
        if name == "END_OBJECT":
            if objects:
                objects.pop()
            key = None
            continue
        key = ".".join(objects + [name]) if objects else name
        entries[key] = value
        entries.setdefault(name, value)
    return entries


def label_number(value: str) -> float:
    """Return the first number in a label value such as ``2.0000000000006 <METERS/PIXEL>``."""
    match = re.search(r"[-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?", value)
    if not match:
        raise HeightMapError(f"No number in label value '{value}'")
    return float(match.group(0))


@dataclass(frozen=True)
class HeightMapLayout:
    """Where the data sit in the file and how pixels map to the projection."""

    data_byte_offset: int
    number_of_lines: int
    number_of_samples: int
    map_scale_m: float
    line_projection_offset: float
    sample_projection_offset: float
    projection_center_latitude_deg: float
    projection_center_longitude_deg: float
    sphere_radius_m: float

    @classmethod
    def from_label(cls, label: dict[str, str]) -> HeightMapLayout:
        """Read the layout from a parsed label; refuse anything this reader does not handle."""
        projection_type = label.get("IMAGE_MAP_PROJECTION.MAP_PROJECTION_TYPE", "").strip('"')
        if projection_type != "EQUIRECTANGULAR":
            raise HeightMapError(f"Only equirectangular products are supported, got '{projection_type}'")
        sample_type = label.get("IMAGE.SAMPLE_TYPE", "")
        if sample_type != "PC_REAL" or int(label_number(label.get("IMAGE.SAMPLE_BITS", "0"))) != 32:
            raise HeightMapError(f"Only 32-bit little-endian floats are supported, got '{sample_type}'")
        pointer = label.get("^IMAGE", "")
        match = re.search(r"(\d+)\s*<BYTES>", pointer)
        if not match:
            raise HeightMapError(f"Cannot read the data pointer '^IMAGE = {pointer}'")
        radius_km = label_number(label["IMAGE_MAP_PROJECTION.A_AXIS_RADIUS"])
        if abs(radius_km - label_number(label["IMAGE_MAP_PROJECTION.C_AXIS_RADIUS"])) > 1e-6:
            raise HeightMapError("Only products on a sphere are supported")
        return cls(
            data_byte_offset=int(match.group(1)) - 1,  # the label counts bytes from 1
            number_of_lines=int(label_number(label["IMAGE.LINES"])),
            number_of_samples=int(label_number(label["IMAGE.LINE_SAMPLES"])),
            map_scale_m=label_number(label["IMAGE_MAP_PROJECTION.MAP_SCALE"]),
            line_projection_offset=label_number(label["IMAGE_MAP_PROJECTION.LINE_PROJECTION_OFFSET"]),
            sample_projection_offset=label_number(label["IMAGE_MAP_PROJECTION.SAMPLE_PROJECTION_OFFSET"]),
            projection_center_latitude_deg=label_number(label["IMAGE_MAP_PROJECTION.CENTER_LATITUDE"]),
            projection_center_longitude_deg=label_number(label["IMAGE_MAP_PROJECTION.CENTER_LONGITUDE"]),
            sphere_radius_m=1000.0 * radius_km,
        )


class LunarHeightMap:
    """One NAC DTM product, sampled in a local east/north/up frame around a site."""

    def __init__(self, label_path: Path | str, data_path: Path | str, site_latitude_deg: float, site_longitude_deg: float):
        """Open the product (memory-mapped) and set up the local frame at the site."""
        label_path, data_path = Path(label_path), Path(data_path)
        for path in (label_path, data_path):
            if not path.is_file():
                raise HeightMapError(
                    f"Height-map file {path} is missing. Download it with 'python scripts/download_height_maps.py' "
                    "(see docs/knowledge/apollo17-height-maps.md)."
                )
        self.layout = HeightMapLayout.from_label(parse_planetary_data_system_label(label_path.read_text(errors="replace")))
        layout = self.layout
        expected_bytes = layout.data_byte_offset + 4 * layout.number_of_lines * layout.number_of_samples
        if data_path.stat().st_size < expected_bytes:
            raise HeightMapError(f"{data_path} is shorter than its label says (incomplete download?)")
        self.data = numpy.memmap(
            data_path,
            dtype="<f4",
            mode="r",
            offset=layout.data_byte_offset,
            shape=(layout.number_of_lines, layout.number_of_samples),
        )
        self.site_latitude_deg = float(site_latitude_deg)
        self.site_longitude_deg = float(site_longitude_deg)
        radius = layout.sphere_radius_m
        center_latitude = math.radians(layout.projection_center_latitude_deg)
        self.site_map_x = radius * math.radians(site_longitude_deg - layout.projection_center_longitude_deg) * math.cos(
            center_latitude
        )
        self.site_map_y = radius * math.radians(site_latitude_deg)
        # map-x metres per local east metre
        self.east_to_map_x = math.cos(center_latitude) / math.cos(math.radians(site_latitude_deg))

    def local_to_pixel(self, east: numpy.ndarray, north: numpy.ndarray) -> tuple[numpy.ndarray, numpy.ndarray]:
        """Return fractional (column, row) pixel-centre coordinates of local points."""
        layout = self.layout
        map_x = self.site_map_x + numpy.asarray(east, dtype=numpy.float64) * self.east_to_map_x
        map_y = self.site_map_y + numpy.asarray(north, dtype=numpy.float64)
        column = map_x / layout.map_scale_m + layout.sample_projection_offset
        row = layout.line_projection_offset - map_y / layout.map_scale_m
        return column, row

    def sample_absolute(self, east: numpy.ndarray, north: numpy.ndarray) -> numpy.ndarray:
        """Return bilinear heights (m, relative to the product's sphere) at local points.

        Raises :class:`HeightMapError` if any point lies outside the data or next to no-data pixels.
        """
        from scipy.ndimage import map_coordinates

        east, north = numpy.asarray(east, dtype=numpy.float64), numpy.asarray(north, dtype=numpy.float64)
        column, row = self.local_to_pixel(east, north)
        first_column, last_column = int(numpy.floor(column.min())) - 1, int(numpy.ceil(column.max())) + 2
        first_row, last_row = int(numpy.floor(row.min())) - 1, int(numpy.ceil(row.max())) + 2
        layout = self.layout
        if first_column < 0 or first_row < 0 or last_column > layout.number_of_samples or last_row > layout.number_of_lines:
            raise HeightMapError("The requested area reaches outside the height map; choose a smaller area or another site")
        window = numpy.array(self.data[first_row:last_row, first_column:last_column], dtype=numpy.float64)
        window[window < NO_DATA_BELOW] = numpy.nan
        heights = map_coordinates(window, [row - first_row, column - first_column], order=1, mode="nearest")
        if not numpy.all(numpy.isfinite(heights)):
            missing = float(numpy.mean(~numpy.isfinite(heights)))
            raise HeightMapError(f"{missing:.1%} of the requested points fall on no-data pixels of the height map")
        return heights

    def sample_relative(self, east: numpy.ndarray, north: numpy.ndarray) -> numpy.ndarray:
        """Return heights relative to the height at the site itself (the site is at z = 0)."""
        site_height = float(self.sample_absolute(numpy.zeros(1), numpy.zeros(1))[0])
        return self.sample_absolute(east, north) - site_height
