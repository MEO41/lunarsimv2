# SPDX-License-Identifier: BSD-3-Clause
"""Side-view panel of the L2 plume grains for the replay video (``scripts/render_replay.py``).

The 3D render shows the grain parcels as dots much larger than the grains (4-512 microns cannot
be seen at the camera distance). This panel shows the same parcels at their true positions:
distance from the lander axis against height above the ground, coloured by size bin, in two
views:

- near field, 0-60 m out and 0-12 m up: the cloud grid in which the optical depth is counted
  (dashed box) and the sensor line of sight from the nozzle at the set depression angle;
- far field, 0-700 m out (the grain domain) and 0-40 m up.

Below the views, the readings of both plume levels at that moment, next to the Apollo
references used by ``scripts/compare_plume_levels.py``: eroded mass (Apollo 11, 47-74 kg from
the 36-57 L of Mason 1970 as cited in Metzger, Lane and Immer 2008), lofted particle density
(1e8-1e13 per m^3, Immer, Metzger, Hintze et al. 2008).
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy  # noqa: E402

BACKGROUND = "#0b0d12"
PANEL = "#11141b"
TEXT = "#e8e8e8"
MUTED = "#9aa3ad"
GROUND = "#6b625a"
SIGHT = "#5ab4ff"
LANDER = "#ffffff"
APOLLO_11_ERODED_MASS_KG = (46.8, 74.1)
APOLLO_LOFTED_DENSITY_PER_M3 = (1e8, 1e13)
NEAR_LIMITS_M = (60.0, 12.0)
FAR_LIMITS_M = (700.0, 40.0)


def bin_colors(number_of_bins: int) -> numpy.ndarray:
    """Return one RGB colour per size bin: one hue, pale for the finest grains, deep for the coarsest."""
    ramp = plt.get_cmap("Oranges")
    return numpy.array([ramp(0.25 + 0.7 * index / max(number_of_bins - 1, 1))[:3] for index in range(number_of_bins)])


def micrometres(diameter_m: float) -> str:
    """Return a diameter as a short micrometre label."""
    value = diameter_m * 1e6
    return f"{value:.0f} um" if value >= 10 else f"{value:.1f} um"


class GrainSidePanel:
    """A reusable matplotlib figure; ``draw`` updates it for one frame and returns the RGB pixels."""

    def __init__(self, replay, width_px: int = 640, height_px: int = 720) -> None:
        """Build the two views, the legend and the text block from the replay's grain metadata."""
        self.diameters = numpy.asarray(replay["grain_bin_diameters_m"])
        self.colors = bin_colors(len(self.diameters))
        self.capacity = int(replay["grain_capacity"])
        self.depression = numpy.radians(float(replay["grain_line_of_sight_depression_deg"]))
        grid_radius, grid_height = (float(value) for value in replay["grain_cloud_grid_m"])
        dots_per_inch = 100
        self.figure = plt.figure(figsize=(width_px / dots_per_inch, height_px / dots_per_inch), dpi=dots_per_inch, facecolor=BACKGROUND)
        self.figure.text(0.04, 0.975, "Grain parcels at their true positions (side view)", color=TEXT, fontsize=11, weight="bold", va="top")
        self.figure.text(0.04, 0.945, "distance from the lander axis vs height above the ground; all azimuths folded together",
                         color=MUTED, fontsize=7.5, va="top")
        self.near = self.figure.add_axes((0.11, 0.62, 0.85, 0.28))
        self.far = self.figure.add_axes((0.11, 0.30, 0.85, 0.22))
        self.scatters, self.landers = [], []
        for axes, (radius_limit, height_limit), title in (
            (self.near, NEAR_LIMITS_M, "near field: optical depth is counted in the dashed box"),
            (self.far, FAR_LIMITS_M, "far field: the grain domain (700 m)"),
        ):
            axes.set_facecolor(PANEL)
            axes.set_xlim(0.0, radius_limit)
            axes.set_ylim(0.0, height_limit)
            axes.set_xlabel("distance from the lander axis (m)", color=TEXT, fontsize=8)
            axes.set_ylabel("height (m)", color=TEXT, fontsize=8)
            axes.set_title(title, color=TEXT, fontsize=8.5, loc="left")
            axes.tick_params(colors=MUTED, labelsize=7)
            for spine in axes.spines.values():
                spine.set_color("#3a3f48")
            axes.axhline(0.0, color=GROUND, linewidth=2.0)
            self.scatters.append(axes.scatter([], [], s=3.0, linewidths=0.0))
            self.landers.append(axes.plot([], [], marker="v", color=LANDER, markersize=7, linestyle="none")[0])
        self.near.add_patch(plt.Rectangle((0.0, 0.0), grid_radius, grid_height, fill=False, edgecolor=MUTED, linestyle="--", linewidth=0.9))
        (self.sight,) = self.near.plot([], [], color=SIGHT, linewidth=1.3)
        self.near.text(0.99, 0.95, f"blue: sensor line of sight, {numpy.degrees(self.depression):.0f} deg down",
                       transform=self.near.transAxes, color=SIGHT, fontsize=7, ha="right", va="top")
        # Size legend: one swatch per bin, finest first.
        legend = self.figure.add_axes((0.11, 0.545, 0.85, 0.03))
        legend.set_axis_off()
        legend.set_xlim(0.0, len(self.diameters))
        legend.set_ylim(0.0, 1.0)
        for index, diameter in enumerate(self.diameters):
            legend.scatter([index + 0.12], [0.5], s=30, color=self.colors[index])
            legend.text(index + 0.25, 0.5, micrometres(float(diameter)), color=TEXT, fontsize=7, va="center")
        self.figure.text(0.04, 0.585, "grain size (bin middle):", color=MUTED, fontsize=7.5)
        self.text = self.figure.text(0.04, 0.215, "", color=TEXT, fontsize=8.2, va="top", family="DejaVu Sans Mono", linespacing=1.55)

    def draw(self, offsets: numpy.ndarray, bins: numpy.ndarray, readings: dict[str, float]) -> numpy.ndarray:
        """Update the views for one frame and return its pixels (H, W, 3) uint8.

        ``offsets`` (P, 4): parcel offset x, y, z from the lander body and height above ground (NaN
        rows are dead parcels); ``bins`` (P,) size bin; ``readings``: the frame's GRAIN_COLUMNS
        values plus ``time``.
        """
        alive = numpy.isfinite(offsets[:, 0])
        radius = numpy.hypot(offsets[alive, 0], offsets[alive, 1])
        height = offsets[alive, 3]
        colors = self.colors[bins[alive].astype(int)]
        points = numpy.column_stack((radius, height))
        for scatter in self.scatters:
            scatter.set_offsets(points if len(points) else numpy.empty((0, 2)))
            scatter.set_facecolor(colors if len(points) else "none")
        nozzle_height = readings["nozzle_height"]
        for marker in self.landers:
            marker.set_data([0.0], [nozzle_height])
        reach = nozzle_height / numpy.tan(self.depression)
        self.sight.set_data([0.0, reach], [nozzle_height, 0.0])
        optical_depth = readings["grain_optical_depth"]
        through = 100.0 * numpy.exp(-optical_depth)
        low, high = APOLLO_11_ERODED_MASS_KG
        lofted_low, lofted_high = APOLLO_LOFTED_DENSITY_PER_M3
        self.text.set_text(
            f"t = {readings['time']:5.1f} s     nozzle {nozzle_height:5.1f} m above ground\n"
            f"Eroded soil     grains {readings['grain_eroded_mass']:5.1f} kg   L1 {readings['l1_eroded_mass']:5.1f} kg\n"
            f"                Apollo 11 whole descent: {low:.0f}-{high:.0f} kg\n"
            f"Sensor sight    grains: optical depth {optical_depth:.1e}\n"
            f"                -> {through:6.2f} % of light gets through\n"
            f"                L1 dust density {readings['l1_dust_density']:.2f} (blinded at 0.50)\n"
            f"Densest cell    grains {readings['grain_peak_number_density']:.1e} /m^3\n"
            f"                Apollo videos: {lofted_low:.0e}-{lofted_high:.0e} /m^3\n"
            f"Live parcels    {int(readings['grain_particles']):4d} of {self.capacity} (each = many grains)"
        )
        self.figure.canvas.draw()
        return numpy.asarray(self.figure.canvas.buffer_rgba())[..., :3].copy()

    def close(self) -> None:
        """Close the figure."""
        plt.close(self.figure)
