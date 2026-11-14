# SPDX-License-Identifier: BSD-3-Clause
"""Draw the glide-slope constraint in 3-D: the idea, and our logged landings against the cone.

Run from the repository root::

    python docs/figures/make_glide_slope_figure.py runs/<run_dir> [minimum_slope_angle_deg]

The glide-slope constraint of powered-descent guidance keeps the lander inside an upward cone:
height above the apex >= tan(gamma) * horizontal distance from the apex, with gamma the
minimum slope angle. In the minimum-landing-error problem the apex is the point where the
lander really lands, not the original target. Panel (a) is a sketch of this idea. Panel (b)
plots the trajectories of ``trajectories.parquet`` of a run, each shifted so that its own
landed position is the apex; parts below the cone are drawn in orange. The ZEM/ZEV agent of
milestone 1a has no glide-slope constraint, so it is free to leave the cone.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pyplot  # noqa: E402
import numpy  # noqa: E402
import pandas  # noqa: E402

INK, SECONDARY_INK, MUTED = "#0b0b0b", "#52514e", "#898781"
INSIDE_COLOR, OUTSIDE_COLOR, CONE_COLOR, GROUND_COLOR = "#2a78d6", "#eb6834", "#86b6ef", "#c3c2b7"


def draw_cone(axes, apex, slope_angle: float, top_height: float) -> None:
    """Draw a translucent upward cone with its apex at ``apex`` and a rim at ``top_height``."""
    azimuth = numpy.linspace(0.0, 2.0 * numpy.pi, 60)
    height = numpy.linspace(0.0, top_height, 2)
    azimuth_grid, height_grid = numpy.meshgrid(azimuth, height)
    radius = height_grid / numpy.tan(slope_angle)
    axes.plot_surface(
        apex[0] + radius * numpy.cos(azimuth_grid),
        apex[1] + radius * numpy.sin(azimuth_grid),
        height_grid,
        color=CONE_COLOR,
        alpha=0.22,
        linewidth=0,
        shade=False,
    )
    rim = top_height / numpy.tan(slope_angle)
    axes.plot(apex[0] + rim * numpy.cos(azimuth), apex[1] + rim * numpy.sin(azimuth), top_height, color=INSIDE_COLOR, lw=0.9)
    for angle in (0.25 * numpy.pi, 1.25 * numpy.pi):
        axes.plot([apex[0], apex[0] + rim * numpy.cos(angle)], [apex[1], apex[1] + rim * numpy.sin(angle)], [0, top_height], color=INSIDE_COLOR, lw=0.9)


def draw_ground(axes, half_size: float) -> None:
    """Draw a flat grey ground square."""
    corners = numpy.array([[-half_size, half_size], [-half_size, half_size]])
    grid_x, grid_y = numpy.meshgrid(corners[0], corners[1])
    axes.plot_surface(grid_x, grid_y, numpy.zeros_like(grid_x), color=GROUND_COLOR, alpha=0.25, linewidth=0, shade=False)


def style(axes, half_size: float, top_height: float, title: str) -> None:
    """Set limits, labels, view and a quiet look."""
    axes.set_xlim(-half_size, half_size)
    axes.set_ylim(-half_size, half_size)
    axes.set_zlim(0.0, top_height)
    axes.set_box_aspect((2 * half_size, 2 * half_size, top_height))
    axes.set_xlabel("x (m)", color=SECONDARY_INK, labelpad=2)
    axes.set_ylabel("y (m)", color=SECONDARY_INK, labelpad=2)
    axes.set_zlabel("altitude (m)", color=SECONDARY_INK, labelpad=2)
    axes.tick_params(colors=MUTED, labelsize=7, pad=0)
    for axis in (axes.xaxis, axes.yaxis, axes.zaxis):
        axis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        axis._axinfo["grid"]["color"] = "#e1e0d9"
        axis._axinfo["grid"]["linewidth"] = 0.5
    axes.view_init(elev=18, azim=-58)
    axes.set_title(title, loc="left", fontsize=10, color=INK, pad=0)


def main() -> None:
    """Build the two-panel figure and save it next to this script."""
    run_directory = Path(sys.argv[1])
    slope_angle_degrees = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    slope_angle = numpy.radians(slope_angle_degrees)
    figure = pyplot.figure(figsize=(12.5, 6.0))

    # ---- (a) the idea ----
    concept = figure.add_subplot(1, 2, 1, projection="3d")
    half_size, top_height = 26.0, 14.0
    apex = numpy.array([6.0, 3.0, 0.0])
    target = numpy.array([-4.0, -3.0, 0.0])
    draw_ground(concept, half_size)
    draw_cone(concept, apex, slope_angle, top_height)
    progress = numpy.linspace(0.0, 1.0, 120)
    start = numpy.array([-8.0, -9.0, 13.0])
    path = numpy.stack([apex[k] + (start[k] - apex[k]) * (1.0 - progress) ** power for k, power in enumerate((1.6, 1.4, 1.1))], axis=1)
    concept.plot(path[:, 0], path[:, 1], path[:, 2], color=INK, lw=1.8)
    concept.scatter(*start, color=INK, s=22)
    concept.text(start[0], start[1], start[2] + 0.8, "start", color=INK, fontsize=9, ha="center")
    concept.scatter(*apex, color=INSIDE_COLOR, s=40, depthshade=False)
    concept.text(apex[0] + 3.0, apex[1] + 6.0, 3.2, "landed position\n= apex of the cone", color=INSIDE_COLOR, fontsize=9, va="bottom")
    concept.scatter(*target, color=OUTSIDE_COLOR, s=70, marker="x", linewidths=2.2, depthshade=False)
    concept.text(target[0] - 1.0, target[1], 0.3, "original target", color=OUTSIDE_COLOR, fontsize=9, ha="right", va="bottom")
    concept.plot([target[0], apex[0]], [target[1], apex[1]], [0, 0], color=SECONDARY_INK, lw=1.0, ls=(0, (3, 2)))
    middle = 0.5 * (target + apex)
    concept.text(middle[0], middle[1] - 4.0, 0.0, "landing error\n(minimised)", color=SECONDARY_INK, fontsize=8, ha="center", va="top")
    arc = numpy.linspace(0.0, slope_angle, 20)
    concept.plot(apex[0] + 6.0 * numpy.cos(arc), numpy.full_like(arc, apex[1]), 6.0 * numpy.sin(arc), color=INK, lw=1.0)
    concept.plot([apex[0], apex[0] + 9.0], [apex[1], apex[1]], [0, 0], color=INK, lw=0.8)
    concept.text(apex[0] + 7.4, apex[1], 1.2, r"$\gamma$", color=INK, fontsize=13)
    style(concept, half_size, top_height, rf"(a) Glide-slope cone, minimum slope angle $\gamma$ = {slope_angle_degrees:.0f}°")

    # ---- (b) our landings ----
    landings = figure.add_subplot(1, 2, 2, projection="3d")
    trajectories = pandas.read_parquet(run_directory / "trajectories.parquet")
    tracks, outside_count = [], 0
    for _, track in trajectories.groupby("environment_index"):
        track = track.sort_values("time_s")
        x = (track["position_x_m"] - track["position_x_m"].iloc[-1]).to_numpy()
        y = (track["position_y_m"] - track["position_y_m"].iloc[-1]).to_numpy()
        z = track["altitude_m"].to_numpy()
        # Below 1 m the lander is in its final vertical drop; the angle there has no meaning.
        outside = (z < numpy.tan(slope_angle) * numpy.hypot(x, y)) & (z > 1.0)
        outside_count += int(outside.any())
        tracks.append((x, y, z, outside))
    reach = max(float(numpy.abs(numpy.concatenate([numpy.concatenate((x, y)) for x, y, _, _ in tracks])).max()), 5.0)
    top = float(max(z.max() for _, _, z, _ in tracks)) * 1.02
    half = max(reach * 1.1, 0.5 * top / numpy.tan(slope_angle) * 0.6)
    draw_ground(landings, half)
    draw_cone(landings, numpy.zeros(3), slope_angle, min(top, half * numpy.tan(slope_angle)))
    for x, y, z, outside in tracks:
        landings.plot(x, y, z, color=INSIDE_COLOR, lw=1.0, alpha=0.9)
        masked = numpy.where(outside, z, numpy.nan)
        landings.plot(x, y, masked, color=OUTSIDE_COLOR, lw=2.0)
    landings.scatter(0, 0, 0, color=INSIDE_COLOR, s=40, depthshade=False)
    landings.plot([], [], color=INSIDE_COLOR, lw=1.0, label="inside the cone")
    landings.plot([], [], color=OUTSIDE_COLOR, lw=2.0, label="below the cone (too shallow)")
    landings.legend(frameon=False, fontsize=8.5, labelcolor=SECONDARY_INK, loc="upper right")
    style(
        landings,
        half,
        top,
        f"(b) {len(tracks)} logged ZEM/ZEV landings, each shifted so that it lands at the apex\n"
        f"     {outside_count} of {len(tracks)} leave the {slope_angle_degrees:.0f}° cone (the agent has no glide-slope constraint)",
    )

    figure.subplots_adjust(left=0.0, right=0.98, bottom=0.02, top=0.93, wspace=0.02)
    output = Path(__file__).resolve().parent / "glide_slope_constraint_3d.png"
    figure.savefig(output, dpi=170, facecolor="white", bbox_inches="tight")
    print(output, f"landings outside the cone: {outside_count} of {len(tracks)}")


if __name__ == "__main__":
    main()
