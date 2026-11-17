# SPDX-License-Identifier: BSD-3-Clause
"""Frames of the S7 mission video before the PhysX phase: descent orbit, braking and approach.

The orbit phases fly 2,000 km of orbit and 480 km of powered descent, far outside the 2 km
terrain patch that the 3D renderer draws, so they are drawn as plots (matplotlib, 1280 x 720):

- descent orbit: the Moon in the orbit plane (Moon-fixed frame), the lander's path drawn as it
  is flown, burns in orange, the site, and a telemetry panel;
- braking and approach: altitude over the range to the site along the surface (full profile and
  the last 9 km), Apollo 11's high and low gates ([D-6846] p. 10), a lander symbol with its
  thrust axis and a flame that follows the throttle, and the telemetry panel.

Every frame shows its speed-up over real time. The trajectories come from
``scripts/run_mission.py record=<lander>`` (``trajectory_lander<i>.npz`` in each phase directory).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy  # noqa: E402

FRAMES_PER_SECOND = 25
WIDTH_INCHES, HEIGHT_INCHES, DOTS_PER_INCH = 12.8, 7.2, 100
BACKGROUND = "#0b0d12"
PANEL = "#11141b"
TEXT = "#e8e8e8"
TRACK = "#5ab4ff"
BURN = "#ff8c3a"
GATE = "#7fd67f"
# [D-6846] p. 10: high gate at 7,000 ft and 4.5 n.mi. from the site; low gate at 500 ft and 2,000 ft.
HIGH_GATE = (8334.0, 2133.6)
LOW_GATE = (609.6, 152.4)


def load_phase(path: Path) -> dict[str, numpy.ndarray]:
    """Return one phase's recorded columns as a dictionary of arrays."""
    data = numpy.load(path)
    rows, names = data["rows"], [str(name) for name in data["columns"]]
    return {name: rows[:, index] for index, name in enumerate(names)}


def surface_range(phase: dict[str, numpy.ndarray], site_direction: numpy.ndarray, site_radius: float) -> numpy.ndarray:
    """Return the distance to the site along the Moon's surface (m), from the Moon-fixed positions."""
    position = numpy.stack((phase["mcmf_x"], phase["mcmf_y"], phase["mcmf_z"]), axis=1)
    cosine = position @ site_direction / numpy.linalg.norm(position, axis=1)
    return site_radius * numpy.arccos(numpy.clip(cosine, -1.0, 1.0))


def sample_indices(times: numpy.ndarray, speed_up: float) -> numpy.ndarray:
    """Return the row index for each video frame at ``speed_up`` times real time (the last row included)."""
    frame_times = numpy.append(numpy.arange(times[0], times[-1], speed_up / FRAMES_PER_SECOND), times[-1])
    return numpy.clip(numpy.searchsorted(times, frame_times), 0, len(times) - 1)


def row_at(phase: dict[str, numpy.ndarray], index: int) -> dict[str, float]:
    """Return one sample of every column."""
    return {name: float(values[index]) for name, values in phase.items()}


def new_figure():
    """Return a dark 1280 x 720 figure."""
    return plt.figure(figsize=(WIDTH_INCHES, HEIGHT_INCHES), dpi=DOTS_PER_INCH, facecolor=BACKGROUND)


def save_frame(figure, directory: Path, number: int) -> int:
    """Save and close the figure as frame ``number``; return the next number."""
    figure.savefig(directory / f"frame_{number:06d}.png", facecolor=BACKGROUND)
    plt.close(figure)
    return number + 1


def telemetry_lines(phase_name: str, row: dict[str, float], speed_up: float, dry_mass: float) -> list[str]:
    """Return the telemetry panel's lines for one sample (time is the mission time from the start)."""
    horizontal = float(numpy.hypot(row["site_velocity_x"], row["site_velocity_y"]))
    minutes, seconds = divmod(row["time"], 60.0)
    altitude = f"{row['altitude'] / 1000:8.2f} km" if row["altitude"] > 5000.0 else f"{row['altitude']:8.0f} m"
    return [
        phase_name,
        f"mission time  {int(minutes):3d}:{seconds:04.1f}",
        f"altitude      {altitude}",
        f"horizontal    {horizontal:8.1f} m/s",
        f"vertical      {row['site_velocity_z']:+8.1f} m/s",
        f"throttle      {100 * row['throttle']:8.1f} %",
        f"mass          {row['mass']:8.0f} kg",
        f"propellant    {row['mass'] - dry_mass:8.0f} kg",
        f"video speed   x{speed_up:g}",
    ]


def draw_panel(figure, lines: list[str]) -> None:
    """Write the telemetry panel on the right side of the figure."""
    figure.text(0.765, 0.86, lines[0], color=TRACK, fontsize=15, fontweight="bold", family="monospace")
    for index, line in enumerate(lines[1:]):
        figure.text(0.765, 0.80 - 0.045 * index, line, color=TEXT, fontsize=11.5, family="monospace")


def title_card(directory: Path, number: int, lines: list[str], seconds: float) -> int:
    """Write a still title card shown for ``seconds``; return the next frame number."""
    figure = new_figure()
    figure.text(0.5, 0.60, lines[0], color=TEXT, fontsize=26, ha="center", fontweight="bold")
    for index, line in enumerate(lines[1:]):
        figure.text(0.5, 0.50 - 0.05 * index, line, color=TEXT, fontsize=14, ha="center")
    first = directory / f"frame_{number:06d}.png"
    save_frame(figure, directory, number)
    count = int(seconds * FRAMES_PER_SECOND)
    for offset in range(1, count):
        (directory / f"frame_{number + offset:06d}.png").symlink_to(first.name)
    return number + count


def orbit_frames(directory: Path, number: int, phase: dict[str, numpy.ndarray], site_direction: numpy.ndarray, site_radius: float, speed_up: float, dry_mass: float) -> int:
    """Draw the descent orbit seen from the orbit plane's normal; return the next frame number."""
    position = numpy.stack((phase["mcmf_x"], phase["mcmf_y"], phase["mcmf_z"]), axis=1) / 1000.0
    normal = numpy.cross(position[0], position[len(position) // 4])
    normal /= numpy.linalg.norm(normal)
    first_axis = site_direction - (site_direction @ normal) * normal
    first_axis /= numpy.linalg.norm(first_axis)
    second_axis = numpy.cross(normal, first_axis)
    planar = numpy.stack((position @ first_axis, position @ second_axis), axis=1)
    site_planar = numpy.array([site_radius / 1000.0, 0.0])
    burning = phase["throttle"] > 0.0
    limit = 1.12 * float(numpy.abs(planar).max())
    for index in sample_indices(phase["time"], speed_up):
        figure = new_figure()
        axes = figure.add_axes([0.03, 0.04, 0.70, 0.84], facecolor=BACKGROUND)
        axes.add_patch(plt.Circle((0.0, 0.0), site_radius / 1000.0, color="#8a8a86"))
        axes.plot(planar[:, 0], planar[:, 1], color=TRACK, alpha=0.18, linewidth=1.0)
        axes.plot(planar[: index + 1, 0], planar[: index + 1, 1], color=TRACK, linewidth=2.0)
        burned = numpy.flatnonzero(burning[: index + 1])
        if burned.size:
            axes.plot(planar[burned, 0], planar[burned, 1], ".", color=BURN, markersize=4)
            axes.annotate("DOI burn", planar[burned[0]], xytext=(10, -18), textcoords="offset points", color=BURN, fontsize=11)
        axes.plot(*site_planar, marker="*", color=GATE, markersize=16)
        axes.annotate("landing site (Apollo 17)", site_planar, xytext=(-60, 14), textcoords="offset points", color=GATE, fontsize=11)
        axes.plot(*planar[index], marker="o", color="white", markersize=8)
        if index == len(planar) - 1:
            axes.annotate("PDI: the powered descent starts", planar[-1], xytext=(-80, 22), textcoords="offset points", color=BURN, fontsize=11)
        axes.set_xlim(-limit, limit)
        axes.set_ylim(-limit, limit)
        axes.set_aspect("equal")
        axes.axis("off")
        figure.text(0.03, 0.92, "1  Descent orbit: DOI burn, then half an orbit down to the 15 km perilune", color=TEXT, fontsize=14)
        draw_panel(figure, telemetry_lines("DESCENT ORBIT", row_at(phase, index), speed_up, dry_mass))
        number = save_frame(figure, directory, number)
    return number


def thrust_axis_on_screen(row: dict[str, float]) -> numpy.ndarray:
    """Return the thrust axis (body +z) as (forward, up): forward is the flight direction, to the right on screen."""
    w, x, y, z = row["quaternion_w"], row["quaternion_x"], row["quaternion_y"], row["quaternion_z"]
    body_z = numpy.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])
    horizontal = numpy.array([row["site_velocity_x"], row["site_velocity_y"]])
    speed = float(numpy.linalg.norm(horizontal))
    forward = horizontal / speed if speed > 0.5 else numpy.array([-1.0, 0.0])  # S7 flies west (-x)
    return numpy.array([body_z[0] * forward[0] + body_z[1] * forward[1], body_z[2]])


def draw_lander(axes, centre: numpy.ndarray, row: dict[str, float], height: float, x_per_y: float) -> None:
    """Draw the lander symbol: a body along the thrust axis and a flame whose length follows the throttle.

    ``height`` is the symbol size in data y units; ``x_per_y`` converts a screen-equal length to data
    x units. The x axis is inverted (range falls to the right), so a step right lowers x.
    """
    axis = thrust_axis_on_screen(row)
    axis /= max(float(numpy.linalg.norm(axis)), 1e-9)
    side = numpy.array([-axis[1], axis[0]])

    def point(offset: numpy.ndarray) -> numpy.ndarray:
        """Turn a screen offset (in symbol sizes) into data coordinates."""
        return centre + numpy.array([-offset[0] * height * x_per_y, offset[1] * height])

    body = [point(0.5 * axis + 0.28 * side), point(0.5 * axis - 0.28 * side), point(-0.5 * axis - 0.28 * side), point(-0.5 * axis + 0.28 * side)]
    axes.add_patch(plt.Polygon(body, closed=True, color="#d9d2b0", zorder=5))
    if row["throttle"] > 0.0:
        flame = [point(-0.5 * axis + 0.18 * side), point(-0.5 * axis - 0.18 * side), point(-(0.5 + 1.4 * row["throttle"]) * axis)]
        axes.add_patch(plt.Polygon(flame, closed=True, color=BURN, alpha=0.9, zorder=4))


def descent_frames(directory: Path, number: int, phases: list[tuple[str, dict, float]], site_direction: numpy.ndarray, site_radius: float, dry_mass: float) -> int:
    """Draw braking and approach as altitude over range (two panels); return the next frame number."""
    ranges = [surface_range(phase, site_direction, site_radius) for _, phase, _ in phases]
    altitudes = [phase["altitude"] for _, phase, _ in phases]
    full_range, full_altitude = numpy.concatenate(ranges), numpy.concatenate(altitudes)
    panels = (
        ([0.05, 0.12, 0.33, 0.72], 1e-3, "km", (float(full_range.max()) * 1e-3 * 1.03, 0.0), (0.0, float(full_altitude.max()) * 1e-3 * 1.1), "full powered descent"),
        ([0.43, 0.12, 0.30, 0.72], 1.0, "m", (9000.0, 0.0), (0.0, 2600.0), "last 9 km"),
    )
    for phase_number, (name, phase, speed_up) in enumerate(phases):
        for index in sample_indices(phase["time"], speed_up):
            row = row_at(phase, index)
            flown_range = numpy.concatenate(ranges[:phase_number] + [ranges[phase_number][: index + 1]])
            flown_altitude = numpy.concatenate(altitudes[:phase_number] + [altitudes[phase_number][: index + 1]])
            figure = new_figure()
            for box, scale, unit, x_limit, y_limit, title in panels:
                axes = figure.add_axes(box, facecolor=PANEL)
                axes.plot(full_range * scale, full_altitude * scale, color=TRACK, alpha=0.2, linewidth=1.0)
                axes.plot(flown_range * scale, flown_altitude * scale, color=TRACK, linewidth=2.0)
                for label, (gate_range, gate_altitude) in (("high gate", HIGH_GATE), ("low gate", LOW_GATE)):
                    axes.plot(gate_range * scale, gate_altitude * scale, marker="s", color=GATE, markersize=8, fillstyle="none", markeredgewidth=2)
                    axes.annotate(f"Apollo {label}", (gate_range * scale, gate_altitude * scale), xytext=(-75 if label == "low gate" else 6, 10), textcoords="offset points", color=GATE, fontsize=9)
                centre = numpy.array([ranges[phase_number][index] * scale, row["altitude"] * scale])
                if x_limit[1] <= centre[0] <= x_limit[0] and y_limit[0] <= centre[1] <= y_limit[1]:
                    x_per_y = ((x_limit[0] - x_limit[1]) / (box[2] * WIDTH_INCHES)) / ((y_limit[1] - y_limit[0]) / (box[3] * HEIGHT_INCHES))
                    draw_lander(axes, centre, row, 0.07 * (y_limit[1] - y_limit[0]), x_per_y)
                axes.set_xlim(*x_limit)
                axes.set_ylim(*y_limit)
                axes.tick_params(colors=TEXT, labelsize=9)
                for spine in axes.spines.values():
                    spine.set_color("#444a55")
                axes.set_xlabel(f"range to the site along the surface ({unit})", color=TEXT)
                axes.set_ylabel(f"altitude ({unit})", color=TEXT)
                axes.set_title(title, color=TEXT, fontsize=11)
            figure.text(0.05, 0.92, f"2  Powered descent, {name.lower()}: Apollo-style quadratic guidance", color=TEXT, fontsize=14)
            draw_panel(figure, telemetry_lines(name, row, speed_up, dry_mass))
            number = save_frame(figure, directory, number)
    return number
