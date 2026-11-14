# SPDX-License-Identifier: BSD-3-Clause
"""Figures of a run. They are made only from the files in the run directory.

``lunarsim figures <run_dir>`` reads ``manifest.yaml``, ``episodes.parquet`` and, if present,
``trajectories.parquet``, and writes PNG files into ``<run_dir>/figures``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pyplot  # noqa: E402
import numpy  # noqa: E402
import pandas  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from lunarsim.tools.episode_records import TRAJECTORIES_FILE_NAME, read_episode_records  # noqa: E402
from lunarsim.tools.run_manifest import read_manifest  # noqa: E402

# Colours: success and failure are two fixed categorical hues; they also differ in marker
# shape, so the figures do not depend on colour alone.
SUCCESS_COLOR = "#2a78d6"
FAILURE_COLOR = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
SUCCESS_RATE_COLORMAP = LinearSegmentedColormap.from_list("success_rate", ["#cde2fb", "#5598e7", "#0d366b"])


def _style(axes, x_label: str, y_label: str, title: str) -> None:
    """Apply the shared look: quiet grid and axes, dark text."""
    axes.set_facecolor(SURFACE)
    axes.set_xlabel(x_label, color=SECONDARY_INK)
    axes.set_ylabel(y_label, color=SECONDARY_INK)
    axes.set_title(title, color=INK, loc="left", fontsize=11)
    axes.grid(True, color=GRID, linewidth=0.6)
    axes.set_axisbelow(True)
    axes.tick_params(colors=MUTED, labelcolor=SECONDARY_INK)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color("#c3c2b7")


def _run_label(manifest: dict) -> str:
    """Return a one-line description of the run for figure footers."""
    return (
        f"{manifest['scenario_name']} · agent {manifest['extra'].get('agent', '?')} · seed {manifest['root_seed']} · "
        f"git {str(manifest['git']['commit'])[:8]}"
    )


def _save(figure, path: Path, manifest: dict) -> Path:
    """Add the run label, save the figure as PNG and close it."""
    figure.text(0.01, -0.02, _run_label(manifest), color=MUTED, fontsize=7, ha="left", va="top")
    figure.savefig(path, dpi=150, facecolor="white", bbox_inches="tight")
    pyplot.close(figure)
    return path


def plot_outcomes(episodes: pandas.DataFrame, manifest: dict, directory: Path) -> Path:
    """Bar chart: number of episodes per outcome."""
    counts = episodes["outcome"].value_counts()
    figure, axes = pyplot.subplots(figsize=(7.0, 0.45 * len(counts) + 1.6))
    colors = [SUCCESS_COLOR if name == "success" else FAILURE_COLOR for name in counts.index]
    bars = axes.barh(counts.index[::-1], counts.values[::-1], color=colors[::-1], height=0.6)
    for bar, value in zip(bars, counts.values[::-1], strict=True):
        share = 100.0 * value / len(episodes)
        axes.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f"  {value}  ({share:.1f} %)", va="center", color=INK, fontsize=9)
    _style(axes, "episodes", "", f"Outcome of {len(episodes)} episodes · success rate {100 * episodes['success'].mean():.1f} %")
    axes.grid(False)
    axes.set_xlim(0, counts.max() * 1.25)
    return _save(figure, directory / "outcomes.png", manifest)


def plot_touchdown_envelope(episodes: pandas.DataFrame, manifest: dict, directory: Path) -> Path:
    """Scatter: horizontal against vertical speed at contact, with the lander's design envelope."""
    limits = manifest["resolved_scenario"]["lander"]["touchdown_limits"]
    maximum_vertical = limits["maximum_vertical_speed_mps"]["value"]
    maximum_horizontal = limits["maximum_horizontal_speed_mps"]["value"]
    corner = limits["vertical_speed_where_horizontal_limit_starts_to_drop_mps"]["value"]
    landed = episodes[episodes["had_contact"]]
    figure, axes = pyplot.subplots(figsize=(6.4, 4.8))
    for success, color, marker, label in ((True, SUCCESS_COLOR, "o", "success"), (False, FAILURE_COLOR, "x", "failure")):
        group = landed[landed["success"] == success]
        axes.scatter(group["touchdown_horizontal_speed_mps"], -group["touchdown_vertical_speed_mps"], s=14, c=color, marker=marker, linewidths=0.9, alpha=0.7, label=f"{label} ({len(group)})")
    axes.plot([0, maximum_horizontal, maximum_horizontal, 0], [0, 0, corner, maximum_vertical], color=INK, linewidth=1.2, label="design envelope")
    _style(axes, "horizontal speed at contact (m/s)", "descent speed at contact (m/s)", "Touchdown speeds against the landing-gear envelope")
    axes.set_xlim(left=0)
    axes.set_ylim(bottom=0)
    axes.legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    return _save(figure, directory / "touchdown_envelope.png", manifest)


def plot_success_by_start(episodes: pandas.DataFrame, manifest: dict, directory: Path) -> Path:
    """Heat map: success rate against start altitude and start vertical speed."""
    altitude_edges = numpy.linspace(episodes["start_altitude_m"].min(), episodes["start_altitude_m"].max(), 10)
    speed_edges = numpy.linspace(episodes["start_vertical_speed_mps"].min(), episodes["start_vertical_speed_mps"].max(), 7)
    table = episodes.assign(
        altitude_bin=pandas.cut(episodes["start_altitude_m"], altitude_edges, include_lowest=True),
        speed_bin=pandas.cut(episodes["start_vertical_speed_mps"], speed_edges, include_lowest=True),
    ).pivot_table(index="speed_bin", columns="altitude_bin", values="success", aggfunc="mean", observed=False)
    figure, axes = pyplot.subplots(figsize=(8.0, 4.2))
    image = axes.pcolormesh(altitude_edges, speed_edges, table.to_numpy(dtype=float), cmap=SUCCESS_RATE_COLORMAP, vmin=0.0, vmax=1.0, edgecolors="white", linewidth=1.5)
    for row, speed_interval in enumerate(table.index):
        for column, altitude_interval in enumerate(table.columns):
            value = table.iloc[row, column]
            if pandas.notna(value):
                axes.text(altitude_interval.mid, speed_interval.mid, f"{100 * value:.0f}", ha="center", va="center", fontsize=8, color="white" if value > 0.45 else INK)
    _style(axes, "start altitude of the footpads (m)", "start vertical speed (m/s)", "Success rate (%) by start altitude and vertical speed")
    axes.grid(False)
    colorbar = figure.colorbar(image, ax=axes, pad=0.02)
    colorbar.set_label("success rate", color=SECONDARY_INK)
    colorbar.outline.set_visible(False)
    return _save(figure, directory / "success_by_start.png", manifest)


def plot_trajectories(trajectories: pandas.DataFrame, episodes: pandas.DataFrame, manifest: dict, directory: Path) -> Path:
    """Four panels for the logged environments: altitude, descent speed, ground track and tilt."""
    success = episodes[episodes["episode_number"] == 0].set_index("environment_index")["success"]
    figure, panels = pyplot.subplots(2, 2, figsize=(10.0, 7.0))
    panel_definitions = (
        (panels[0, 0], "time_s", "altitude_m", "time (s)", "footpad altitude (m)", "Altitude"),
        (panels[0, 1], "altitude_m", "velocity_z_mps", "footpad altitude (m)", "vertical speed (m/s)", "Vertical speed against altitude"),
        (panels[1, 0], "position_x_m", "position_y_m", "x (m)", "y (m)", "Ground track (target at 0, 0)"),
        (panels[1, 1], "time_s", "tilt_deg", "time (s)", "tilt (deg)", "Tilt"),
    )
    labelled = set()
    for environment_index, track in trajectories.groupby("environment_index"):
        landed_well = bool(success.get(environment_index, False))
        color, style, label = (SUCCESS_COLOR, "-", "success") if landed_well else (FAILURE_COLOR, "--", "failure")
        for axes, x_name, y_name, *_ in panel_definitions:
            axes.plot(track[x_name], track[y_name], color=color, linestyle=style, linewidth=1.0, alpha=0.8, label=None if label in labelled else label)
        labelled.add(label)
    for axes, _, _, x_label, y_label, title in panel_definitions:
        _style(axes, x_label, y_label, title)
    panels[1, 0].set_aspect("equal", adjustable="datalim")
    panels[0, 0].legend(frameon=False, fontsize=8, labelcolor=SECONDARY_INK)
    figure.tight_layout()
    return _save(figure, directory / "trajectories.png", manifest)


def make_figures(run_directory: Path | str) -> list[Path]:
    """Create all figures of a run and return their paths."""
    run_directory = Path(run_directory)
    manifest = read_manifest(run_directory)
    episodes = read_episode_records(run_directory)
    figure_directory = run_directory / "figures"
    figure_directory.mkdir(parents=True, exist_ok=True)
    paths = [
        plot_outcomes(episodes, manifest, figure_directory),
        plot_touchdown_envelope(episodes, manifest, figure_directory),
        plot_success_by_start(episodes, manifest, figure_directory),
    ]
    trajectory_path = run_directory / TRAJECTORIES_FILE_NAME
    if trajectory_path.is_file():
        trajectories = pandas.read_parquet(trajectory_path)
        if len(trajectories):
            paths.append(plot_trajectories(trajectories, episodes, manifest, figure_directory))
    return paths
