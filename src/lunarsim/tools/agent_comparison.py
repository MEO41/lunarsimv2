# SPDX-License-Identifier: BSD-3-Clause
"""Compare agents on the same held-out start states (milestone 1b task 6, milestone 2 S1 vs S2).

Each group is a label and one or more evaluation run directories (for example five PPO seeds).
For each group the table gives the success rate as mean and standard deviation over its runs,
the success rate on low-and-fast starts (below 20 m and faster than 3 m/s down, the region
reported separately since milestone 1a) and on all other starts, and medians of the touchdown
speeds and the propellant used. :func:`check_same_starts` makes sure every run flew the same
start states, so differences come from the agents.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import pandas

from lunarsim.tools.episode_records import read_episode_records

START_COLUMNS = ("start_altitude_m", "start_vertical_speed_mps", "start_horizontal_velocity_x_mps", "start_horizontal_velocity_y_mps")


def first_episodes(run_directory: Path | str) -> pandas.DataFrame:
    """Return the first episode of every environment of a run, sorted by environment."""
    episodes = read_episode_records(run_directory)
    first = episodes[episodes["episode_number"] == 0]
    return first.sort_values("environment_index").reset_index(drop=True)


def low_and_fast(episodes: pandas.DataFrame) -> pandas.Series:
    """Return True for starts below 20 m with more than 3 m/s of descent rate."""
    return (episodes["start_altitude_m"] < 20.0) & (episodes["start_vertical_speed_mps"] < -3.0)


def check_same_starts(runs: dict[str, pandas.DataFrame], tolerance: float = 1e-4) -> None:
    """Raise ValueError if the runs did not fly the same start states."""
    names = list(runs)
    reference = runs[names[0]]
    for name in names[1:]:
        other = runs[name]
        if len(other) != len(reference):
            raise ValueError(f"{name} has {len(other)} first episodes, {names[0]} has {len(reference)}")
        for column in START_COLUMNS:
            if not numpy.allclose(other[column].to_numpy(), reference[column].to_numpy(), atol=tolerance):
                raise ValueError(f"{name} and {names[0]} started from different states ({column})")


def compare_groups(groups: dict[str, list[Path | str]]) -> pandas.DataFrame:
    """Return one row per group with the comparison numbers (rates in percent)."""
    loaded = {f"{label}/{Path(directory).name}": first_episodes(directory) for label, directories in groups.items() for directory in directories}
    check_same_starts(loaded)
    rows = []
    for label, directories in groups.items():
        frames = [loaded[f"{label}/{Path(directory).name}"] for directory in directories]
        success = [100.0 * frame["success"].mean() for frame in frames]
        hard = [100.0 * frame.loc[low_and_fast(frame), "success"].mean() for frame in frames]
        easy = [100.0 * frame.loc[~low_and_fast(frame), "success"].mean() for frame in frames]
        merged = pandas.concat(frames)
        landed = merged[merged["had_contact"]]
        rows.append(
            {
                "agent": label,
                "runs": len(frames),
                "episodes per run": len(frames[0]),
                "success %": numpy.mean(success),
                "success sd": numpy.std(success, ddof=1) if len(success) > 1 else 0.0,
                "low-and-fast success %": numpy.mean(hard),
                "other starts success %": numpy.mean(easy),
                "touchdown vertical speed median (m/s)": landed["touchdown_vertical_speed_mps"].abs().median(),
                "touchdown horizontal speed median (m/s)": landed["touchdown_horizontal_speed_mps"].median(),
                "propellant median (kg)": merged["propellant_used_kg"].median(),
            }
        )
    return pandas.DataFrame(rows)


def outcome_counts(groups: dict[str, list[Path | str]]) -> pandas.DataFrame:
    """Return the outcome counts per group, summed over its runs."""
    table = {}
    for label, directories in groups.items():
        merged = pandas.concat([first_episodes(directory) for directory in directories])
        table[label] = merged["outcome"].value_counts()
    return pandas.DataFrame(table).fillna(0).astype(int)


def comparison_figure(groups: dict[str, list[Path | str]], output: Path | str, title: str) -> Path:
    """Draw success (all, low-and-fast, other starts; one dot per run), touchdown speed and propellant."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ("#5f6b7a", "#1f6fd1", "#e07b00", "#2e9e5b", "#7b4fc4")
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    regions = (("all starts", None), ("low and fast", True), ("other starts", False))
    width = 0.8 / len(groups)
    for index, (label, directories) in enumerate(groups.items()):
        frames = [first_episodes(directory) for directory in directories]
        for region_index, (_, hard) in enumerate(regions):
            rates = []
            for frame in frames:
                chosen = frame if hard is None else frame[low_and_fast(frame) == hard]
                rates.append(100.0 * chosen["success"].mean())
            position = region_index + (index - (len(groups) - 1) / 2) * width
            axes[0].bar(position, numpy.mean(rates), width * 0.9, color=colors[index], label=label if region_index == 0 else None)
            axes[0].scatter([position] * len(rates), rates, color="black", s=10, zorder=3)
            axes[0].text(position, numpy.mean(rates) + 0.3, f"{numpy.mean(rates):.1f}", ha="center", fontsize=8)
        merged = pandas.concat(frames)
        landed = merged[merged["had_contact"]]
        axes[1].hist(landed["touchdown_vertical_speed_mps"].abs(), bins=numpy.linspace(0, 3.2, 65), histtype="step", linewidth=1.6, color=colors[index], density=True, label=label)
        axes[2].hist(merged["propellant_used_kg"], bins=numpy.linspace(0, 300, 61), histtype="step", linewidth=1.6, color=colors[index], density=True, label=label)
    axes[0].set_xticks(range(len(regions)), [name for name, _ in regions])
    axes[0].set_ylim(80, 101)
    axes[0].set_ylabel("success (%)")
    axes[0].set_title("Success on the same held-out starts (dots: one run each)", fontsize=10, loc="left")
    axes[1].axvline(3.048, color="#999999", linestyle=":", linewidth=1)
    axes[1].set_xlabel("vertical speed at first contact (m/s); limit 3.05 m/s")
    axes[1].set_title("Touchdown vertical speed", fontsize=10, loc="left")
    axes[2].set_xlabel("propellant used (kg)")
    axes[2].set_title("Propellant used per landing", fontsize=10, loc="left")
    for axis in axes:
        axis.grid(color="#dddddd", linewidth=0.6)
        axis.legend(fontsize=8, frameon=False)
        for side in ("top", "right"):
            axis.spines[side].set_visible(False)
    figure.suptitle(title, fontsize=12, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=120)
    plt.close(figure)
    return output
