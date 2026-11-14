# SPDX-License-Identifier: BSD-3-Clause
"""Figure command: makes its PNG files from a run directory alone (CPU, synthetic run)."""

import numpy
import pandas
import pytest

from lunarsim.core.binding import SlotName
from lunarsim.core.random_generators import SlotRandomGenerators
from lunarsim.env.scenario_loader import load_scenario
from lunarsim.tools.command_line import main
from lunarsim.tools.run_manifest import write_manifest


def write_synthetic_run(run_directory):
    """Write a manifest, 200 random episode rows and two short trajectories."""
    scenario = load_scenario("s1_terminal_truth")
    write_manifest(run_directory, scenario, SlotRandomGenerators(0, SlotName.ALL).seeds(), extra={"agent": "zem_zev"})
    random = numpy.random.default_rng(0)
    count = 200
    success = random.random(count) > 0.2
    pandas.DataFrame(
        {
            "environment_index": numpy.arange(count),
            "episode_number": 0,
            "start_altitude_m": random.uniform(5, 50, count),
            "start_vertical_speed_mps": random.uniform(-6, 0, count),
            "had_contact": True,
            "touchdown_vertical_speed_mps": -random.uniform(0.5, 3.5, count),
            "touchdown_horizontal_speed_mps": random.uniform(0, 1.5, count),
            "success": success,
            "outcome": numpy.where(success, "success", "hard_vertical"),
        }
    ).to_parquet(run_directory / "episodes.parquet", index=False)
    time = numpy.linspace(0, 10, 50)
    pandas.concat(
        pandas.DataFrame(
            {
                "environment_index": index,
                "time_s": time,
                "altitude_m": 30 - 3 * time,
                "velocity_z_mps": -3.0,
                "position_x_m": 0.1 * time * index,
                "position_y_m": 0.0,
                "tilt_deg": numpy.abs(numpy.sin(time)),
            }
        )
        for index in range(2)
    ).to_parquet(run_directory / "trajectories.parquet", index=False)


def test_figures_command_writes_four_png_files(tmp_path, capsys):
    """``lunarsim figures <run_dir>`` creates the four figures and prints their paths."""
    write_synthetic_run(tmp_path)
    assert main(["figures", str(tmp_path)]) == 0
    names = sorted(path.name for path in (tmp_path / "figures").glob("*.png"))
    assert names == ["outcomes.png", "success_by_start.png", "touchdown_envelope.png", "trajectories.png"]
    assert all((tmp_path / "figures" / name).stat().st_size > 5000 for name in names)
    assert "outcomes.png" in capsys.readouterr().out


def test_summary_command_prints_the_success_rate(tmp_path, capsys):
    """``lunarsim summary <run_dir>`` prints the episode count and outcome table."""
    write_synthetic_run(tmp_path)
    assert main(["summary", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    assert "episodes: 200" in output and "hard_vertical" in output


def test_failure_groups_separate_impossible_starts():
    """A start that even full thrust cannot stop is grouped as 'impossible start'."""
    import tempfile
    from pathlib import Path

    from lunarsim.tools.failure_analysis import add_failure_groups, failure_table
    from lunarsim.tools.run_manifest import read_manifest

    with tempfile.TemporaryDirectory() as directory:
        scenario = load_scenario("s1_terminal_truth")
        write_manifest(directory, scenario, {}, extra={})
        manifest = read_manifest(Path(directory))
    episodes = pandas.DataFrame(
        {
            "start_altitude_m": [5.0, 10.0, 40.0, 40.0],
            "start_vertical_speed_mps": [-6.0, -5.0, -1.0, -1.0],
            "start_horizontal_velocity_x_mps": [0.0, 2.0, 0.0, 0.0],
            "start_horizontal_velocity_y_mps": [0.0, 0.0, 0.0, 0.0],
            "start_mass_kg": [7327.0] * 4,
            "success": [False, False, True, False],
            "outcome": ["hard_vertical", "hard_horizontal", "success", "tip_over"],
        }
    )
    table = add_failure_groups(episodes, manifest)
    # 5 m at 6 m/s: sqrt(36 - 2 * 2.52 * 5) = 3.29 m/s > 3.05 m/s, so no controller can land it.
    assert table["lowest_possible_contact_speed_mps"].iloc[0] == pytest.approx(3.29, abs=0.02)
    assert table["failure_group"].tolist() == ["impossible start", "low and fast start", "success", "other"]
    assert failure_table(episodes, manifest).loc["all", "all"] == 4
