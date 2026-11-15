# SPDX-License-Identifier: BSD-3-Clause
"""Agent comparison (CPU): same-start check, rates per group, low-and-fast split."""

import pandas
import pytest

from lunarsim.tools.agent_comparison import compare_groups
from lunarsim.tools.episode_records import EPISODES_FILE_NAME


def write_run(directory, successes, altitudes, vertical_speeds):
    """Write a minimal episodes file with one first episode per environment."""
    directory.mkdir(parents=True)
    count = len(successes)
    frame = pandas.DataFrame(
        {
            "environment_index": range(count),
            "episode_number": [0] * count,
            "start_altitude_m": altitudes,
            "start_vertical_speed_mps": vertical_speeds,
            "start_horizontal_velocity_x_mps": [0.0] * count,
            "start_horizontal_velocity_y_mps": [0.0] * count,
            "success": successes,
            "had_contact": [True] * count,
            "touchdown_vertical_speed_mps": [-1.0] * count,
            "touchdown_horizontal_speed_mps": [0.1] * count,
            "propellant_used_kg": [50.0] * count,
            "outcome_code": [0 if s else 4 for s in successes],
        }
    )
    frame.to_parquet(directory / EPISODES_FILE_NAME)


def test_rates_and_low_and_fast_split(tmp_path):
    """Two PPO seeds and one ZEM/ZEV run on the same 4 starts (2 of them low and fast)."""
    altitudes, speeds = [10.0, 10.0, 40.0, 40.0], [-5.0, -5.0, -1.0, -1.0]
    write_run(tmp_path / "ppo0", [True, False, True, True], altitudes, speeds)
    write_run(tmp_path / "ppo1", [True, True, True, True], altitudes, speeds)
    write_run(tmp_path / "zem", [False, False, True, True], altitudes, speeds)
    table = compare_groups({"ppo": [tmp_path / "ppo0", tmp_path / "ppo1"], "zem": [tmp_path / "zem"]}).set_index("agent")
    assert table.loc["ppo", "success %"] == pytest.approx(87.5)
    assert table.loc["ppo", "low-and-fast success %"] == pytest.approx(75.0)
    assert table.loc["zem", "low-and-fast success %"] == pytest.approx(0.0)
    assert table.loc["zem", "other starts success %"] == pytest.approx(100.0)


def test_different_starts_are_rejected(tmp_path):
    """Runs that flew different start states cannot be compared."""
    write_run(tmp_path / "a", [True, True], [10.0, 20.0], [-1.0, -1.0])
    write_run(tmp_path / "b", [True, True], [10.0, 21.0], [-1.0, -1.0])
    with pytest.raises(ValueError):
        compare_groups({"a": [tmp_path / "a"], "b": [tmp_path / "b"]})
