# SPDX-License-Identifier: BSD-3-Clause
"""Paired comparison of two simulators (test T3 tool) and the optional start heading (CPU)."""

import math

import pandas
import pytest
import torch
from analytic_environment_support import make_environment

from lunarsim.core.binding import SlotName
from lunarsim.core.rotation_math import body_up_axis_in_parent, rotate_vector_body_to_parent, tilt_angle_from_vertical
from lunarsim.tools.paired_simulator_comparison import (
    compare_paired_runs,
    format_paired_comparison,
    mcnemar_exact_p_value,
    wilson_interval,
)


def test_wilson_interval_known_values():
    """50 of 100 gives 40.4 % to 59.6 %; 0 of 10 starts at zero; the interval holds the proportion."""
    low, high = wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-4) and high == pytest.approx(0.5962, abs=1e-4)
    low, high = wilson_interval(0, 10)
    assert low == 0.0 and high == pytest.approx(0.2775, abs=1e-4)
    low, high = wilson_interval(4012, 4096)
    assert low < 4012 / 4096 < high


def test_mcnemar_exact_p_value_known_values():
    """No discordant pairs gives 1; 10 against 0 gives 2 / 2^10; symmetric counts give 1."""
    assert mcnemar_exact_p_value(0, 0) == 1.0
    assert mcnemar_exact_p_value(10, 0) == pytest.approx(2.0 / 1024.0)
    assert mcnemar_exact_p_value(0, 10) == pytest.approx(2.0 / 1024.0)
    assert mcnemar_exact_p_value(5, 5) == 1.0


def write_run(directory, success, outcome, vertical_speed, altitude):
    """Write a minimal episodes.parquet with one first episode per environment."""
    count = len(success)
    frame = pandas.DataFrame(
        {
            "environment_index": range(count),
            "episode_number": [0] * count,
            "start_altitude_m": altitude,
            "start_vertical_speed_mps": [-1.0] * count,
            "start_horizontal_velocity_x_mps": [0.0] * count,
            "start_horizontal_velocity_y_mps": [0.0] * count,
            "start_tilt_deg": [1.0] * count,
            "success": success,
            "outcome": outcome,
            "had_contact": [True] * count,
            "touchdown_vertical_speed_mps": vertical_speed,
            "touchdown_horizontal_speed_mps": [0.1] * count,
            "touchdown_tilt_deg": [0.5] * count,
            "touchdown_tilt_rate_deg_per_s": [0.1] * count,
            "touchdown_time_s": [10.0] * count,
            "propellant_used_kg": [50.0] * count,
            "touchdown_position_x_m": [0.0] * count,
            "touchdown_position_y_m": [0.0] * count,
        }
    )
    directory.mkdir()
    frame.to_parquet(directory / "episodes.parquet", index=False)


def test_paired_comparison_counts_agreement_and_differences(tmp_path):
    """Two runs on the same starts: agreement table, McNemar p-value, outcome cross table and touchdown differences."""
    altitude = [10.0, 20.0, 30.0, 40.0]
    write_run(tmp_path / "analytic", [True, True, True, False], ["success", "success", "success", "hard_vertical"], [-1.0, -1.0, -1.0, -3.5], altitude)
    write_run(tmp_path / "physx", [True, True, False, False], ["success", "success", "tip_over", "hard_vertical"], [-1.2, -1.2, -1.2, -3.7], altitude)
    summary = compare_paired_runs(tmp_path / "analytic", tmp_path / "physx", ("analytic", "physx"))
    assert summary["pairs"] == 4
    assert summary["analytic_success_rate"] == 0.75 and summary["physx_success_rate"] == 0.5
    assert summary["agreement"] == {"both_succeed": 2, "only_analytic_succeeds": 1, "only_physx_succeeds": 0, "both_fail": 1}
    assert summary["mcnemar_exact_p_value"] == 1.0
    assert summary["outcome_agreement_rate"] == 0.75
    vertical = summary["touchdown_differences"].set_index("value").loc["touchdown_vertical_speed_mps"]
    assert vertical["mean difference"] == pytest.approx(-0.2)
    assert "exact McNemar" in format_paired_comparison(summary)


def test_paired_comparison_refuses_runs_with_different_starts(tmp_path):
    """Runs that did not fly the same starts cannot be paired."""
    write_run(tmp_path / "first", [True, True], ["success", "success"], [-1.0, -1.0], [10.0, 20.0])
    write_run(tmp_path / "second", [True, True], ["success", "success"], [-1.0, -1.0], [10.0, 25.0])
    with pytest.raises(ValueError, match="different states"):
        compare_paired_runs(tmp_path / "first", tmp_path / "second")


def test_start_heading_turns_the_lander_without_changing_the_other_start_values():
    """heading_deg turns body x about the vertical; tilt and the other drawn values stay the same."""
    plain = make_environment("s1_terminal_truth", ["num_envs=64", "seed=3"])
    turned = make_environment("s1_terminal_truth", ["num_envs=64", "seed=3", "+start.heading_deg=[0,360]"])
    plain.reset()
    turned.reset()
    plain_start = plain.bindings[SlotName.START]
    turned_start = turned.bindings[SlotName.START]
    assert torch.equal(plain_start.position, turned_start.position)
    assert torch.equal(plain_start.velocity, turned_start.velocity)
    assert torch.allclose(tilt_angle_from_vertical(plain_start.orientation), tilt_angle_from_vertical(turned_start.orientation), atol=1e-5)
    assert torch.allclose(body_up_axis_in_parent(plain_start.orientation), body_up_axis_in_parent(turned_start.orientation), atol=1e-5)
    body_x = torch.tensor([1.0, 0.0, 0.0]).repeat(64, 1)
    heading = torch.atan2(*rotate_vector_body_to_parent(turned_start.orientation, body_x)[:, [1, 0]].T)
    assert float(heading.max() - heading.min()) > math.radians(300.0)


def test_touchdown_records_the_start_heading():
    """The episode record has the start heading: 90 degrees when the box fixes it there."""
    environment = make_environment(
        "s1_terminal_truth", ["num_envs=2", "+start.heading_deg=[90,90]", "start.tilt_deg=[0,0]", "start.altitude_m=[0.2,0.2]",
                              "start.v_vertical_mps=[-0.5,-0.5]", "start.v_horizontal_mps=[0,0]", "actuators.initial_throttle=0.0"]
    )
    environment.reset()
    metrics = environment.bindings[SlotName.METRICS]
    assert torch.allclose(metrics.start["heading"], torch.full((2,), math.pi / 2), atol=1e-5)
