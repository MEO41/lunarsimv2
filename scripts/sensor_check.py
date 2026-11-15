# SPDX-License-Identifier: BSD-3-Clause
"""Visual sensor check (milestone 2): free falls on Apollo 17 terrain with every sensor recorded.

Four landers fall from 30 m with the engine off, no jets and no control (``agent: zero``,
``actuators: none``), with the full S2 sensor suite plus the cheap flash LiDAR and ``nav: ekf``:

1. flattest spot, upright, at rest: the cleanest free fall;
2. rim of the largest crater, tilted 10 deg, drifting 1 m/s: the beams sweep the crater;
3. steepest footprint, upright, turning (body rates 0.05, -0.03, 0.08 rad/s): the gyros;
4. roughest spot, tilted 5 deg, drifting and turning.

Writes ``runs/sensor_check/sensors.npz`` (every control step: truth, expected and measured sensor
values, filter estimate; ``lunarsim.tools.sensor_recording``) and ``runs/sensor_check/replay.npz``
(poses and contacts for ``scripts/render_sensor_check.py``). Plots: ``python
scripts/plot_sensor_check.py``.
"""

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record free falls with all sensors.")
AppLauncher.add_app_launcher_args(parser)
arguments, _ = parser.parse_known_args()
arguments.headless = True
simulation_app = AppLauncher(arguments).app

from pathlib import Path  # noqa: E402

import numpy  # noqa: E402
import torch  # noqa: E402

from lunarsim.core.binding import SlotName  # noqa: E402
from lunarsim.core.rotation_math import quaternion_from_axis_angle  # noqa: E402
from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario  # noqa: E402
from lunarsim.tools.episode_runner import build_agent, collect_episodes  # noqa: E402
from lunarsim.tools.replay_recording import ReplayRecorder  # noqa: E402
from lunarsim.tools.sensor_recording import SensorRecorder  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "s2_apollo17_ekf"
OVERRIDES = [
    "num_envs=4", "agent=zero", "actuators=none", "sensors=terminal_suite_with_lidar",
    "metrics.tip_over_tilt_deg=150", "metrics.settle_hold_s=2.0",
]
DROP_HEIGHT_M = 30.0
SEARCH_HALF_SIZE_M = 60.0


def plane_slope_and_relief(field, x, y, radius, device):
    """Return (slope deg, relief m) of the ground under a footprint disc at every (x, y)."""
    rings = [(0.0, 0.0)] + [(radius * r / 5 * math.cos(2 * math.pi * k / 20), radius * r / 5 * math.sin(2 * math.pi * k / 20)) for r in range(1, 6) for k in range(20)]
    disc = torch.tensor(rings, dtype=torch.float32, device=device)
    heights = field.height_at(x.unsqueeze(1) + disc[:, 0], y.unsqueeze(1) + disc[:, 1])
    design = torch.stack((torch.ones_like(heights), disc[:, 0].expand_as(heights), disc[:, 1].expand_as(heights)), dim=2)
    coefficients = torch.linalg.lstsq(design, heights.unsqueeze(2)).solution.squeeze(2)
    slope = torch.rad2deg(torch.atan(torch.hypot(coefficients[:, 1], coefficients[:, 2])))
    return slope, heights.max(dim=1).values - heights.min(dim=1).values


def choose_spots(surface, preset, device):
    """Return [(name, x, y)] for the four drops described in the module docstring."""
    field, craters = surface.height_field, surface.terrain.craters
    lines = torch.arange(-SEARCH_HALF_SIZE_M, SEARCH_HALF_SIZE_M + 1e-6, 1.0, device=device)
    grid_x, grid_y = torch.meshgrid(lines, lines, indexing="xy")
    x, y = grid_x.ravel(), grid_y.ravel()
    slope, relief = plane_slope_and_relief(field, x, y, preset.footpad_radius_from_thrust_axis + 0.5, device)
    inside = (numpy.abs(craters.center_x) < SEARCH_HALF_SIZE_M - 10) & (numpy.abs(craters.center_y) < SEARCH_HALF_SIZE_M - 10)
    big = numpy.flatnonzero(inside)[numpy.argmax(craters.diameter[inside])]
    rim_x = float(craters.center_x[big] - 0.5 * craters.diameter[big])
    return [
        ("flattest spot", float(x[relief.argmin()]), float(y[relief.argmin()])),
        (f"rim of a {craters.diameter[big]:.1f} m crater", rim_x, float(craters.center_y[big])),
        (f"steepest footprint ({float(slope.max()):.1f} deg plane)", float(x[slope.argmax()]), float(y[slope.argmax()])),
        (f"roughest spot ({float(relief.max()):.2f} m relief)", float(x[relief.argmax()]), float(y[relief.argmax()])),
    ]


def main() -> None:
    """Choose the spots, set the starts, fly until every lander rests, save the recordings."""
    scenario = load_scenario(SCENARIO, OVERRIDES)
    environment = LanderEnv(build_lander_env_cfg(scenario, device=arguments.device))
    device = environment.device
    surface = environment.bindings[SlotName.SURFACE]
    start = environment.bindings[SlotName.START]
    preset = environment.bindings[SlotName.PLANT].preset
    spots = choose_spots(surface, preset, device)
    surface.set_fixed_targets(torch.tensor([[spot_x, spot_y] for _, spot_x, spot_y in spots], device=device))
    count = environment.num_envs
    position = torch.zeros(count, 3, device=device)
    position[:, 2] = DROP_HEIGHT_M + preset.center_of_mass_height_above_footpad_plane
    velocity = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, -0.5], [0.0, 0.0, 0.0], [0.0, 0.5, -1.0]], device=device)
    tilt_axis = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], device=device)
    tilt = torch.tensor([0.0, 10.0, 0.0, 5.0], device=device) * (math.pi / 180.0)
    orientation = quaternion_from_axis_angle(tilt_axis, tilt)
    rates = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.05, -0.03, 0.08], [-0.04, 0.06, 0.02]], device=device)
    start.set_fixed_start_states(position, velocity, orientation, angular_velocity=rates)

    sensors = SensorRecorder(environment)
    replay = ReplayRecorder(environment, every_n_steps=2, include_contacts=True)

    def after_step(step_index: int, command) -> None:
        """Record both recordings after every control step."""
        sensors.record(step_index, command)
        replay.record(step_index, command)

    episodes = collect_episodes(
        environment, build_agent(environment), episodes_per_environment=1,
        maximum_steps=int(1.2 * environment.max_episode_length) + 10, after_step=after_step,
    ).sort_values("environment_index").reset_index(drop=True)
    run_directory = REPOSITORY_ROOT / "runs" / "sensor_check"
    names = [name for name, _, _ in spots]
    titles, details = [], []
    for index, name in enumerate(names):
        record = episodes.iloc[index]
        titles.append(f"{index + 1}/4  FREE FALL, engine off, no jets: {name}")
        details.append(f"from {DROP_HEIGHT_M:.0f} m; impact {abs(record['touchdown_vertical_speed_mps']):.2f} m/s; outcome {record['outcome']}")
        print(f"[sensor check] {index + 1} {name}: impact {abs(record['touchdown_vertical_speed_mps']):.2f} m/s, {record['outcome']}")
    sensors.save(run_directory / "sensors.npz", {"names": names, "control_step_s": environment.scenario.control_step_duration})
    replay.save(run_directory / "replay.npz", list(range(count)), SCENARIO, OVERRIDES, titles, details, frames_per_second=25.0,
                notes="sensor check, free fall")
    print(f"[sensor check] wrote {run_directory}")
    environment.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
