# SPDX-License-Identifier: BSD-3-Clause
"""Check the footpad contacts on real terrain: free-fall drops on chosen spots, then compare sensors with geometry.

Usage::

    python scripts/contact_check.py                  # writes runs/contact_check/contacts.npz and contact_table.csv
    python scripts/render_replay.py replay=runs/contact_check/contacts.npz mode=stills

Eight landers fall freely (engine off, no jets, no agent) onto spots of the Apollo 17 detail
patch picked from the terrain itself: the flattest spot, the steepest footprint (plane fitted under it), the middle of the
largest crater, the rim of that crater, one footpad on a rock, a tall rock under the descent
stage, a small crater under one footpad, and the roughest spot. Each lander starts upright with
its heading at 0 (leg 1 along +x), with the footpad plane 1 m above the highest ground under
its footprint, so no footpad starts inside the ground.

At rest (2 s still), every footpad's contact-sensor force is compared with its gap to the ground
lookup (smallest vertical distance from the dish's lower surface to the ground): a loaded pad
must touch (|gap| < 3 cm, the 2 cm contact offset plus 1 cm), a free pad must not be inside the
ground. The result is printed and written to ``contact_table.csv``; the replay can be rendered.
"""

import argparse
import math
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Footpad contact check on real terrain.")
AppLauncher.add_app_launcher_args(parser)
arguments, script_arguments = parser.parse_known_args()
simulation_app = AppLauncher(arguments).app

import numpy  # noqa: E402
import torch  # noqa: E402

from lunarsim.core.binding import SlotName  # noqa: E402
from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario  # noqa: E402
from lunarsim.tools.episode_runner import build_agent, collect_episodes  # noqa: E402
from lunarsim.tools.replay_recording import ReplayRecorder  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "apollo17_contact_check"
INSIDE_LIMIT_M = 65.0  # spots stay inside the detail patch, away from its tapered edge
CLEARANCE_ABOVE_HIGHEST_GROUND_M = 1.0
TOUCHING_GAP_LIMIT_M = 0.03
CONTACT_FORCE_THRESHOLD_N = 100.0


def footprint_samples(radius: float, rings: int = 6, per_ring: int = 24) -> numpy.ndarray:
    """Return (K, 2) offsets that cover a disc of ``radius`` (centre, then rings)."""
    offsets = [(0.0, 0.0)]
    for ring in range(1, rings + 1):
        for index in range(per_ring):
            angle = 2.0 * math.pi * index / per_ring
            offsets.append((radius * ring / rings * math.cos(angle), radius * ring / rings * math.sin(angle)))
    return numpy.array(offsets)


def choose_spots(surface, preset, device) -> list[tuple[str, float, float]]:
    """Return (name, x, y) of the eight centre-of-mass spots described in the module docstring."""
    terrain, field = surface.terrain, surface.height_field
    craters, rocks = terrain.craters, terrain.rocks
    axis_x, axis_y = preset.vehicle_axis_position
    reach = preset.footpad_radius_from_thrust_axis
    disc = torch.tensor(footprint_samples(reach + 0.5), dtype=torch.float32, device=device)
    lines = torch.arange(-INSIDE_LIMIT_M, INSIDE_LIMIT_M + 1e-6, 1.0, device=device)
    grid_x, grid_y = torch.meshgrid(lines, lines, indexing="xy")
    x, y = grid_x.ravel(), grid_y.ravel()
    heights = field.height_at(x.unsqueeze(1) + disc[:, 0], y.unsqueeze(1) + disc[:, 1])
    relief = (heights.max(dim=1).values - heights.min(dim=1).values).cpu().numpy()
    # Slope of a plane fitted through the ground under the footprint (not the slope at one point,
    # which is largest on the flank of a rock).
    design = torch.stack((torch.ones_like(heights), disc[:, 0].expand_as(heights), disc[:, 1].expand_as(heights)), dim=2)
    coefficients = torch.linalg.lstsq(design, heights.unsqueeze(2)).solution.squeeze(2)
    slope = torch.rad2deg(torch.atan(torch.hypot(coefficients[:, 1], coefficients[:, 2]))).cpu().numpy()
    x, y = x.cpu().numpy(), y.cpu().numpy()
    spots = [("flattest spot", x[relief.argmin()], y[relief.argmin()])]
    spots.append((f"steepest footprint ({slope.max():.1f} deg plane)", x[slope.argmax()], y[slope.argmax()]))
    inside = (numpy.abs(craters.center_x) < INSIDE_LIMIT_M - 10) & (numpy.abs(craters.center_y) < INSIDE_LIMIT_M - 10)
    big = numpy.flatnonzero(inside)[numpy.argmax(craters.diameter[inside])]
    spots.append((f"middle of a {craters.diameter[big]:.1f} m crater", craters.center_x[big], craters.center_y[big]))
    spots.append((f"rim of that crater (leg 3 inside)", craters.center_x[big] + 0.5 * craters.diameter[big], craters.center_y[big]))
    rock_inside = (numpy.abs(rocks.center_x) < INSIDE_LIMIT_M - 6) & (numpy.abs(rocks.center_y) < INSIDE_LIMIT_M - 6)
    medium = numpy.flatnonzero(rock_inside & (rocks.height > 0.3) & (rocks.height < 0.9))
    one = medium[numpy.argmax(rocks.height[medium])]
    # Leg 1 sits at (axis + footpad radius, axis): put its dish on the rock.
    spots.append((f"leg 1 on a {rocks.height[one]:.2f} m tall rock", rocks.center_x[one] - reach - axis_x, rocks.center_y[one] - axis_y))
    tall = numpy.flatnonzero(rock_inside)[numpy.argmax(rocks.height[rock_inside])]
    spots.append((f"a {rocks.height[tall]:.2f} m tall rock under the descent stage", rocks.center_x[tall] - axis_x, rocks.center_y[tall] - axis_y))
    small = numpy.flatnonzero(inside & (craters.diameter > 1.5) & (craters.diameter < 3.0) & (craters.depth > 0.15))
    pick = small[numpy.argmax(craters.depth[small])]
    # Leg 2 sits at (axis, axis + footpad radius): put its dish in the crater.
    spots.append((f"leg 2 in a {craters.diameter[pick]:.1f} m crater ({craters.depth[pick]:.2f} m deep)",
                  craters.center_x[pick] - axis_x, craters.center_y[pick] - reach - axis_y))
    spots.append(("roughest spot", x[relief.argmax()], y[relief.argmax()]))
    return spots


def main() -> None:
    """Drop the landers on the spots, record the contacts, compare force and gap at rest."""
    run_directory = REPOSITORY_ROOT / "runs" / "contact_check"
    scenario = load_scenario(SCENARIO)
    environment = LanderEnv(build_lander_env_cfg(scenario, device=arguments.device))
    device = environment.device
    surface = environment.bindings[SlotName.SURFACE]
    start = environment.bindings[SlotName.START]
    preset = environment.bindings[SlotName.PLANT].preset
    spots = choose_spots(surface, preset, device)
    count = environment.num_envs
    if len(spots) != count:
        raise RuntimeError(f"{len(spots)} spots for {count} environments")
    target = torch.tensor([[spot_x, spot_y] for _, spot_x, spot_y in spots], dtype=torch.float32, device=device)
    surface.set_fixed_targets(target)
    disc = torch.tensor(footprint_samples(preset.footpad_radius_from_thrust_axis + 0.6), dtype=torch.float32, device=device)
    ground = surface.height_field.height_at(target[:, 0:1] + disc[:, 0], target[:, 1:2] + disc[:, 1])
    centre_ground = surface.height_field.height_at(target[:, 0], target[:, 1])
    altitude = ground.max(dim=1).values - centre_ground + CLEARANCE_ABOVE_HIGHEST_GROUND_M
    position = torch.zeros(count, 3, device=device)
    position[:, 2] = altitude + preset.center_of_mass_height_above_footpad_plane
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(count, 1)
    start.set_fixed_start_states(position, torch.zeros(count, 3, device=device), identity)

    recorder = ReplayRecorder(environment, every_n_steps=2, include_contacts=True)
    episodes = collect_episodes(
        environment, build_agent(environment), episodes_per_environment=1,
        maximum_steps=int(1.2 * environment.max_episode_length) + 10, after_step=recorder.record,
    )
    episodes = episodes.sort_values("environment_index").reset_index(drop=True)
    frames, valid = numpy.stack(recorder.frames), numpy.stack(recorder.valid)
    columns = {name: index for index, name in enumerate(recorder.columns)}
    rows, failures = [], 0
    titles, details = [], []
    for index, (name, _, _) in enumerate(spots):
        last = int(numpy.flatnonzero(valid[:, index]).max())
        row = frames[last, index]
        record = episodes.iloc[index]
        rest = "AT REST" if record["settled"] else ("TIPPED OVER" if record["final_tilt_deg"] > 45 else "STILL MOVING")
        titles.append(f"{index + 1}/{count}  {rest}: {name}  (free fall from {float(altitude[index]):.2f} m, engine off, no jets)")
        details.append(f"impact {abs(record['touchdown_vertical_speed_mps']):.2f} m/s   ground under footprint: "
                       f"slope {record['touchdown_ground_slope_deg']:.1f} deg, relief {record['touchdown_ground_relief_m']:.2f} m")
        for leg in range(1, 5):
            force = float(numpy.linalg.norm([row[columns[f"leg_{leg}_force_{axis}"]] for axis in "xyz"]))
            gap = float(row[columns[f"leg_{leg}_gap"]])
            touching = force > CONTACT_FORCE_THRESHOLD_N
            agrees = abs(gap) < TOUCHING_GAP_LIMIT_M if touching else gap > -0.005
            failures += not agrees
            rows.append((index + 1, name, leg, force, gap, touching, agrees))
            print(f"[contact] {index + 1} {name[:44]:44s} leg {leg}: {force / 1000:6.2f} kN  gap {100 * gap:+6.2f} cm  "
                  f"{'touching' if touching else 'free    '}  {'agrees' if agrees else 'DISAGREES'}")
        total = sum(entry[3] for entry in rows[-4:])
        print(f"[contact] {index + 1} sum of leg forces {total / 1000:.2f} kN (weight {preset.mass_at_start * 1.6242 / 1000:.2f} kN), "
              f"body force {row[columns['body_force']]:.0f} N, final tilt {record['final_tilt_deg']:.1f} deg, {rest.lower()}")
    run_directory.mkdir(parents=True, exist_ok=True)
    with open(run_directory / "contact_table.csv", "w") as file:
        file.write("spot,name,leg,force_n,gap_m,touching,agrees\n")
        for spot, name, leg, force, gap, touching, agrees in rows:
            file.write(f"{spot},\"{name}\",{leg},{force:.1f},{gap:.4f},{touching},{agrees}\n")
    recorder.save(
        run_directory / "contacts.npz", list(range(count)), SCENARIO, [], titles, details,
        frames_per_second=25.0, notes="contact check, free fall",
    )
    print(f"[contact] {len(rows)} footpads, {failures} disagree; wrote {run_directory}")
    environment.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
