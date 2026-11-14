# SPDX-License-Identifier: BSD-3-Clause
"""Drop test on real terrain: does the lander stand, slide, strike its body, or tip over?

Usage::

    python scripts/run_drop_test.py                       # 1000 drops from 5 m, run name drop_test_5m
    python scripts/run_drop_test.py drops=500 drop_height_m=3 run_name=drop_test_3m

Every environment drops one lander, upright and at rest, with the engine off, from
``drop_height_m`` (footpad plane above the ground straight below the centre of mass) at its own
site of the ``apollo17_drop_test`` terrain (a massif flank of the Apollo 17 2 m height map). A
5 m drop hits at about 4.0 m/s, above the 3.05 m/s design limit of the gear, so the touchdown
outcome is mostly ``hard_vertical``; this test looks at what happens after contact.

Site selection: the footprint slope (plane fitted through the ground under the 4.26 m footprint)
is computed on a 4 m grid over the terrain; sites are then drawn evenly from 2.5 degree slope
bins, so steep ground is well represented. Headings are uniform. Both use ``seed``.

Writes ``runs/<run_name>/``: ``manifest.yaml``, ``episodes.parquet`` (touchdown records plus
site columns) and ``drop_test.png`` (final tilt against slope, and tip-over share per slope bin).
"""

import argparse
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Drop test on real terrain.")
AppLauncher.add_app_launcher_args(parser)
arguments, script_arguments = parser.parse_known_args()
simulation_app = AppLauncher(arguments).app

import numpy  # noqa: E402
import torch  # noqa: E402

from lunarsim.core.binding import SlotName  # noqa: E402
from lunarsim.core.rotation_math import quaternion_from_axis_angle  # noqa: E402
from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario  # noqa: E402
from lunarsim.tools.episode_records import EPISODES_FILE_NAME  # noqa: E402
from lunarsim.tools.episode_runner import build_agent, collect_episodes  # noqa: E402
from lunarsim.tools.run_manifest import write_manifest  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SLOPE_BIN_DEGREES = 2.5
SITE_GRID_SPACING_M = 4.0
SITE_EDGE_MARGIN_M = 40.0
FOOTPRINT_SAMPLE_RINGS = (0.0, 0.5, 1.0)
FOOTPRINT_SAMPLES_PER_RING = 8


def footprint_slopes(height_field, x: torch.Tensor, y: torch.Tensor, radius: float) -> torch.Tensor:
    """Return the slope (deg) of a plane fitted through the ground under a footprint centred at (x, y)."""
    offsets = []
    for fraction in FOOTPRINT_SAMPLE_RINGS:
        count = 1 if fraction == 0.0 else FOOTPRINT_SAMPLES_PER_RING
        for index in range(count):
            angle = 2.0 * math.pi * index / count
            offsets.append((fraction * radius * math.cos(angle), fraction * radius * math.sin(angle)))
    offsets = torch.tensor(offsets, device=x.device)
    sample_x = x.unsqueeze(1) + offsets[:, 0]
    sample_y = y.unsqueeze(1) + offsets[:, 1]
    heights = height_field.height_at(sample_x, sample_y)
    design = torch.stack((torch.ones_like(sample_x), sample_x - x.unsqueeze(1), sample_y - y.unsqueeze(1)), dim=2)
    coefficients = torch.linalg.lstsq(design, heights.unsqueeze(2)).solution.squeeze(2)
    return torch.rad2deg(torch.atan(torch.hypot(coefficients[:, 1], coefficients[:, 2])))


def choose_sites(height_field, half_size: float, radius: float, count: int, random: numpy.random.Generator, device):
    """Return (x, y, slope) of ``count`` sites drawn evenly over 2.5 degree footprint-slope bins."""
    lines = torch.arange(-half_size, half_size + 1e-6, SITE_GRID_SPACING_M, device=device)
    grid_x, grid_y = torch.meshgrid(lines, lines, indexing="xy")
    x, y = grid_x.ravel(), grid_y.ravel()
    slope = footprint_slopes(height_field, x, y, radius).cpu().numpy()
    x, y = x.cpu().numpy(), y.cpu().numpy()
    bins = numpy.floor(slope / SLOPE_BIN_DEGREES).astype(int)
    members = {bin_index: numpy.flatnonzero(bins == bin_index) for bin_index in numpy.unique(bins)}
    chosen: list[int] = []
    remaining = count
    # Fill the bins in order of size, smallest first, so a sparse steep bin gives what it has
    # and the rest is shared by the others.
    order = sorted(members, key=lambda bin_index: len(members[bin_index]))
    for position, bin_index in enumerate(order):
        share = remaining // (len(order) - position)
        take = min(share, len(members[bin_index]))
        chosen += list(random.choice(members[bin_index], size=take, replace=False))
        remaining -= take
    chosen = numpy.array(chosen)
    return x[chosen], y[chosen], slope[chosen]


def plot_drop_test(episodes, path: Path, drop_height: float) -> None:
    """Save the figure: final tilt against footprint slope, and the tip-over share per slope bin."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = (
        ("tipped over", episodes["tipped_over"], "#d1495b"),
        ("body contact, upright", ~episodes["tipped_over"] & (episodes["peak_body_contact_force_n"] > 100.0), "#edae49"),
        ("still moving", ~episodes["tipped_over"] & (episodes["peak_body_contact_force_n"] <= 100.0) & ~episodes["settled"], "#66a182"),
        ("at rest on its legs", ~episodes["tipped_over"] & (episodes["peak_body_contact_force_n"] <= 100.0) & episodes["settled"], "#2e4057"),
    )
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.2), constrained_layout=True)
    for label, mask, color in groups:
        left.scatter(episodes.loc[mask, "site_slope_deg"], episodes.loc[mask, "final_tilt_deg"], s=9, color=color,
                     label=f"{label} ({int(mask.sum())})", alpha=0.8, linewidths=0)
    left.axhline(45.0, color="#888888", lw=0.8, ls="--")
    left.set_xlabel("footprint slope at the site (deg)")
    left.set_ylabel("final tilt (deg)")
    left.set_title(f"a   {len(episodes)} drops from {drop_height:g} m, engine off", loc="left")
    left.legend(fontsize=8, frameon=False, loc="upper left")
    bins = numpy.arange(0.0, episodes["site_slope_deg"].max() + SLOPE_BIN_DEGREES, SLOPE_BIN_DEGREES)
    index = numpy.digitize(episodes["site_slope_deg"], bins) - 1
    centres, shares, counts = [], [], []
    for bin_index in range(len(bins) - 1):
        in_bin = index == bin_index
        if in_bin.sum() == 0:
            continue
        centres.append(bins[bin_index] + 0.5 * SLOPE_BIN_DEGREES)
        shares.append(100.0 * episodes.loc[in_bin, "tipped_over"].mean())
        counts.append(int(in_bin.sum()))
    right.bar(centres, shares, width=0.8 * SLOPE_BIN_DEGREES, color="#d1495b")
    for centre, share, number in zip(centres, shares, counts):
        right.text(centre, share + 1.0, str(number), ha="center", fontsize=7, color="#555555")
    right.set_xlabel("footprint slope at the site (deg)")
    right.set_ylabel("tipped over (%)")
    right.set_ylim(0, 105)
    right.set_title("b   tip-over share per 2.5 deg bin (numbers: drops)", loc="left")
    figure.savefig(path, dpi=150)


def main() -> None:
    """Pick the sites, drop the landers, write the run directory and the figure."""
    options = {"drops": "1000", "drop_height_m": "5.0", "run_name": "drop_test_5m", "seed": "0"}
    overrides = []
    for argument in script_arguments:
        key = argument.split("=", 1)[0]
        if key in options:
            options[key] = argument.split("=", 1)[1]
        else:
            overrides.append(argument)
    drops, drop_height = int(options["drops"]), float(options["drop_height_m"])
    scenario = load_scenario(
        "apollo17_drop_test", [f"num_envs={drops}", f"seed={options['seed']}", *overrides]
    )
    environment = LanderEnv(build_lander_env_cfg(scenario, device=arguments.device))
    device = environment.device
    surface = environment.bindings[SlotName.SURFACE]
    start = environment.bindings[SlotName.START]
    preset = environment.bindings[SlotName.PLANT].preset
    random = numpy.random.default_rng(int(options["seed"]))

    site_x, site_y, site_slope = choose_sites(
        surface.height_field, surface.outer_half_size - SITE_EDGE_MARGIN_M, preset.footpad_radius_from_thrust_axis,
        drops, random, device,
    )
    if len(site_x) < drops:
        raise RuntimeError(f"only {len(site_x)} sites found for {drops} drops")
    surface.set_fixed_targets(torch.tensor(numpy.stack((site_x, site_y), axis=1), dtype=torch.float32))
    heading = torch.tensor(random.uniform(0.0, 2.0 * math.pi, drops), dtype=torch.float32, device=device)
    position = torch.zeros(drops, 3, device=device)
    position[:, 2] = drop_height + preset.center_of_mass_height_above_footpad_plane
    orientation = quaternion_from_axis_angle(torch.tensor([0.0, 0.0, 1.0], device=device).repeat(drops, 1), heading)
    start.set_fixed_start_states(position, torch.zeros(drops, 3, device=device), orientation)

    agent = build_agent(environment)
    start_time = time.perf_counter()
    episodes = collect_episodes(
        environment, agent, episodes_per_environment=1, maximum_steps=int(1.2 * environment.max_episode_length) + 10
    )
    elapsed = time.perf_counter() - start_time
    episodes = episodes.sort_values("environment_index").reset_index(drop=True)
    index = episodes["environment_index"].to_numpy()
    episodes["site_x_m"] = site_x[index]
    episodes["site_y_m"] = site_y[index]
    episodes["site_slope_deg"] = site_slope[index]
    episodes["heading_deg"] = numpy.degrees(heading.cpu().numpy()[index])
    episodes["slide_m"] = numpy.hypot(episodes["final_position_x_m"], episodes["final_position_y_m"])

    run_directory = REPOSITORY_ROOT / "runs" / options["run_name"]
    write_manifest(
        run_directory,
        scenario,
        environment.random_generators.seeds(),
        extra={"drops": drops, "drop_height_m": drop_height, "site_seed": int(options["seed"]), "wall_time_s": round(elapsed, 1)},
    )
    episodes.to_parquet(run_directory / EPISODES_FILE_NAME, index=False)
    plot_drop_test(episodes, run_directory / "drop_test.png", drop_height)
    tipped = episodes["tipped_over"]
    print(f"[drop_test] run directory: {run_directory}  wall time {elapsed:.0f} s")
    print(f"[drop_test] drops {len(episodes)}; tipped over {int(tipped.sum())}; "
          f"body contact while upright {int((~tipped & (episodes['peak_body_contact_force_n'] > 100.0)).sum())}; "
          f"at rest {int(episodes['settled'].sum())}")
    print(episodes.groupby(numpy.floor(episodes["site_slope_deg"] / 5.0) * 5.0)["tipped_over"].agg(["count", "mean"]).to_string())
    environment.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
