# SPDX-License-Identifier: BSD-3-Clause
"""Record and render a video of the L2 plume grains, to check them by eye (milestone 5).

Usage::

    python scripts/record_plume_grain_video.py                       # fly, record, render
    python scripts/record_plume_grain_video.py render=0              # fly and record only

It flies the committed L1 vs L2 check again (``s8_plume_check_l2``, 64 landers, seed 0, the
same setup as ``results/s8_plume_check_l2_seed0``) and records, for two landers, every grain
parcel at 25 frames per second. The landers are chosen from the committed run:

1. the lander that starts above 100 ft with its grain eroded mass closest to the middle of the
   Apollo 11 estimate (47-74 kg), so the whole erosion is seen;
2. the lander with the lowest start, where the densest cloud was recorded.

The rerun's numbers for these landers are printed next to the committed ones (GPU physics is not
bit-reproducible). Writes ``runs/<run_name>/``: ``manifest.yaml``, ``episodes.parquet``,
``plume_grains.npz`` (the replay) and ``chosen_landers.yaml``; then renders
``plume_grains.mp4`` with ``scripts/render_replay.py`` (3D view with the parcels as dots, and a
true-scale side view with both plume levels' readings).

Options (``key=value``): ``run_name`` (``s8_plume_grains_video_seed0``), ``seed`` (0),
``landings`` (64), ``landers`` (comma list, overrides the choice), ``render`` (1),
``committed`` (``results/s8_plume_check_l2_seed0/episodes.parquet``).
"""

import argparse
import subprocess
import sys
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record and render the plume grains.")
AppLauncher.add_app_launcher_args(parser)
arguments, script_arguments = parser.parse_known_args()
simulation_app = AppLauncher(arguments).app

import pandas  # noqa: E402
import yaml  # noqa: E402

from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario  # noqa: E402
from lunarsim.tools.episode_records import EPISODES_FILE_NAME  # noqa: E402
from lunarsim.tools.episode_runner import build_agent, collect_episodes  # noqa: E402
from lunarsim.tools.replay_recording import ReplayRecorder  # noqa: E402
from lunarsim.tools.run_manifest import write_manifest  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "s8_plume_check_l2"
ONSET_HEIGHT_M = 30.48  # 100 ft: Armstrong first noticed the dust below it (ALSJ, Apollo 11)
APOLLO_11_ERODED_MASS_MIDDLE_KG = 0.5 * (46.8 + 74.1)
COMPARED_COLUMNS = ("start_altitude_m", "outcome", "grain_eroded_mass_kg", "ejecta_mass_kg", "grain_erosion_start_nozzle_height_m",
                    "erosion_start_nozzle_height_m", "dust_blinded_time_s", "grain_dust_blinded_time_s",
                    "grain_peak_optical_depth", "grain_peak_number_density_per_m3", "episode_duration_s")


def read_options() -> dict[str, str]:
    """Return the ``key=value`` options with their defaults."""
    options = {"run_name": "s8_plume_grains_video_seed0", "seed": "0", "landings": "64", "landers": "", "render": "1",
               "committed": "results/s8_plume_check_l2_seed0/episodes.parquet"}
    for argument in script_arguments:
        key, _, value = argument.partition("=")
        if key not in options:
            raise SystemExit(f"unknown option '{key}'")
        options[key] = value
    return options


def choose_landers(episodes: pandas.DataFrame) -> list[int]:
    """Return [the high-start lander nearest the Apollo 11 eroded mass, the lowest-start lander]."""
    high = episodes[episodes["start_altitude_m"] >= ONSET_HEIGHT_M]
    nearest = high.iloc[(high["grain_eroded_mass_kg"] - APOLLO_11_ERODED_MASS_MIDDLE_KG).abs().argsort()].iloc[0]
    lowest = episodes.sort_values("start_altitude_m").iloc[0]
    return [int(nearest["environment_index"]), int(lowest["environment_index"])]


def lander_record(row: pandas.Series) -> dict:
    """Return the compared columns of one episode as plain numbers and strings."""
    return {name: (str(row[name]) if name == "outcome" else round(float(row[name]), 4)) for name in COMPARED_COLUMNS}


def clip_texts(number: int, total: int, index: int, row: pandas.Series) -> tuple[str, str]:
    """Return the title and detail lines of one clip."""
    outcome = "SUCCESS" if row["outcome"] == "success" else f"FAILURE: {row['outcome']}"
    title = (f"{number}/{total}  {outcome}   L2 plume grains, lander {index}, start {row['start_altitude_m']:.0f} m up "
             f"(S8, ZEM/ZEV, Apollo 17 terrain)")
    detail = (f"whole episode: eroded grains {row['grain_eroded_mass_kg']:.0f} kg, L1 {row['ejecta_mass_kg']:.0f} kg "
              f"(Apollo 11: 47-74 kg)   sensors blinded: L1 {row['dust_blinded_time_s']:.1f} s, grains "
              f"{row['grain_dust_blinded_time_s']:.1f} s")
    return title, detail


def main() -> None:
    """Fly the landings, record the chosen landers' grains, save the replay and render it."""
    options = read_options()
    committed = pandas.read_parquet(REPOSITORY_ROOT / options["committed"])
    landers = [int(value) for value in options["landers"].split(",")] if options["landers"] else choose_landers(committed)
    print(f"[grain video] landers {landers}", flush=True)
    overrides = [f"num_envs={options['landings']}", f"seed={options['seed']}"]
    scenario = load_scenario(SCENARIO, overrides)
    environment = LanderEnv(build_lander_env_cfg(scenario, device=arguments.device))
    recorder = ReplayRecorder(environment, every_n_steps=2, include_contacts=True, grain_landers=landers)
    episodes = collect_episodes(
        environment, build_agent(environment), episodes_per_environment=1,
        maximum_steps=int(1.2 * environment.max_episode_length) + 10, after_step=recorder.record,
    )
    run_directory = REPOSITORY_ROOT / "runs" / options["run_name"]
    write_manifest(run_directory, scenario, environment.random_generators.seeds(),
                   extra={"agent": scenario.slots["agent"].binding, "episodes": int(len(episodes)), "grain_landers": landers,
                          "truth_readers": environment.bus.truth_read_log})
    episodes.to_parquet(run_directory / EPISODES_FILE_NAME, index=False)
    table, committed_table = episodes.set_index("environment_index"), committed.set_index("environment_index")
    titles, details = [], []
    for number, index in enumerate(landers, start=1):
        title, detail = clip_texts(number, len(landers), index, table.loc[index])
        titles.append(title)
        details.append(detail)
    chosen = {
        "rule": "1: start above 100 ft, grain eroded mass nearest the Apollo 11 middle (60.5 kg); 2: lowest start (committed run)",
        "landers": {index: {"rerun": lander_record(table.loc[index]), "committed": lander_record(committed_table.loc[index])} for index in landers},
    }
    with open(run_directory / "chosen_landers.yaml", "w") as file:
        yaml.safe_dump(chosen, file, sort_keys=False)
    print(yaml.safe_dump(chosen, sort_keys=False))
    replay = recorder.save(run_directory / "plume_grains.npz", landers, SCENARIO, [f"seed={options['seed']}"], titles, details,
                           frames_per_second=25.0, notes="L2 plume grains in shadow of the L1 plume")
    print(f"[grain video] {len(episodes)} landings, success rate {episodes['success'].mean():.4f}; wrote {replay}", flush=True)
    environment.close()
    if options["render"] == "1":
        subprocess.run([sys.executable, str(REPOSITORY_ROOT / "scripts" / "render_replay.py"), f"replay={replay}", "mode=video",
                        "camera_distance_m=40", "camera_height_m=10"], check=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
