# SPDX-License-Identifier: BSD-3-Clause
"""Record landings on the Apollo 17 terrain and keep chosen successes and failures for a video.

Usage::

    python scripts/record_landing_videos.py                                   # fly 3072 landings, keep 10
    python scripts/render_replay.py replay=runs/landing_videos/landings.npz mode=video

Options (``key=value``): ``run_name`` (default ``landing_videos``), ``landings`` (3072),
``seed`` (0), ``successes`` (5), ``failures`` (5), ``contacts`` (1: also record footpad contacts).

The run flies ``landings`` episodes of ``apollo17_terminal_truth`` with ZEM/ZEV (no camera, full
speed) and records every lander at 25 frames per second until its first episode ends. It keeps:
failures with ``body_contact`` first, then other failure causes, and successes spread over the
ground relief under the footprint. Each clip starts 12 s before the first contact. Writes
``runs/<run_name>/landings.npz`` (for ``scripts/render_replay.py``) and ``episodes.parquet``.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record landings for a video.")
AppLauncher.add_app_launcher_args(parser)
arguments, script_arguments = parser.parse_known_args()
simulation_app = AppLauncher(arguments).app

import numpy  # noqa: E402

from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario  # noqa: E402
from lunarsim.tools.episode_records import EPISODES_FILE_NAME  # noqa: E402
from lunarsim.tools.episode_runner import build_agent, collect_episodes  # noqa: E402
from lunarsim.tools.replay_recording import ReplayRecorder  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCENARIO = "apollo17_terminal_truth"
SECONDS_BEFORE_CONTACT = 12.0


def read_options() -> dict[str, str]:
    """Return the ``key=value`` options with their defaults."""
    options = {"run_name": "landing_videos", "landings": "3072", "seed": "0", "successes": "5", "failures": "5", "contacts": "1"}
    for argument in script_arguments:
        key, _, value = argument.partition("=")
        if key not in options:
            raise SystemExit(f"unknown option '{key}'")
        options[key] = value
    return options


def choose_clips(episodes, successes: int, failures: int, random: numpy.random.Generator) -> list[int]:
    """Return environment indices: failures (body_contact first) and successes spread over ground relief."""
    failed = episodes[~episodes["success"]]
    order = ["body_contact"] + [name for name in failed["outcome"].value_counts().index if name != "body_contact"]
    chosen_failures: list[int] = []
    for outcome in order:
        candidates = failed[failed["outcome"] == outcome]["environment_index"].tolist()
        random.shuffle(candidates)
        chosen_failures += candidates[: failures - len(chosen_failures)]
        if len(chosen_failures) >= failures:
            break
    succeeded = episodes[episodes["success"]].sort_values("touchdown_ground_relief_m")
    positions = numpy.linspace(0, len(succeeded) - 1, successes).round().astype(int)
    return succeeded.iloc[positions]["environment_index"].tolist() + chosen_failures


def main() -> None:
    """Fly the landings, record them, pick the clips and save them."""
    options = read_options()
    count = int(options["landings"])
    overrides = [f"seed={options['seed']}"]
    scenario = load_scenario(SCENARIO, [f"num_envs={count}", *overrides])
    environment = LanderEnv(build_lander_env_cfg(scenario, device=arguments.device))
    recorder = ReplayRecorder(environment, every_n_steps=2, include_contacts=options["contacts"] == "1")
    episodes = collect_episodes(
        environment, build_agent(environment), episodes_per_environment=1,
        maximum_steps=int(1.2 * environment.max_episode_length) + 10, after_step=recorder.record,
    )
    clips = choose_clips(episodes, int(options["successes"]), int(options["failures"]), numpy.random.default_rng(int(options["seed"])))
    table = episodes.set_index("environment_index")
    titles, details, starts = [], [], []
    for number, index in enumerate(clips):
        row = table.loc[index]
        outcome = "SUCCESS" if row["outcome"] == "success" else f"FAILURE: {row['outcome']}"
        titles.append(f"{number + 1}/{len(clips)}  {outcome}   (ZEM/ZEV, true state, Apollo 17 terrain)")
        details.append(
            f"ground under footprint at contact: slope {row['touchdown_ground_slope_deg']:.1f} deg, relief "
            f"{row['touchdown_ground_relief_m']:.2f} m   peak body force {row['peak_body_contact_force_n']:.0f} N   "
            f"lowest nozzle clearance {row['minimum_nozzle_clearance_m']:.2f} m"
        )
        contact_time = float(row["touchdown_time_s"])
        starts.append(max(0.0, contact_time - SECONDS_BEFORE_CONTACT) if contact_time > 0 else 0.0)
    run_directory = REPOSITORY_ROOT / "runs" / options["run_name"]
    run_directory.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(run_directory / EPISODES_FILE_NAME, index=False)
    path = recorder.save(run_directory / "landings.npz", clips, SCENARIO, overrides, titles, details,
                         frames_per_second=25.0, notes="ZEM/ZEV landings", clip_start_times=starts)
    print(f"[videos] {len(episodes)} landings, success rate {episodes['success'].mean():.4f}")
    print(episodes["outcome"].value_counts().to_string())
    print(f"[videos] wrote {path}; render with: python scripts/render_replay.py replay={path} mode=video")
    environment.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
