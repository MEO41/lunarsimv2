# SPDX-License-Identifier: BSD-3-Clause
"""Record chosen landings of a trained PPO policy in PhysX for a video (analytic-to-PhysX transfer study).

Usage::

    python scripts/record_policy_landings.py policy=runs/transfer_gate_analytic_seed0/policy.pt \\
        run_name=video_t5_analytic outcomes=success:2 label="Trained in the analytic simulator"
    python scripts/render_replay.py replay=runs/video_t5_analytic/landings.npz mode=video

The policy flies ``num_envs`` episodes of ``s1_terminal_truth`` in PhysX with the held-out starts
of ``seed`` (defaults 4096 and 1000: the same starts as every evaluation of the study, so the clips
are flights that the study's success rates count). Every lander is recorded at 25 frames per
second until its first episode ends (``lunarsim.tools.replay_recording``). The clips are chosen
by outcome: ``outcomes=success:2,not_settled:1`` keeps two successes and one ``not_settled``
landing, drawn at random (generator seed ``seed``) among the landers with that outcome. A clip
longer than ``maximum_clip_s`` (default 25 s) starts that long before its end, so a lander that
hovers until its propellant runs out shows the end of the flight.

Options (``key=value``): ``policy`` (required), ``run_name`` (default ``policy_landings``),
``outcomes`` (``success:2``), ``label`` (the first part of every clip title), ``num_envs``
(4096), ``seed`` (1000), ``maximum_clip_s`` (25). Any other ``key=value`` is a scenario override.
Writes ``runs/<run_name>/landings.npz`` (for ``scripts/render_replay.py``) and ``episodes.parquet``.
Adapted from ``scripts/record_landing_videos.py`` (ZEM/ZEV on the Apollo 17 terrain).
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record chosen landings of a PPO policy in PhysX.")
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
SCENARIO = "s1_terminal_truth"
RECORD_EVERY_N_STEPS = 2
SCRIPT_OPTIONS = {
    "policy": "",
    "run_name": "policy_landings",
    "outcomes": "success:2",
    "label": "PPO policy",
    "num_envs": "4096",
    "seed": "1000",
    "maximum_clip_s": "25",
}


def read_options() -> tuple[dict[str, str], list[str]]:
    """Return the script options with their defaults and the remaining scenario overrides."""
    options = dict(SCRIPT_OPTIONS)
    overrides = []
    for argument in script_arguments:
        key, _, value = argument.partition("=")
        if key in options:
            options[key] = value
        else:
            overrides.append(argument)
    if not options["policy"]:
        raise SystemExit("policy=<path to policy.pt> is required")
    return options, overrides


def parse_outcome_counts(text: str) -> list[tuple[str, int]]:
    """Turn ``success:2,not_settled:1`` into [("success", 2), ("not_settled", 1)]."""
    counts = []
    for part in text.split(","):
        name, _, count = part.partition(":")
        counts.append((name.strip(), int(count or 1)))
    return counts


def choose_clips(episodes, outcome_counts: list[tuple[str, int]], random: numpy.random.Generator) -> list[int]:
    """Return environment indices: the asked number of landers of each outcome, drawn at random."""
    chosen: list[int] = []
    for outcome, count in outcome_counts:
        candidates = episodes[episodes["outcome"] == outcome]["environment_index"].to_numpy()
        if len(candidates) < count:
            print(f"[videos] only {len(candidates)} landers with outcome '{outcome}' (asked for {count})")
        chosen += [int(index) for index in random.permutation(candidates)[:count]]
    return chosen


def main() -> None:
    """Fly the policy on the held-out starts, record every lander, pick the clips and save them."""
    options, extra_overrides = read_options()
    overrides = [
        "agent=ppo_mlp",
        f"agent.policy_path={options['policy']}",
        f"seed={options['seed']}",
        f"num_envs={options['num_envs']}",
        *extra_overrides,
    ]
    scenario = load_scenario(SCENARIO, overrides)
    environment = LanderEnv(build_lander_env_cfg(scenario, device=arguments.device))
    recorder = ReplayRecorder(environment, every_n_steps=RECORD_EVERY_N_STEPS, include_contacts=True)
    episodes = collect_episodes(
        environment,
        build_agent(environment),
        episodes_per_environment=1,
        maximum_steps=int(1.2 * environment.max_episode_length) + 10,
        after_step=recorder.record,
    )
    clips = choose_clips(
        episodes, parse_outcome_counts(options["outcomes"]), numpy.random.default_rng(int(options["seed"]))
    )
    table = episodes.set_index("environment_index")
    maximum_clip = float(options["maximum_clip_s"])
    titles, details, starts = [], [], []
    for number, index in enumerate(clips):
        row = table.loc[index]
        outcome = "SUCCESS" if row["outcome"] == "success" else f"FAILURE: {row['outcome']}"
        titles.append(f"{number + 1}/{len(clips)}  {options['label']}, flown in PhysX: {outcome}")
        if bool(row["had_contact"]):
            details.append(
                f"at first contact: vertical {row['touchdown_vertical_speed_mps']:.2f} m/s, horizontal "
                f"{row['touchdown_horizontal_speed_mps']:.2f} m/s, tilt {row['touchdown_tilt_deg']:.1f} deg, "
                f"tilt rate {row['touchdown_tilt_rate_deg_per_s']:.1f} deg/s   propellant used {row['propellant_used_kg']:.0f} kg"
            )
        else:
            details.append(f"no ground contact   propellant used {row['propellant_used_kg']:.0f} kg")
        starts.append(max(0.0, float(row["episode_duration_s"]) - maximum_clip))
    frames_per_second = 1.0 / (RECORD_EVERY_N_STEPS * environment.step_dt)
    run_directory = REPOSITORY_ROOT / "runs" / options["run_name"]
    run_directory.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(run_directory / EPISODES_FILE_NAME, index=False)
    path = recorder.save(
        run_directory / "landings.npz",
        clips,
        SCENARIO,
        overrides,
        titles,
        details,
        frames_per_second=frames_per_second,
        notes=f"{options['label']}; policy {options['policy']}",
        clip_start_times=starts,
    )
    print(f"[videos] {len(episodes)} landings, success rate {episodes['success'].mean():.4f}")
    print(episodes["outcome"].value_counts().to_string())
    print(f"[videos] wrote {path}; render with: python scripts/render_replay.py replay={path} mode=video")
    environment.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
