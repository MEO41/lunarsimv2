# SPDX-License-Identifier: BSD-3-Clause
"""Run a multi-phase mission (S7 full mission, S5 TRN descent) phase by phase (milestone 6).

Example::

    python scripts/run_mission.py mission=s7_full_mission num_envs=16 run_name=s7_seed0

For every phase of ``configs/missions/<mission>.yaml``: build the phase's scenario (the previous
phase's gate log becomes ``start.log_path``), fly every environment to the phase's gate (or to
touchdown in the last phase), and write ``runs/<run_name>_<phase>/`` with ``manifest.yaml`` and
``episodes.parquet`` (one row per environment; the gate states for the next phase). At the end
``runs/<run_name>_mission_summary.yaml`` lists the S7 measures per phase: gate reach, dispersion
at each gate (site frame), propellant used, time, and the landing results; and the handoff check.

Script arguments: ``mission=``, ``num_envs=`` (default 16), ``run_name=``, ``seed=`` (default 0),
``from_phase=`` and ``log=`` (resume at a later phase from a saved gate log, for development).
"""

import argparse
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a LunarSim mission phase by phase.")
AppLauncher.add_app_launcher_args(parser)
arguments, script_arguments = parser.parse_known_args()
simulation_app = AppLauncher(arguments).app

import yaml  # noqa: E402

from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario  # noqa: E402
from lunarsim.slots.plant.orbit_float64 import site_from_parameters  # noqa: E402
from lunarsim.tools.episode_records import EPISODES_FILE_NAME  # noqa: E402
from lunarsim.tools.episode_runner import build_agent, collect_episodes  # noqa: E402
from lunarsim.tools.mission_phases import load_mission, merge_agent_decision_records, summarize_phase  # noqa: E402
from lunarsim.tools.run_manifest import write_manifest  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run every phase of the mission and write one run directory per phase plus the summary."""
    options = {"mission": None, "num_envs": "16", "run_name": None, "seed": "0", "from_phase": None, "log": None}
    for argument in script_arguments:
        key, _, value = argument.partition("=")
        if key not in options:
            raise SystemExit(f"unknown argument '{argument}' (mission=, num_envs=, run_name=, seed=, from_phase=, log=)")
        options[key] = value
    if not options["mission"]:
        raise SystemExit("give mission=<name> (configs/missions/<name>.yaml)")
    mission = load_mission(options["mission"])
    run_name = options["run_name"] or f"{mission['name']}_{time.strftime('%Y%m%d_%H%M%S')}"
    summaries, previous_log, handoff = [], options["log"], None
    phases = mission["phases"]
    if options["from_phase"]:
        names = [phase["name"] for phase in phases]
        if options["from_phase"] not in names or not options["log"]:
            raise SystemExit(f"from_phase must be one of {names} and needs log=<previous gate log>")
        phases = phases[names.index(options["from_phase"]):]
    for phase in phases:
        overrides = [f"env.num_envs={options['num_envs']}", f"seed={options['seed']}"] + list(phase.get("overrides") or [])
        if previous_log is not None:
            overrides.append(f"start.log_path={previous_log}")
        scenario = load_scenario(phase["scenario"], overrides)
        environment = LanderEnv(build_lander_env_cfg(scenario, device=arguments.device))
        agent = build_agent(environment)
        start_binding = environment.bindings["start"]
        if getattr(start_binding, "handoff_report", None) is not None:
            handoff = dict(start_binding.handoff_report, at=phase["name"])
            print(f"[mission] handoff check before {phase['name']}: {handoff}", flush=True)
        began = time.perf_counter()
        maximum_steps = int(1.2 * environment.max_episode_length) + 10
        episodes = collect_episodes(environment, agent, episodes_per_environment=1, maximum_steps=maximum_steps)
        episodes = merge_agent_decision_records(episodes, agent)
        elapsed = time.perf_counter() - began
        phase_directory = REPOSITORY_ROOT / "runs" / f"{run_name}_{phase['name']}"
        write_manifest(
            phase_directory,
            scenario,
            environment.random_generators.seeds(),
            extra={"mission": mission["name"], "phase": phase["name"], "previous_log": previous_log, "handoff": handoff, "wall_time_s": round(elapsed, 1)},
        )
        episodes.to_parquet(phase_directory / EPISODES_FILE_NAME, index=False)
        site = site_from_parameters((scenario.slots["plant"].parameters.get("site")) or (scenario.slots["start"].parameters.get("site")))
        summary = summarize_phase(phase["name"], episodes, site)
        summary["wall_time_s"] = round(elapsed, 1)
        summaries.append(summary)
        print(f"[mission] {phase['name']}: {summary}", flush=True)
        previous_log = str(phase_directory / EPISODES_FILE_NAME)
        environment.close()
    summary_path = REPOSITORY_ROOT / "runs" / f"{run_name}_mission_summary.yaml"
    with open(summary_path, "w") as file:
        yaml.safe_dump({"mission": mission["name"], "run_name": run_name, "handoff_check": handoff, "phases": summaries}, file, sort_keys=False)
    print(f"[mission] summary: {summary_path}", flush=True)


if __name__ == "__main__":
    main()
    simulation_app.close()
