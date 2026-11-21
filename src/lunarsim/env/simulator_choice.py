# SPDX-License-Identifier: BSD-3-Clause
"""Pick the environment class of a scenario: LanderEnv (Isaac Sim, PhysX) or AnalyticLanderEnvironment (pure torch).

A script must know before it starts Isaac Sim whether it needs Isaac Sim at all, and Isaac Sim
should start before torch and the LunarSim slot code are imported. So
:func:`command_line_uses_analytic_simulator` reads only the command line and the scenario file
(plain YAML) and imports nothing heavy. After the scenario is loaded,
:func:`check_simulator_choice` confirms the guess and :func:`build_environment` builds the
environment.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

# Plant bindings that run in AnalyticLanderEnvironment (their class has ``simulator = "analytic"``).
ANALYTIC_PLANT_BINDINGS = ("analytic_rigid_body", "analytic_point_mass")
CONFIG_DIRECTORY_ENVIRONMENT_VARIABLE = "LUNARSIM_CONFIG_DIR"


def _config_directory() -> Path:
    """Return the ``configs`` directory (same rule as scenario_loader.default_config_directory)."""
    from_environment = os.environ.get(CONFIG_DIRECTORY_ENVIRONMENT_VARIABLE)
    if from_environment:
        return Path(from_environment).resolve()
    return Path(__file__).resolve().parents[3] / "configs"


def plant_binding_on_command_line(arguments: list[str], config_directory: Path | str | None = None) -> str | None:
    """Return the plant binding that ``arguments`` select, or None when it cannot be told without Hydra.

    A ``plant=<name>`` override wins. Otherwise the ``/plant: <name>`` entry of the scenario
    file's defaults list is used.
    """
    scenario_name = None
    for argument in arguments:
        key, _, value = argument.partition("=")
        if key == "plant":
            return value
        if key == "scenario":
            scenario_name = value
    if scenario_name is None:
        return None
    directory = Path(config_directory) if config_directory else _config_directory()
    scenario_file = directory / "scenarios" / f"{scenario_name}.yaml"
    if not scenario_file.is_file():
        return None
    with open(scenario_file, encoding="utf-8") as file:
        content = yaml.safe_load(file) or {}
    for entry in content.get("defaults", []) or []:
        if isinstance(entry, dict):
            for key, value in entry.items():
                if key.replace("override", "").strip().lstrip("/") == "plant":
                    return str(value)
    return None


def command_line_uses_analytic_simulator(arguments: list[str], config_directory: Path | str | None = None) -> bool:
    """Return True when the command line selects an analytic plant (then Isaac Sim is not started)."""
    return plant_binding_on_command_line(arguments, config_directory) in ANALYTIC_PLANT_BINDINGS


def scenario_uses_analytic_simulator(scenario: Any) -> bool:
    """Return True when the loaded scenario's plant class runs in the analytic environment."""
    from lunarsim.core.binding import SlotName

    plant_class = scenario.binding_classes()[SlotName.PLANT]
    return getattr(plant_class, "simulator", "isaac") == "analytic"


def check_simulator_choice(scenario: Any, analytic_was_chosen: bool) -> None:
    """Raise if the loaded scenario needs a different simulator than the one the script started."""
    analytic_needed = scenario_uses_analytic_simulator(scenario)
    if analytic_needed != analytic_was_chosen:
        plant = scenario.slots["plant"].binding
        raise SystemExit(
            f"The scenario's plant is '{plant}', but the script guessed a different simulator from the "
            f"command line. Name the plant on the command line: plant={plant}"
        )


def build_environment(scenario: Any, device: str = "cuda:0"):
    """Return the environment for ``scenario``: AnalyticLanderEnvironment or LanderEnv (Isaac Sim must run)."""
    if scenario_uses_analytic_simulator(scenario):
        from lunarsim.env.analytic_lander_environment import AnalyticLanderEnvironment

        return AnalyticLanderEnvironment(scenario, device=device)
    from lunarsim.env.lander_env import LanderEnv
    from lunarsim.env.lander_env_cfg import build_lander_env_cfg

    return LanderEnv(build_lander_env_cfg(scenario, device=device))
