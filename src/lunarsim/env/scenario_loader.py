# SPDX-License-Identifier: BSD-3-Clause
"""Load a Hydra scenario file into a plain :class:`ScenarioConfig`.

This module needs Hydra but not Isaac Lab, so it is covered by CPU tests. The Isaac Lab config
class is built from the ScenarioConfig in ``lander_env_cfg.py``.

A scenario file lists one binding per slot::

    # configs/scenarios/s1_terminal_truth.yaml
    defaults:
      - /start: terminal_box
      - /nav: truth
      ...

Any slot can be changed on the command line: ``scenario=s1_terminal_truth agent=zem_zev seed=3``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from lunarsim.core.binding import SlotName
from lunarsim.core.registry import check_slot_loop, get_binding_class

CONFIG_DIRECTORY_ENVIRONMENT_VARIABLE = "LUNARSIM_CONFIG_DIR"
# Short command-line names for values that live under ``env:`` in the scenario file.
ENVIRONMENT_OVERRIDE_SHORTCUTS = ("num_envs",)


class ScenarioError(ValueError):
    """The scenario file is incomplete or names something that does not exist."""


def default_config_directory() -> Path:
    """Return the ``configs`` directory: the environment variable if set, else the repo's folder."""
    from_environment = os.environ.get(CONFIG_DIRECTORY_ENVIRONMENT_VARIABLE)
    if from_environment:
        return Path(from_environment).resolve()
    return Path(__file__).resolve().parents[3] / "configs"


@dataclass
class SlotSelection:
    """The binding chosen for one slot, with its parameters from the scenario file."""

    binding: str
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioConfig:
    """A fully resolved scenario."""

    name: str
    root_seed: int
    number_of_environments: int
    environment_spacing: float
    physics_step_duration: float
    physics_steps_per_control_step: int
    episode_length_seconds: float
    slots: dict[str, SlotSelection]
    lander: dict[str, Any] | None
    resolved: dict[str, Any]  # the complete resolved config, saved in the run manifest

    @property
    def control_step_duration(self) -> float:
        """Return the length of one control step in seconds."""
        return self.physics_step_duration * self.physics_steps_per_control_step

    @property
    def number_of_jets(self) -> int:
        """Return the number of reaction-control jets of the lander (0 without a lander)."""
        if not self.lander:
            return 0
        return len(self.lander["reaction_control_system"]["jets"])

    def binding_classes(self) -> dict[str, type]:
        """Return the registered binding class of every slot."""
        import lunarsim.agents  # noqa: F401  (registers the agents)
        import lunarsim.slots  # noqa: F401  (registers the bindings)

        return {slot: get_binding_class(slot, selection.binding) for slot, selection in self.slots.items()}

    def validate(self) -> None:
        """Run the registry checks: known binding names, closed loop, no truth leak."""
        check_slot_loop(self.binding_classes())

    @classmethod
    def from_resolved(cls, resolved: dict[str, Any]) -> ScenarioConfig:
        """Build a ScenarioConfig from a resolved plain dictionary (as stored in a manifest)."""
        missing_slots = [slot for slot in SlotName.ALL if slot not in resolved]
        if missing_slots:
            raise ScenarioError(f"Scenario '{resolved.get('name', '?')}' has no binding for slots {missing_slots}")
        slots = {}
        for slot in SlotName.ALL:
            entry = dict(resolved[slot])
            if "binding" not in entry:
                raise ScenarioError(f"Slot '{slot}' has parameters but no 'binding' name")
            slots[slot] = SlotSelection(binding=str(entry.pop("binding")), parameters=entry)
        environment = resolved.get("env", {})
        try:
            return cls(
                name=str(resolved["name"]),
                root_seed=int(resolved.get("seed", 0)),
                number_of_environments=int(environment["num_envs"]),
                environment_spacing=float(environment.get("env_spacing_m", 20.0)),
                physics_step_duration=float(environment["physics_step_s"]),
                physics_steps_per_control_step=int(environment["physics_steps_per_control_step"]),
                episode_length_seconds=float(environment["episode_length_s"]),
                slots=slots,
                lander=resolved.get("lander"),
                resolved=resolved,
            )
        except KeyError as error:
            raise ScenarioError(f"Scenario '{resolved.get('name', '?')}' misses the key {error}") from error


def load_scenario(
    scenario_name: str,
    overrides: list[str] | None = None,
    config_directory: Path | str | None = None,
) -> ScenarioConfig:
    """Compose ``configs/scenarios/<scenario_name>.yaml`` with Hydra and validate it.

    ``overrides`` are Hydra overrides such as ``["agent=zem_zev", "seed=3", "num_envs=64"]``.
    """
    directory = Path(config_directory).resolve() if config_directory else default_config_directory()
    if not (directory / "scenarios" / f"{scenario_name}.yaml").is_file():
        raise ScenarioError(f"No scenario file {directory / 'scenarios' / (scenario_name + '.yaml')}")
    hydra_overrides = []
    for override in overrides or []:
        key = override.split("=", 1)[0]
        hydra_overrides.append(f"env.{override}" if key in ENVIRONMENT_OVERRIDE_SHORTCUTS else override)

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(directory), version_base=None):
            composed = compose(config_name=f"scenarios/{scenario_name}", overrides=hydra_overrides)
    except Exception as error:  # Hydra raises many types; all mean "this scenario does not compose"
        raise ScenarioError(f"Scenario '{scenario_name}' does not compose: {error}") from error
    finally:
        GlobalHydra.instance().clear()

    resolved = OmegaConf.to_container(composed, resolve=True)
    resolved.setdefault("name", scenario_name)
    scenario = ScenarioConfig.from_resolved(resolved)
    try:
        scenario.validate()
    except ValueError as error:
        raise ScenarioError(f"Scenario '{scenario_name}' is not valid: {error}") from error
    return scenario


def split_command_line(arguments: list[str]) -> tuple[str, list[str]]:
    """Split ``["scenario=name", "agent=x", ...]`` into the scenario name and the other overrides."""
    scenario_name = None
    overrides = []
    for argument in arguments:
        if argument.startswith("scenario="):
            scenario_name = argument.split("=", 1)[1]
        else:
            overrides.append(argument)
    if scenario_name is None:
        raise ScenarioError("Give the scenario on the command line: scenario=<name>")
    return scenario_name, overrides
