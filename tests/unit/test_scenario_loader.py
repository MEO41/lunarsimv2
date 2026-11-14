# SPDX-License-Identifier: BSD-3-Clause
"""Scenario loader: load, override one slot from the command line, reject unknown names."""

import pytest

from lunarsim.core.binding import SlotName
from lunarsim.env.scenario_loader import ScenarioError, load_scenario, split_command_line


def test_null_loop_scenario_loads_with_one_binding_per_slot():
    """The milestone 0 scenario resolves all nine slots and its timing."""
    scenario = load_scenario("m0_null_loop")
    assert set(scenario.slots) == set(SlotName.ALL)
    assert scenario.slots[SlotName.NAVIGATION].binding == "truth"
    assert scenario.slots[SlotName.PLANT].parameters["mass_kg"] == 100.0
    assert scenario.control_step_duration == pytest.approx(0.02)
    assert scenario.number_of_jets == 0
    assert scenario.resolved["name"] == "m0_null_loop"


def test_command_line_overrides_change_only_what_they_name():
    """``num_envs``, ``seed`` and a slot parameter can be set; other slots stay the same."""
    base = load_scenario("m0_null_loop")
    changed = load_scenario("m0_null_loop", ["num_envs=8", "seed=3", "start.position_m=[0,0,5]"])
    assert changed.number_of_environments == 8 and changed.root_seed == 3
    assert changed.slots[SlotName.START].parameters["position_m"] == [0, 0, 5]
    for slot in SlotName.ALL:
        if slot != SlotName.START:
            assert changed.slots[slot] == base.slots[slot]


def test_slot_binding_can_be_swapped_in_one_override(tmp_path):
    """Goal G1 in small: ``metrics=<other>`` swaps one binding file for another."""
    import shutil

    from lunarsim.env.scenario_loader import default_config_directory

    config_directory = tmp_path / "configs"
    shutil.copytree(default_config_directory(), config_directory)
    (config_directory / "faults" / "other.yaml").write_text("binding: none\nlabel: swapped\n")
    scenario = load_scenario("m0_null_loop", ["faults=other"], config_directory=config_directory)
    assert scenario.slots[SlotName.FAULTS].parameters == {"label": "swapped"}


def test_unknown_names_fail_with_scenario_error():
    """Unknown scenario files, binding files and binding names all stop the run."""
    with pytest.raises(ScenarioError, match="No scenario file"):
        load_scenario("does_not_exist")
    with pytest.raises(ScenarioError, match="does not compose"):
        load_scenario("m0_null_loop", ["nav=ekf"])


def test_registered_name_is_checked(tmp_path):
    """A config file that names an unregistered binding fails the registry check."""
    import shutil

    from lunarsim.env.scenario_loader import default_config_directory

    config_directory = tmp_path / "configs"
    shutil.copytree(default_config_directory(), config_directory)
    (config_directory / "nav" / "ghost.yaml").write_text("binding: ghost\n")
    with pytest.raises(ScenarioError, match="No binding 'ghost'"):
        load_scenario("m0_null_loop", ["nav=ghost"], config_directory=config_directory)


def test_split_command_line():
    """``scenario=`` is taken out; the rest are overrides."""
    assert split_command_line(["a=1", "scenario=s1", "b=2"]) == ("s1", ["a=1", "b=2"])
    with pytest.raises(ScenarioError):
        split_command_line(["a=1"])
