# SPDX-License-Identifier: BSD-3-Clause
"""Run manifest and episode records: written to and read back from a run directory."""

import torch

from lunarsim.core.binding import SlotName
from lunarsim.core.random_generators import SlotRandomGenerators
from lunarsim.env.scenario_loader import load_scenario
from lunarsim.tools.episode_records import EpisodeRecordWriter, read_episode_records
from lunarsim.tools.run_manifest import read_manifest, write_manifest


def test_manifest_holds_config_git_hash_seeds_and_versions(tmp_path):
    """The manifest has every item the design lists."""
    scenario = load_scenario("s1_terminal_truth", ["seed=7"])
    seeds = SlotRandomGenerators(scenario.root_seed, SlotName.ALL).seeds()
    write_manifest(tmp_path, scenario, seeds, extra={"note": "unit test"})
    manifest = read_manifest(tmp_path)
    assert manifest["root_seed"] == 7 and manifest["scenario_name"] == "s1_terminal_truth"
    assert manifest["slot_seeds"] == seeds
    assert len(manifest["git"]["commit"]) >= 7
    assert set(manifest["versions"]) >= {"isaac_sim", "isaac_lab", "torch", "python", "lunarsim"}
    assert "gpu_name" in manifest
    assert manifest["resolved_scenario"]["nav"]["binding"] == "truth"
    assert manifest["resolved_scenario"]["lander"]["mass"]["mass_at_start_kg"]["value"] == 7327.0


def test_episode_records_round_trip(tmp_path):
    """Records appended in two batches come back as one table with an outcome name column."""
    writer = EpisodeRecordWriter(tmp_path, outcome_names=("success", "hard_vertical"))
    writer.append({"outcome_code": torch.tensor([0, 1]), "touchdown_vertical_speed_mps": torch.tensor([-1.0, -4.0])})
    writer.append({})
    writer.append({"outcome_code": torch.tensor([0]), "touchdown_vertical_speed_mps": torch.tensor([-0.5])})
    assert writer.number_of_episodes == 3
    writer.write()
    frame = read_episode_records(tmp_path)
    assert frame["outcome"].tolist() == ["success", "hard_vertical", "success"]
    assert frame["touchdown_vertical_speed_mps"].tolist() == [-1.0, -4.0, -0.5]
