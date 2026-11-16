# SPDX-License-Identifier: BSD-3-Clause
"""S7 and S5 smoke test (GPU, milestone 6): mission phases, gate logs, the low-gate handoff, TRN fixes.

Checks:

- the three orbit phases of S7 (descent orbit in MCI, braking and approach in MCMF), each with a
  short extra gate (30 s), end every lander at the gate and log float64 gate states that the
  next phase starts from;
- the low-gate handoff: an Apollo low-gate state saved in MCMF float64 starts the PhysX terminal
  phase in the SITE frame without a jump (within the ``from_log`` tolerances);
- S5: the camera TRN filter makes terrain fixes within the first 30 s from the high gate.
"""

from pathlib import Path

import pandas
import torch
from isaac_test_support import build_agent, make_environment, read_truth

from lunarsim.core.frames import convert_truth_state
from lunarsim.core.message_types import Frame, TruthState
from lunarsim.slots.metrics.phase_gate import FRAME_CODES
from lunarsim.slots.plant.orbit_float64 import site_from_parameters
from lunarsim.tools.episode_runner import collect_episodes, run_agent_in_environment

LANDERS = 4


def run_phase(scenario_name: str, overrides: list[str], log_path: Path) -> pandas.DataFrame:
    """Run one phase until every lander reaches its gate; save and return the gate log."""
    environment = make_environment(scenario_name, [f"env.num_envs={LANDERS}"] + overrides)
    agent = build_agent(environment)
    episodes = collect_episodes(environment, agent, episodes_per_environment=1, maximum_steps=int(1.2 * environment.max_episode_length) + 10)
    environment.close()
    episodes.to_parquet(log_path, index=False)
    return episodes


def test_orbit_phases_hand_their_gate_states_on(tmp_path):
    """Descent orbit -> braking -> approach, 30 s each: every lander reaches the gate; frames and float64 kept."""
    short_gate = "+metrics.until.elapsed_at_least_s=30.0"
    previous = None
    for scenario_name, frame in (("s7_phase_descent_orbit", "MCI"), ("s7_phase_braking", "MCMF"), ("s7_phase_approach", "MCMF")):
        overrides = [short_gate] + ([f"start.log_path={previous}"] if previous else [])
        log_path = tmp_path / f"{scenario_name}.parquet"
        episodes = run_phase(scenario_name, overrides, log_path)
        assert bool(episodes["gate_reached"].all()), scenario_name
        assert {FRAME_CODES[int(code)] for code in episodes["gate_frame_code"]} == {frame}
        assert episodes["gate_position_x_m"].dtype == "float64"
        # 30 s plus at most one control step.
        assert float(episodes["elapsed_s"].max()) < 30.5, scenario_name
        previous = log_path


def apollo_low_gate_log(path: Path, scenario_name: str) -> TruthState:
    """Write a gate log with the Apollo low gate (site frame) saved in MCMF float64; return the site-frame state."""
    from lunarsim.env.scenario_loader import load_scenario

    scenario = load_scenario(scenario_name, ["start.log_path=unused"])
    site = site_from_parameters(scenario.slots["start"].parameters["site"])
    count = LANDERS
    dtype = torch.float64
    # [D-6846] p. 10: low gate 2,000 ft from the site, 500 ft up, 60 fps forward, -16 fps; flying west (-x).
    site_state = TruthState(
        frame=Frame.SITE,
        time=torch.zeros(count, dtype=dtype),
        position=torch.tensor([[609.6, 0.0, 156.0]], dtype=dtype).repeat(count, 1) + torch.tensor([[0.0, 10.0, 0.0]], dtype=dtype) * torch.arange(count, dtype=dtype).unsqueeze(1),
        velocity=torch.tensor([[-18.29, 0.0, -4.877]], dtype=dtype).repeat(count, 1),
        orientation_quaternion=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=dtype).repeat(count, 1),
        angular_velocity=torch.zeros(count, 3, dtype=dtype),
        mass=torch.full((count,), 7788.0, dtype=dtype),
        contact=torch.zeros(count, 0, dtype=dtype),
    )
    saved = convert_truth_state(site_state, Frame.MOON_CENTERED_MOON_FIXED, site)
    columns = {
        "environment_index": list(range(count)),
        "gate_reached": [True] * count,
        "gate_frame_code": [FRAME_CODES.index("MCMF")] * count,
        "gate_time_s": saved.time.tolist(),
        "gate_mass_kg": saved.mass.tolist(),
    }
    for index, name in enumerate(("x", "y", "z")):
        columns[f"gate_position_{name}_m"] = saved.position[:, index].tolist()
        columns[f"gate_velocity_{name}_mps"] = saved.velocity[:, index].tolist()
        columns[f"gate_body_rate_{name}_rad_s"] = saved.angular_velocity[:, index].tolist()
    for index, name in enumerate(("w", "x", "y", "z")):
        columns[f"gate_orientation_{name}"] = saved.orientation_quaternion[:, index].tolist()
    pandas.DataFrame(columns).to_parquet(path, index=False)
    return site_state


def test_low_gate_handoff_to_physx_does_not_jump(tmp_path):
    """The terminal phase starts from a saved MCMF low-gate state at the same site-frame state."""
    log_path = tmp_path / "low_gate.parquet"
    site_state = apollo_low_gate_log(log_path, "s7_phase_terminal")
    environment = make_environment("s7_phase_terminal", [f"env.num_envs={LANDERS}", f"start.log_path={log_path}"])
    report = environment.bindings["start"].handoff_report
    assert report is not None
    assert report["position_jump_m"] < 0.01 and report["velocity_jump_mps"] < 1e-4
    environment.reset()
    truth = read_truth(environment)
    # The PhysX state right after the reset is the handed-over state (float32 near the site).
    assert torch.allclose(truth.position.cpu().double(), site_state.position, atol=0.05)
    assert torch.allclose(truth.velocity.cpu().double(), site_state.velocity, atol=0.01)
    environment.close()


def test_terrain_relative_navigation_makes_fixes():
    """S5: within 30 s from the high gate the TRN filter accepts terrain fixes."""
    environment = make_environment("s5_trn_descent", [f"env.num_envs={LANDERS}"])
    agent = build_agent(environment)
    navigation = environment.bindings["nav"]
    run_agent_in_environment(environment, agent, number_of_steps=300)
    assert bool((navigation.last_fix_time > 0.0).any())
    environment.close()
