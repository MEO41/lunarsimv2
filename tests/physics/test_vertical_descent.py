# SPDX-License-Identifier: BSD-3-Clause
"""Controller build-up step 3 (GPU): vertical descent to a soft touchdown from straight above.

16 landers start upright with no horizontal speed, at footpad altitudes from 5 m to 50 m and
vertical speeds from 0 to -2 m/s. The agent follows the descent-rate profile and stops the
engine 1 s after the contact light of the 1.727 m surface probes (Apollo probe mode). The
touchdown metric judges every landing.

Pass criteria and why:
- All 16 episodes are a success (inside the Apollo landing-gear envelope, settled, upright).
- The contact light comes on when the footpads are 1.727 m above the ground (upright lander,
  probe length), within 3 cm (20 ms sampling at about 1.2 m/s).
- Vertical speed at contact between 1.2 and 2.0 m/s. At probe height the profile asks for
  sqrt(0.6^2 + 2 * 0.8 * (1.727 - 1.0)) = 1.24 m/s; during the 1 s delay the lander slows to
  about 0.9 m/s and the pads come down to about 0.7 m. The 0.5 s thrust tail-off and the free
  drop give sqrt(v^2 + 2 g h) of about 1.6 m/s; the band allows for the tail-off thrust.
- Horizontal speed at contact below 5 cm/s and tilt below 1 degree: nothing excites the
  lateral motion in this step.
- Propellant below 250 kg: the longest descent (50 m) takes about 35 s at about 4 kg/s.
"""

import pytest
import torch
from isaac_test_support import build_agent, collect_episodes, make_environment, set_start_states

COUNT = 16


@pytest.fixture(scope="module")
def episodes():
    """Fly one vertical descent per environment and return the episode table."""
    environment = make_environment("s1_terminal_truth", [f"num_envs={COUNT}", "agent=vertical_descent"])
    agent = build_agent(environment)
    device = environment.device
    set_start_states(
        environment,
        altitude=torch.linspace(5.0, 50.0, COUNT, device=device),
        vertical_speed=torch.linspace(0.0, -2.0, COUNT, device=device),
    )
    frame = collect_episodes(environment, agent, episodes_per_environment=1, maximum_steps=5000)
    yield frame.sort_values("environment_index").reset_index(drop=True)
    environment.close()


def test_every_descent_is_a_successful_landing(episodes):
    """One record per lander, all with outcome 'success'."""
    print(episodes[["start_altitude_m", "touchdown_vertical_speed_mps", "touchdown_tilt_deg", "propellant_used_kg", "episode_duration_s", "outcome"]].to_string())
    assert len(episodes) == COUNT
    assert episodes["outcome"].tolist() == ["success"] * COUNT


def test_contact_light_comes_on_at_the_probe_length(episodes):
    """Upright landers see the light with the footpads 1.727 m above the ground, within 3 cm."""
    print(episodes["footpad_height_at_probe_contact_m"].describe().to_string())
    assert episodes["footpad_height_at_probe_contact_m"].between(1.697, 1.727 + 0.005).all()


def test_touchdown_is_soft_and_straight(episodes):
    """Contact speed 1.2-2.0 m/s down, below 5 cm/s sideways, tilt below 1 degree."""
    assert episodes["touchdown_vertical_speed_mps"].between(-2.0, -1.2).all()
    assert (episodes["touchdown_horizontal_speed_mps"] < 0.05).all()
    assert (episodes["touchdown_tilt_deg"] < 1.0).all()
    assert (episodes["final_tilt_deg"] < 0.5).all()


def test_propellant_use_is_plausible(episodes):
    """Every descent burns less than 250 kg, and higher starts burn more."""
    assert (episodes["propellant_used_kg"] < 250.0).all()
    assert episodes["propellant_used_kg"].iloc[-1] > episodes["propellant_used_kg"].iloc[0]
