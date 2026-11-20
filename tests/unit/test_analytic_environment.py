# SPDX-License-Identifier: BSD-3-Clause
"""The analytic environment (CPU): slot loop, start states, closed-loop landing, PPO path, simulator choice.

Part of test T1 of the analytic-to-PhysX study: the analytic environment must run the same
scenario, bindings and training code as the PhysX environment, with only the plant changed.
"""

import numpy
import pytest
import torch
from analytic_environment_support import make_actions, make_environment, read_truth, set_start_states

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey, NavigationStateLayout, Observation
from lunarsim.env.analytic_lander_environment import AnalyticLanderEnvironment
from lunarsim.env.scenario_loader import load_scenario
from lunarsim.env.simulator_choice import (
    build_environment,
    command_line_uses_analytic_simulator,
    plant_binding_on_command_line,
    scenario_uses_analytic_simulator,
)
from lunarsim.slots.metrics.touchdown import OUTCOME_CODE
from lunarsim.tools.episode_runner import build_agent, collect_episodes


class _StartStateProbe:
    """A test-only reader of the start slot's output."""

    requires = frozenset({MessageKey.START_STATE})
    provides = frozenset()
    reads_truth = False


# ---- interface ----


def test_reset_and_step_return_the_lander_env_shapes():
    """reset gives a 16-value policy group; step gives observations, rewards and done flags of length N."""
    environment = make_environment("s1_terminal_truth", ["num_envs=8"])
    observations, extras = environment.reset()
    assert observations["policy"].shape == (8, NavigationStateLayout.DIMENSION)
    assert isinstance(extras, dict)
    observations, rewards, terminated, truncated, _ = environment.step(make_actions(environment, throttle=0.3))
    assert observations["policy"].shape == (8, NavigationStateLayout.DIMENSION)
    assert rewards.shape == (8,) and terminated.dtype == torch.bool and truncated.dtype == torch.bool
    assert environment.max_episode_length == 90 * 50
    assert environment.cfg.action_space == 3 + environment.bindings[SlotName.PLANT].preset.number_of_jets


def test_the_plant_starts_where_the_start_slot_says():
    """After a reset the truth state equals the sampled start state (position, velocity, attitude, mass)."""
    environment = make_environment("s1_terminal_truth", ["num_envs=16"])
    environment.reset()
    start = environment.bus.read(MessageKey.START_STATE, _StartStateProbe())
    truth = read_truth(environment)
    assert torch.allclose(truth.position, start.position.to(torch.float32))
    assert torch.allclose(truth.velocity, start.velocity.to(torch.float32))
    assert torch.allclose(truth.orientation_quaternion, start.orientation_quaternion.to(torch.float32))
    preset = environment.bindings[SlotName.PLANT].preset
    assert torch.allclose(truth.mass, torch.full_like(truth.mass, preset.mass_at_start))


def test_the_same_seed_gives_the_same_starts_and_the_same_flight():
    """Two environments built from the same scenario and seed fly identically (needed for paired tests)."""
    histories = []
    for _ in range(2):
        environment = make_environment("s1_terminal_truth", ["num_envs=4", "seed=7"])
        environment.reset()
        for _ in range(50):
            environment.step(make_actions(environment, throttle=0.25))
        histories.append(read_truth(environment).position.clone())
    assert torch.equal(histories[0], histories[1])


def test_the_scenario_must_name_an_analytic_plant_and_a_flat_surface():
    """A PhysX plant or a terrain surface is refused with a plain message."""
    with pytest.raises(ValueError, match="analytic plants only"):
        AnalyticLanderEnvironment(load_scenario("s1_terminal_truth", ["num_envs=2"]), device="cpu")


# ---- closed-loop landing ----

NOMINAL_COUNT = 64
NOMINAL_START = ["start.altitude_m=[20,50]", "start.v_vertical_mps=[-3,0]"]


@pytest.fixture(scope="module")
def nominal_episodes():
    """Fly one ZEM/ZEV landing per environment from the nominal starts of the PhysX test."""
    environment = make_environment("s1_terminal_truth", [f"num_envs={NOMINAL_COUNT}", "agent=zem_zev", *NOMINAL_START])
    agent = build_agent(environment)
    frame = collect_episodes(environment, agent, episodes_per_environment=1, maximum_steps=6000)
    return frame, environment


def test_zem_zev_lands_every_nominal_start(nominal_episodes):
    """Every nominal start lands inside the limits, settles and stays upright (as with PhysX)."""
    frame, _ = nominal_episodes
    print(frame["outcome"].value_counts().to_string())
    assert len(frame) == NOMINAL_COUNT
    assert (frame["outcome"] == "success").all()


def test_zem_zev_touchdown_has_margin_and_hits_the_target(nominal_episodes):
    """Contact speeds, tilt, propellant and the miss distance stay inside the PhysX test's bounds."""
    frame, _ = nominal_episodes
    assert frame["touchdown_vertical_speed_mps"].between(-2.0, -0.5).all()
    assert (frame["touchdown_horizontal_speed_mps"] < 0.6).all()
    assert (frame["touchdown_tilt_deg"] < 3.0).all()
    assert (frame["propellant_used_kg"] < 250.0).all()
    miss = numpy.hypot(frame["touchdown_position_x_m"], frame["touchdown_position_y_m"])
    assert (miss < 3.0).all()


def test_only_navigation_and_metrics_read_the_truth(nominal_episodes):
    """Truth leak check: no agent, sensor or actuator reads TruthState."""
    _, environment = nominal_episodes
    assert set(environment.bus.truth_read_log) <= {"TruthNavigation", "TouchdownMetrics", "_TrainingTruthReader"}


# ---- touchdown metric: end at the first contact ----


def test_end_at_first_contact_judges_the_approach_and_stops():
    """With end_at_first_contact the episode ends in the contact step; a soft touchdown is a success, a hard one is not."""
    environment = make_environment(
        "s1_terminal_truth", ["num_envs=2", "actuators.initial_throttle=0.0", "metrics.end_at_first_contact=true"]
    )
    vertical_speed = torch.tensor([-1.0, -4.0])
    set_start_states(environment, altitude=0.05, vertical_speed=vertical_speed)
    outcome = torch.full((2,), -1, dtype=torch.long)
    steps_taken = 0
    while steps_taken < 100 and not bool((outcome >= 0).all()):
        _, _, terminated, _, _ = environment.step(make_actions(environment))
        events = environment.bus.read(MessageKey.EPISODE_EVENTS, environment._episode_events_reader)
        outcome = torch.where(terminated & (outcome < 0), events.outcome_code, outcome)
        steps_taken += 1
    assert steps_taken < 10, "the episodes should end within a few control steps of the contact"
    assert int(outcome[0]) == OUTCOME_CODE["success"]
    assert int(outcome[1]) == OUTCOME_CODE["hard_vertical"]


# ---- the PPO path without Isaac Sim ----


def test_ppo_trains_two_iterations_and_the_export_flies(tmp_path):
    """rsl_rl's runner trains on the analytic environment; the exported actor flies in the ppo_mlp agent."""
    from rsl_rl.runners import OnPolicyRunner

    from lunarsim.agents.rsl_rl.lander_vector_environment import LanderVectorEnvironment
    from lunarsim.agents.rsl_rl.training_configuration import export_policy, runner_configuration

    environment = make_environment("s1_terminal_truth", ["num_envs=16", "agent=ppo_mlp", "training=landing_ppo"])
    parameters = environment.scenario.slots["agent"].parameters
    vector_environment = LanderVectorEnvironment(environment, parameters)
    observations = vector_environment.get_observations()
    assert observations["critic"].shape == (16, NavigationStateLayout.DIMENSION)
    configuration = runner_configuration(parameters["ppo"], 0, "smoke", has_critic_group=True)
    configuration["max_iterations"] = 2
    runner = OnPolicyRunner(vector_environment, configuration, log_dir=None, device="cpu")
    runner.learn(num_learning_iterations=2)
    policy_path = tmp_path / "policy.pt"
    export_policy(runner.alg.actor, policy_path)

    environment.scenario.slots["agent"].parameters["policy_path"] = str(policy_path)
    agent = build_agent(environment)
    with torch.inference_mode():
        observations, _ = environment.reset()
        for _ in range(20):
            command = agent.compute_command(Observation(policy=observations["policy"]), environment.clock.control_step_duration)
            observations, rewards, _, _, _ = environment.step(command.to_action_tensor())
    assert bool(torch.isfinite(rewards).all())
    assert set(environment.bus.truth_read_log) <= {"TruthNavigation", "TouchdownMetrics", "_TrainingTruthReader"}


# ---- which simulator a command line needs ----


def test_the_command_line_tells_which_simulator_to_start():
    """plant=analytic_rigid_body selects the analytic simulator; S1 alone selects PhysX (from its defaults list)."""
    assert command_line_uses_analytic_simulator(["scenario=s1_terminal_truth", "plant=analytic_rigid_body"])
    assert not command_line_uses_analytic_simulator(["scenario=s1_terminal_truth", "num_envs=4"])
    assert plant_binding_on_command_line(["scenario=s1_terminal_truth"]) == "physx_site"
    assert plant_binding_on_command_line(["num_envs=4"]) is None


def test_build_environment_returns_the_analytic_environment():
    """build_environment picks AnalyticLanderEnvironment for an analytic plant."""
    scenario = load_scenario("s1_terminal_truth", ["num_envs=2", "plant=analytic_rigid_body"])
    assert scenario_uses_analytic_simulator(scenario)
    assert isinstance(build_environment(scenario, device="cpu"), AnalyticLanderEnvironment)
