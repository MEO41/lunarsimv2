# SPDX-License-Identifier: BSD-3-Clause
"""PPO smoke test (GPU, milestone 1b): train two iterations, export, fly the exported policy.

Checks the whole learned-agent path without judging its quality:
- the scenario with ``agent=ppo_mlp training=landing_ppo`` builds, has a reward and a ``critic``
  group from TruthState, and the actor's ``policy`` group keeps its 16 values;
- rsl_rl's OnPolicyRunner runs two iterations on the vector environment, and the rewards are finite;
- the exported actor loads in the ``ppo_mlp`` agent and gives valid commands;
- truth leak check: only ``nav: truth``, the metrics and the training-side reader read TruthState.
"""

import pytest
import torch
from isaac_test_support import build_agent, make_environment, step_with_agent

from lunarsim.core.message_types import NavigationStateLayout, Observation

COUNT = 64


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train two iterations and export the actor; yield the environment and the policy path."""
    from rsl_rl.runners import OnPolicyRunner

    from lunarsim.agents.rsl_rl.lander_vector_environment import LanderVectorEnvironment
    from lunarsim.agents.rsl_rl.training_configuration import export_policy, runner_configuration

    directory = tmp_path_factory.mktemp("ppo_smoke")
    environment = make_environment("s1_terminal_truth", [f"num_envs={COUNT}", "agent=ppo_mlp", "training=landing_ppo"])
    parameters = environment.scenario.slots["agent"].parameters
    vector_environment = LanderVectorEnvironment(environment, parameters)
    configuration = runner_configuration(parameters["ppo"], 0, "smoke", has_critic_group=True)
    configuration["max_iterations"] = 2
    runner = OnPolicyRunner(vector_environment, configuration, log_dir=None, device="cuda:0")
    runner.learn(num_learning_iterations=2)
    policy_path = directory / "policy.pt"
    export_policy(runner.alg.actor, policy_path)
    yield environment, vector_environment, policy_path
    environment.close()


def test_observation_groups_and_reward(trained):
    """policy has 16 values, critic has 16 truth values, and one step gives finite rewards."""
    environment, vector_environment, _ = trained
    observations = vector_environment.get_observations()
    assert observations["policy"].shape == (COUNT, NavigationStateLayout.DIMENSION)
    assert observations["critic"].shape == (COUNT, NavigationStateLayout.DIMENSION)
    _, rewards, dones, extras = vector_environment.step(torch.zeros(COUNT, vector_environment.num_actions, device="cuda:0"))
    assert bool(torch.isfinite(rewards).all())
    assert "time_outs" in extras


def test_exported_policy_flies_in_the_agent(trained):
    """The ppo_mlp agent loads the export and returns commands inside their ranges."""
    environment, _, policy_path = trained
    environment.scenario.slots["agent"].parameters["policy_path"] = str(policy_path)
    agent = build_agent(environment)
    with torch.inference_mode():  # the environment's tensors are inference tensors after training
        observations, _ = environment.reset()
        command = agent.compute_command(Observation(policy=observations["policy"]), environment.clock.control_step_duration)
    assert bool(((command.throttle >= 0.0) & (command.throttle <= 0.65)).all())
    step_with_agent(environment, agent, observations, 20)


def test_only_allowed_readers_read_the_truth(trained):
    """Truth reads come from nav: truth, the metrics and the training-side reader only."""
    environment, _, _ = trained
    assert set(environment.bus.truth_read_log) <= {"TruthNavigation", "TouchdownMetrics", "_TrainingTruthReader"}
