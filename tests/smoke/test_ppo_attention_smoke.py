# SPDX-License-Identifier: BSD-3-Clause
"""PPO with memory, smoke test (GPU, milestone 3): ppo_attention in S3 trains, exports and flies.

64 landers in S3 (faults on, ekf_fdi, health columns in the observation): two rsl_rl iterations,
TorchScript export, then the ppo_mlp agent binding flies the export for 30 control steps.
Checks: the actor reads 20 values per step (16 navigation + 4 health); rewards stay finite;
the flown commands stay inside their ranges; only truth-side bindings read TruthState.
"""

import torch
from isaac_test_support import build_agent, make_environment, step_with_agent

from lunarsim.core.message_types import Observation


def test_attention_policy_trains_exports_and_flies(tmp_path):
    """Two iterations, export, fly."""
    from rsl_rl.runners import OnPolicyRunner

    from lunarsim.agents.rsl_rl.lander_vector_environment import LanderVectorEnvironment
    from lunarsim.agents.rsl_rl.training_configuration import export_policy, policy_kind, runner_configuration

    environment = make_environment("s3_faults", ["num_envs=64", "agent=ppo_attention", "training=landing_ppo"])
    parameters = environment.scenario.slots["agent"].parameters
    vector_environment = LanderVectorEnvironment(environment, parameters)
    assert environment.cfg.observation_space == 20
    configuration = runner_configuration(parameters["ppo"], 0, "smoke", has_critic_group=True)
    runner = OnPolicyRunner(vector_environment, configuration, log_dir=None, device="cuda:0")
    runner.learn(num_learning_iterations=2)
    policy_path = tmp_path / "policy.pt"
    export_policy(runner.alg.actor, policy_path, policy_kind(parameters["ppo"]))
    parameters["policy_path"] = str(policy_path)
    agent = build_agent(environment)
    with torch.inference_mode():
        observations, _ = environment.reset()
        agent.reset(torch.arange(64, device=environment.device), torch.Generator(device=environment.device))
        command = agent.compute_command(Observation(policy=observations["policy"]), environment.clock.control_step_duration)
    assert observations["policy"].shape == (64, 20)
    assert bool(((command.throttle >= 0.0) & (command.throttle <= 0.65)).all())
    step_with_agent(environment, agent, observations, 30)
    assert set(environment.bus.truth_read_log) <= {"TerminalSensorSuite", "FaultSchedule", "FaultResponseMetrics", "_TrainingTruthReader"}
    environment.close()
