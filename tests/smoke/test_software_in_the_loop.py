# SPDX-License-Identifier: BSD-3-Clause
"""S10 smoke test (GPU, milestone 7): the bridged nodes fly the same S3 start as the in-loop code.

One lander, the S3 start of seed 1000, 150 control steps (3 s): ``ekf_fdi`` and ``ppo_gru`` in the
loop, then as loopback nodes (ONNX policy). The true positions stay within 5 cm (the reference
filter is compiled on the GPU, the node's runs eagerly on the CPU, so they are close, not equal),
every reply comes back, and the slot-loop check accepts the bridge (no truth read by the nodes).

Needs ``runs/ppo_gru_s3_seed0_to1500/policy.pt`` and ``policy.onnx`` (paper 1 GRU; ``runs/`` is not in
git); skipped without them.
"""

from pathlib import Path

import pytest
import torch
from isaac_test_support import build_agent, make_environment, read_truth

import lunarsim.agents.bridge_agent  # noqa: F401  (registers agent: ros2_bridge)
import lunarsim.slots.navigation.ros2_bridge  # noqa: F401  (registers nav: ros2_bridge)
from lunarsim.tools.episode_runner import run_agent_in_environment

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
POLICY = "runs/ppo_gru_s3_seed0_to1500/policy.pt"
ONNX = "runs/ppo_gru_s3_seed0_to1500/policy.onnx"
STEPS = 150

pytestmark = pytest.mark.skipif(
    not ((REPOSITORY_ROOT / POLICY).exists() and (REPOSITORY_ROOT / ONNX).exists()),
    reason="needs the paper 1 GRU policy and its ONNX export in runs/",
)


def fly(scenario_name: str, overrides: list[str]) -> tuple[list[torch.Tensor], object, object, object]:
    """Fly STEPS control steps from the seed-1000 start; return the true positions, the agent, the nav binding and the environment."""
    environment = make_environment(scenario_name, ["env.num_envs=1", "seed=1000"] + overrides)
    agent = build_agent(environment)
    positions = []
    run_agent_in_environment(environment, agent, number_of_steps=STEPS, after_step=lambda step: positions.append(read_truth(environment).position[0].cpu().clone()))
    navigation = environment.bindings["nav"]
    return positions, agent, navigation, environment


def test_bridged_nodes_fly_like_the_code_in_the_loop():
    """Same start, same faults: the bridged run stays within 5 cm of the in-loop run for 3 s."""
    reference, _, _, environment = fly("s3_faults", ["agent=ppo_gru", f"agent.policy_path={POLICY}"])
    environment.close()
    bridged, agent, navigation, environment = fly("s10_ros2_sil", [f"agent.node.policy_path={POLICY}", f"agent.onnx_path={ONNX}"])
    agent_summary, navigation_summary = agent.deadline.summary(), navigation.deadline.summary()
    agent.close()
    navigation.close()
    environment.close()
    assert agent_summary["deadline_misses"] == 0 and navigation_summary["deadline_misses"] == 0
    assert agent_summary["replies"] >= STEPS and navigation_summary["replies"] >= STEPS
    difference = max(float((a - b).norm()) for a, b in zip(reference, bridged))
    assert difference < 0.05, difference
