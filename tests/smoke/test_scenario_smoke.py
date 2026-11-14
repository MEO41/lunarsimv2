# SPDX-License-Identifier: BSD-3-Clause
"""Smoke test (GPU): scenario S1 runs 100 control steps with 64 environments.

Also checks the truth-access log: only the bindings that are allowed to read TruthState did so.
"""

import pytest
import torch
from isaac_test_support import build_agent, make_environment

from lunarsim.tools.episode_runner import run_agent_in_environment

ALLOWED_TRUTH_READERS = {"TruthNavigation", "TouchdownMetrics"}


@pytest.fixture(scope="module")
def environment():
    """Build S1 with its default agent (zem_zev) and 64 environments."""
    environment = make_environment("s1_terminal_truth", ["num_envs=64"])
    yield environment
    environment.close()


def test_s1_runs_100_steps_with_64_environments(environment):
    """The closed loop runs without error and all observations stay finite."""
    agent = build_agent(environment)
    run_agent_in_environment(environment, agent, number_of_steps=100)
    observations = environment._get_observations()["policy"]
    assert observations.shape == (64, 14)
    assert torch.isfinite(observations).all()
    assert environment.cfg.action_space == 19


def test_only_allowed_bindings_read_truth(environment):
    """Runtime guard of goal G3: the bus saw TruthState reads only from nav: truth and metrics."""
    readers = set(environment.bus.truth_read_log)
    assert readers, "nobody read TruthState; the log is not working"
    assert readers <= ALLOWED_TRUTH_READERS
