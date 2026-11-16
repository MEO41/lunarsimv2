# SPDX-License-Identifier: BSD-3-Clause
"""Software-in-the-loop bridge (CPU, milestone 7): exact parity of a bridged node, deadline holds, ONNX export.

- ``nav: ros2_bridge`` running ``ekf_fdi`` in a loopback node gives exactly the estimates of
  ``ekf_fdi`` run in this process, on the synthetic flight of ``test_ekf_consistency`` (same code,
  same device, so equality is exact).
- With an added latency above the timeout, every reply misses its deadline; the first one is used
  (there is no estimate to hold yet) and the bridge keeps it.
- ``agent: ros2_bridge`` running ``zem_zev`` gives exactly the in-process commands.
- A small GRU policy exported to ONNX gives the TorchScript outputs to 1e-5.
"""

import math

import torch
import yaml
from helpers import make_binding_context
from test_ekf_consistency import CONFIG, CONTROL_STEP, COUNT, WORLD, load, truth_at

from lunarsim.agents.bridge_agent import AgentBridge
from lunarsim.agents.classical.zero_effort_miss_zero_effort_velocity_agent import ZeroEffortMissZeroEffortVelocityAgent
from lunarsim.assets.lander_preset import load_lander_preset
from lunarsim.bridge.onnx_policy import OnnxPolicy, export_policy_to_onnx
from lunarsim.core.message_types import MessageKey, NavigationStateLayout, Observation, SurfaceGeometry
from lunarsim.slots.navigation.ekf_fdi import FaultDetectingKalmanFilterNavigation
from lunarsim.slots.navigation.ros2_bridge import NavigationBridge
from lunarsim.slots.sensors.terminal_suite import TerminalSensorSuite
from lunarsim.slots.start.initial_navigation_error import InitialNavigationError

LANDERS = COUNT  # the synthetic flight has 200 landers
STEPS = 60


class _EstimateReader:
    """Test-side reader of the navigation estimate."""

    requires = frozenset({MessageKey.NAVIGATION_ESTIMATE})
    provides = frozenset()
    reads_truth = False


def navigation_parameters() -> dict:
    """Return the ekf_fdi parameters as the scenario composes them (ekf plus detection), eager on the CPU."""
    with open(CONFIG / "nav" / "ekf.yaml") as file:
        parameters = yaml.safe_load(file)
    parameters["detection"] = load("nav/ekf_fdi.yaml")["detection"]
    parameters["compile"] = False
    parameters.pop("binding", None)
    return parameters


def fly_filter(make_navigation) -> tuple[list[torch.Tensor], object]:
    """Fly the synthetic trajectory through the sensor suite and a navigation binding; return the estimate means."""
    context = make_binding_context(LANDERS)
    preset = load_lander_preset(CONFIG / "landers" / "apollo_lunar_module.yaml")
    suite_parameters = load("sensors/terminal_suite.yaml")
    context.shared_data.update({"lander_preset": preset, "sensor_suite_parameters": suite_parameters, "control_step_duration": CONTROL_STEP})
    suite = TerminalSensorSuite(suite_parameters, context)
    navigation = make_navigation(context)
    bus = context.bus
    bus.write(MessageKey.SURFACE_GEOMETRY, SurfaceGeometry(site_origin=torch.zeros(LANDERS, 3)), WORLD)
    error = InitialNavigationError({"position_m": 1.0, "velocity_mps": 0.2, "attitude_deg": 0.1}, LANDERS, "cpu")
    generator = torch.Generator().manual_seed(7)
    everyone = torch.arange(LANDERS)
    error.reset(everyone, generator)
    bus.write(MessageKey.INITIAL_NAVIGATION_STATE, error.estimate(truth_at(0.0)), WORLD)
    suite.reset(everyone, generator)
    navigation.reset(everyone, generator)
    means = []
    for step in range(STEPS):
        bus.write(MessageKey.TRUTH_STATE, truth_at(step * CONTROL_STEP), WORLD)
        suite.step(bus, CONTROL_STEP)
        navigation.step(bus, CONTROL_STEP)
        means.append(bus.read(MessageKey.NAVIGATION_ESTIMATE, _EstimateReader()).mean.clone())
    return means, navigation


def test_bridged_filter_equals_the_filter_in_this_process():
    """Loopback node running ekf_fdi: every estimate equals the in-process ekf_fdi exactly."""
    in_process, _ = fly_filter(lambda context: FaultDetectingKalmanFilterNavigation(navigation_parameters(), context))
    bridged, bridge = fly_filter(lambda context: NavigationBridge({"node": {"binding": "ekf_fdi", **navigation_parameters()}}, context))
    summary = bridge.deadline.summary()
    bridge.close()
    assert bridge.requires == FaultDetectingKalmanFilterNavigation.requires
    assert summary["replies"] == STEPS and summary["deadline_misses"] == 0
    for expected, produced in zip(in_process, bridged):
        assert torch.equal(expected, produced)


def test_late_replies_keep_the_previous_estimate():
    """Added latency above the timeout: every reply misses; the first is used (nothing to hold yet) and then stays."""
    parameters = {
        "node": {"binding": "ekf_fdi", **navigation_parameters()},
        "timeout_ms": 20.0, "added_latency_ms": 30.0, "count_measured_round_trip": False,
    }
    means, bridge = fly_filter(lambda context: NavigationBridge(parameters, context))
    summary = bridge.deadline.summary()
    bridge.close()
    assert summary["deadline_misses"] == STEPS
    assert all(torch.equal(means[0], mean) for mean in means)


def test_bridged_agent_equals_the_agent_in_this_process():
    """Loopback node running zem_zev: the commands equal the in-process agent's exactly."""
    preset = load_lander_preset(CONFIG / "landers" / "apollo_lunar_module.yaml")
    with open(CONFIG / "agent" / "zem_zev.yaml") as file:
        agent_parameters = yaml.safe_load(file)
    agent_parameters.pop("binding")
    shared = {"lander_preset": preset}
    count, jets = 4, 16
    local = ZeroEffortMissZeroEffortVelocityAgent(agent_parameters, count, jets, "cpu", dict(shared))
    bridged = AgentBridge({"node": {"binding": "zem_zev", **agent_parameters}, "policy_format": "torchscript"}, count, jets, "cpu", dict(shared))
    everyone = torch.arange(count)
    local.reset(everyone, torch.Generator())
    bridged.reset(everyone, torch.Generator())
    layout = NavigationStateLayout
    state = torch.zeros(count, layout.DIMENSION)
    state[:, layout.POSITION] = torch.tensor([[20.0, -10.0, 40.0], [0.0, 0.0, 15.0], [-30.0, 5.0, 60.0], [5.0, 5.0, 8.0]])
    state[:, layout.VELOCITY] = torch.tensor([[-1.0, 0.5, -3.0], [0.0, 0.0, -1.0], [2.0, 0.0, -4.0], [0.2, -0.1, -1.5]])
    state[:, layout.ORIENTATION_QUATERNION] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    state[:, layout.MASS] = 7300.0
    state[:, layout.HEIGHT_ABOVE_GROUND] = state[:, 2:3] - 2.0
    try:
        for step in range(20):
            observation = Observation(policy=state.clone())
            expected = local.compute_command(observation, 0.02)
            produced = bridged.compute_command(observation, 0.02)
            assert torch.equal(expected.throttle, produced.throttle)
            assert torch.equal(expected.reaction_control_jets, produced.reaction_control_jets)
            state[:, layout.POSITION] += 0.02 * state[:, layout.VELOCITY]
            state[:, layout.HEIGHT_ABOVE_GROUND] = state[:, 2:3] - 2.0
    finally:
        bridged.close()


class _TinyNormaliser(torch.nn.Module):
    """Observation normaliser with the buffer names of rsl_rl's (``_mean``, ``_std``)."""

    def __init__(self, size: int) -> None:
        """Create fixed statistics."""
        super().__init__()
        self.register_buffer("_mean", torch.linspace(-1.0, 1.0, size).unsqueeze(0))
        self.register_buffer("_std", torch.linspace(0.5, 2.0, size).unsqueeze(0))

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """Return the normalised observation."""
        return (observation - self._mean) / self._std


class _TinyRecurrentPolicy(torch.nn.Module):
    """Same signature as the GRU export of training: (observation, hidden) -> (action, new hidden)."""

    def __init__(self) -> None:
        """Build a small normaliser, GRU and head."""
        super().__init__()
        torch.manual_seed(3)
        self.obs_normalizer = _TinyNormaliser(6)
        self.rnn = torch.nn.GRU(6, 8)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ELU(), torch.nn.Linear(16, 3))

    def forward(self, observation: torch.Tensor, hidden_state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the action and the new hidden state."""
        output, new_hidden_state = self.rnn(self.obs_normalizer(observation).unsqueeze(0), hidden_state)
        return self.mlp(output.squeeze(0)), new_hidden_state


def test_onnx_export_of_a_recurrent_policy_matches_torchscript(tmp_path):
    """Export a small GRU policy; the ONNX runner gives the TorchScript outputs, for any batch size."""
    policy_path = tmp_path / "policy.pt"
    torch.jit.script(_TinyRecurrentPolicy().eval()).save(str(policy_path))
    report = export_policy_to_onnx(policy_path, tmp_path / "policy.onnx", "gru", hidden_size=8)
    assert report["maximum_output_difference"] < 1e-5
    runner = OnnxPolicy(tmp_path / "policy.onnx")
    policy = torch.jit.load(str(policy_path))
    observation, hidden = torch.randn(5, 6), torch.zeros(1, 5, 8)
    with torch.no_grad():
        expected_action, expected_hidden = policy(observation, hidden)
    action, new_hidden = runner(observation, hidden)
    assert math.isclose(float((action - expected_action).abs().max()), 0.0, abs_tol=1e-5)
    assert math.isclose(float((new_hidden - expected_hidden).abs().max()), 0.0, abs_tol=1e-5)
