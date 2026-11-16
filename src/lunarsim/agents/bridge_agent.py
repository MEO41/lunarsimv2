# SPDX-License-Identifier: BSD-3-Clause
"""``agent: ros2_bridge`` - the agent runs as a separate node (scenario S10, milestone 7).

The inner agent is named in ``node`` (``node.binding`` plus its parameters, e.g. the whole
``ppo_gru`` config with ``policy_path``). With ``policy_format: onnx`` the node flies the ONNX
export of the policy (``onnx_path``; design S10: the policy node "loads the ONNX export of
ppo_gru") instead of the TorchScript file; the rest of the agent (observation scaling, action
adapter, attitude controller) is the same code.

Each control step the observation goes to the node and its command comes back. A reply that
misses the deadline (:mod:`lunarsim.bridge.deadline_model`) is not used: the previous command is
flown again for this step, as a flight loop with a deadline would.

Parameters: ``node``, ``transport`` (``loopback`` or ``ros2``), ``node_name``, ``policy_format``
(``torchscript`` or ``onnx``), ``onnx_path``, ``timeout_ms``, ``added_latency_ms``, ``jitter_ms``,
``count_measured_round_trip``, ``seed``.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor

from lunarsim.agents.agent_base import AgentBase
from lunarsim.bridge.bridge_support import EpisodeTimingRecords, close_at_exit, inner_binding_selection, picklable_shared_data
from lunarsim.bridge.deadline_model import DeadlineModel
from lunarsim.bridge.message_transport import make_transport
from lunarsim.bridge.node_programs import tensors_to
from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import Command, Observation
from lunarsim.core.registry import get_binding_class, register_binding

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def repository_path(path: str) -> str:
    """Return ``path`` as an absolute path (relative paths are taken from the repository root)."""
    candidate = Path(path)
    return str(candidate if candidate.is_absolute() else REPOSITORY_ROOT / candidate)


@register_binding(SlotName.AGENT, "ros2_bridge")
class AgentBridge(AgentBase):
    """Runs the inner agent in a node process (see the module text)."""

    def __init__(self, parameters, number_of_environments, number_of_jets, device, shared_data=None) -> None:
        """Start the node with the inner agent."""
        super().__init__(parameters, number_of_environments, number_of_jets, device, shared_data)
        name, inner_parameters = inner_binding_selection(self.parameters, "agent")
        inner = get_binding_class(SlotName.AGENT, name)
        if inner_parameters.get("policy_path"):
            inner_parameters["policy_path"] = repository_path(str(inner_parameters["policy_path"]))
        seed = int(self.parameters.get("seed", 0))
        policy_format = str(self.parameters.get("policy_format", "torchscript"))
        spec = {
            "kind": "agent",
            "module": inner.__module__,
            "class_name": inner.__name__,
            "parameters": inner_parameters,
            "number_of_environments": number_of_environments,
            "number_of_jets": number_of_jets,
            "shared_data": picklable_shared_data(self.shared_data),
            "policy_format": policy_format,
            "onnx_path": repository_path(str(self.parameters["onnx_path"])) if policy_format == "onnx" else None,
            "seed": seed,
        }
        self.transport = make_transport(str(self.parameters.get("transport", "loopback")), spec, str(self.parameters.get("node_name", "lunarsim_policy")))
        self.deadline = DeadlineModel.from_parameters(self.parameters, seed + 1)
        self.timing = EpisodeTimingRecords("bridge_agent", number_of_environments)
        self.command: Command | None = None
        self.must_use_next_reply = True
        close_at_exit(self)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Reset the node's environments; the next reply is used whatever its timing."""
        self.transport.request(("reset", env_ids.tolist()))
        self.timing.reset(env_ids)
        self.must_use_next_reply = True

    def compute_command(self, observation: Observation, control_step_duration: float) -> Command:
        """Send the observation, take the node's command (or fly the last one again on a deadline miss)."""
        command, round_trip = self.transport.request(("command", tensors_to(observation, "cpu"), control_step_duration))
        on_time = self.deadline.reply_is_on_time(round_trip)
        self.timing.add(on_time, 1e3 * round_trip)
        if on_time or self.must_use_next_reply or self.command is None:
            self.command = tensors_to(command, self.device)
            self.must_use_next_reply = False
        return self.command

    def drain_decision_records(self) -> dict[str, Tensor]:
        """Return the per-episode timing records (replies, deadline misses, longest round trip)."""
        return self.timing.drain()

    def close(self) -> None:
        """Stop the node."""
        if getattr(self, "transport", None) is not None:
            self.transport.close()
            self.transport = None
