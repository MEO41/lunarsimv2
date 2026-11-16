# SPDX-License-Identifier: BSD-3-Clause
"""What a bridged node runs: an unchanged navigation binding or an unchanged agent (S10).

A node is described by a picklable ``spec`` dictionary (built by the bridge binding in the
simulator process) and serves requests one at a time, in order:

- ``("reset", env_ids, messages)`` -> ``None``: the changed input messages are written (a filter
  reads its start state at reset), then the inner binding's ``reset`` runs for those environments;
- ``("step", messages, dt)`` -> ``NavEstimate`` (navigation node): the changed input messages are
  written on the node's own message bus, then the inner binding's ``step`` runs;
- ``("command", observation, dt)`` -> ``Command`` (agent node): the inner agent's ``compute_command``;
- ``("close",)`` -> ``None``: the node stops.

The node never imports Isaac Sim: it imports only the inner binding's module (named in the spec)
and runs on the CPU, as GNC software on a flight computer would. Tensors cross the transport on
the CPU.
"""

from __future__ import annotations

import dataclasses
import importlib
from typing import Any

import torch

from lunarsim.core.binding import BindingContext
from lunarsim.core.message_bus import MessageBus
from lunarsim.core.message_types import MessageKey, Observation
from lunarsim.core.simulation_clock import SimulationClock


def tensors_to(value: Any, device: torch.device | str) -> Any:
    """Return ``value`` with every tensor inside it (dataclass fields, lists, dicts) moved to ``device``."""
    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {key: tensors_to(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(tensors_to(item, device) for item in value)
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        changes = {item.name: tensors_to(getattr(value, item.name), device) for item in dataclasses.fields(value) if item.init}
        return dataclasses.replace(value, **changes)
    if hasattr(value, "to") and callable(value.to) and type(value).__module__.startswith("lunarsim"):
        return value.to(device)
    return value


class _BridgeWriter:
    """Writer of the input messages on a node's bus: it provides every key the inner binding reads."""

    reads_truth = False

    def __init__(self, keys: frozenset[str]) -> None:
        """Declare the keys this writer may write."""
        self.provides = frozenset(keys)
        self.requires = frozenset()


class _OutputReader:
    """Reader of the inner binding's output message on a node's bus."""

    reads_truth = False

    def __init__(self, keys: frozenset[str]) -> None:
        """Declare the keys this reader may read."""
        self.requires = frozenset(keys)
        self.provides = frozenset()


def binding_class_from_spec(spec: dict) -> type:
    """Import the inner binding's module and return its class (this also registers it)."""
    module = importlib.import_module(spec["module"])
    return getattr(module, spec["class_name"])


class NavigationNodeProgram:
    """Runs one navigation binding on its own bus, fed with the simulator's messages."""

    def __init__(self, spec: dict) -> None:
        """Build the inner navigation binding on the CPU from the spec."""
        self.device = spec.get("device", "cpu")
        count = int(spec["number_of_environments"])
        self.bus = MessageBus()
        clock = SimulationClock(count, float(spec["physics_step_duration"]), int(spec["physics_steps_per_control_step"]), self.device)
        context = BindingContext(number_of_environments=count, device=self.device, clock=clock, bus=self.bus, shared_data=dict(spec["shared_data"]))
        binding_class = binding_class_from_spec(spec)
        self.binding = binding_class(dict(spec["parameters"]), context)
        self.writer = _BridgeWriter(self.binding.requires)
        self.reader = _OutputReader(self.binding.provides)
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(spec.get("seed", 0)))

    def write_messages(self, messages: dict[str, Any]) -> None:
        """Write the changed input messages on the node's bus."""
        for key, message in messages.items():
            self.bus.write(key, tensors_to(message, self.device), self.writer)

    def reset(self, env_ids: list[int], messages: dict[str, Any] | None = None) -> None:
        """Write the changed inputs, then reset the inner binding's environments ``env_ids``."""
        self.write_messages(messages or {})
        self.binding.reset(torch.as_tensor(env_ids, dtype=torch.long, device=self.device), self.generator)

    def step(self, messages: dict[str, Any], dt: float):
        """Write the changed input messages, run the inner binding, return its estimate (CPU)."""
        self.write_messages(messages)
        self.binding.step(self.bus, dt)
        return tensors_to(self.bus.read(MessageKey.NAVIGATION_ESTIMATE, self.reader), "cpu")


class AgentNodeProgram:
    """Runs one agent; with ``policy_format: onnx`` its policy network is the ONNX export."""

    def __init__(self, spec: dict) -> None:
        """Build the inner agent on the CPU from the spec."""
        self.device = spec.get("device", "cpu")
        agent_class = binding_class_from_spec(spec)
        self.agent = agent_class(
            parameters=dict(spec["parameters"]),
            number_of_environments=int(spec["number_of_environments"]),
            number_of_jets=int(spec["number_of_jets"]),
            device=self.device,
            shared_data=dict(spec["shared_data"]),
        )
        if spec.get("policy_format", "torchscript") == "onnx":
            from lunarsim.bridge.onnx_policy import OnnxPolicy

            self.agent.policy = OnnxPolicy(spec["onnx_path"])
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(int(spec.get("seed", 0)))

    def reset(self, env_ids: list[int], messages: dict[str, Any] | None = None) -> None:
        """Reset the inner agent's environments ``env_ids`` (an agent gets no bus messages)."""
        self.agent.reset(torch.as_tensor(env_ids, dtype=torch.long, device=self.device), self.generator)

    def command(self, observation: Observation, dt: float):
        """Return the inner agent's command (CPU) for this observation."""
        return tensors_to(self.agent.compute_command(tensors_to(observation, self.device), dt), "cpu")


def build_node_program(spec: dict):
    """Return the node program that the spec names (``kind``: navigation or agent)."""
    torch.set_num_threads(int(spec.get("threads", 1)))
    if spec["kind"] == "navigation":
        return NavigationNodeProgram(spec)
    if spec["kind"] == "agent":
        return AgentNodeProgram(spec)
    raise ValueError(f"unknown node kind '{spec['kind']}'")


def handle_request(program, request: tuple):
    """Serve one request on a node program; return the reply (None for reset)."""
    kind = request[0]
    if kind == "reset":
        program.reset(request[1], request[2] if len(request) > 2 else None)
        return None
    if kind == "step":
        return program.step(request[1], request[2])
    if kind == "command":
        return program.command(request[1], request[2])
    raise ValueError(f"unknown request '{kind}'")
