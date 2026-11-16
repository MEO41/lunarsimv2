# SPDX-License-Identifier: BSD-3-Clause
"""``nav: ros2_bridge`` - the navigation binding runs as a separate node (scenario S10, milestone 7).

Design S10: "The ros2_bridge binding has the same interface as every other binding. It publishes
the slot's input as a topic and waits for the node's output. Thus, the EKF and the policy move
out of the simulator with no change to the other slots."

The inner binding is named in ``node`` (``node.binding`` plus its parameters, e.g. the whole
``ekf_fdi`` config). This binding declares the inner binding's ``requires`` and ``provides``, so
the slot-loop check (including the truth-leak check) sees the real data flow; a navigation binding
that reads truth cannot be bridged. Each control step it sends the input messages that changed
since the last step (the node keeps the rest), waits for the estimate, and writes it on the bus.
A reply that misses the deadline (see :mod:`lunarsim.bridge.deadline_model`) is not used: the
previous estimate stays on the bus for this step.

Parameters: ``node``, ``transport`` (``loopback`` or ``ros2``), ``node_name``, ``timeout_ms``,
``added_latency_ms``, ``jitter_ms``, ``count_measured_round_trip``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.bridge.bridge_support import EpisodeTimingRecords, close_at_exit, inner_binding_selection, picklable_shared_data
from lunarsim.bridge.deadline_model import DeadlineModel
from lunarsim.bridge.message_transport import make_transport
from lunarsim.bridge.node_programs import tensors_to
from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import MessageKey
from lunarsim.core.registry import TRUTH_NAVIGATION_NAMES, get_binding_class, register_binding


@register_binding(SlotName.NAVIGATION, "ros2_bridge")
class NavigationBridge(BindingBase):
    """Runs the inner navigation binding in a node process (see the module text)."""

    provides = frozenset({MessageKey.NAVIGATION_ESTIMATE})
    reads_truth = False

    def __init__(self, parameters, context) -> None:
        """Start the node with the inner binding and take over its declarations."""
        super().__init__(parameters, context)
        name, inner_parameters = inner_binding_selection(self.parameters, "nav")
        inner = get_binding_class(SlotName.NAVIGATION, name)
        if getattr(inner, "reads_truth", False) or name in TRUTH_NAVIGATION_NAMES or MessageKey.TRUTH_STATE in inner.requires:
            raise ValueError(f"nav: ros2_bridge cannot run '{name}': it reads the true state")
        self.inner_name = name
        self.requires = frozenset(inner.requires)
        self.provides = frozenset(inner.provides)
        self.health_dimension = int(getattr(inner, "health_dimension", 0))
        seed = int(self.parameters.get("seed", 0))
        spec = {
            "kind": "navigation",
            "module": inner.__module__,
            "class_name": inner.__name__,
            "parameters": inner_parameters,
            "number_of_environments": self.number_of_environments,
            "shared_data": picklable_shared_data(context.shared_data),
            "physics_step_duration": context.clock.physics_step_duration,
            "physics_steps_per_control_step": context.clock.physics_steps_per_control_step,
            "seed": seed,
        }
        self.transport = make_transport(str(self.parameters.get("transport", "loopback")), spec, str(self.parameters.get("node_name", "lunarsim_ekf")))
        self.deadline = DeadlineModel.from_parameters(self.parameters, seed)
        self.timing = EpisodeTimingRecords("bridge_navigation", self.number_of_environments)
        self.sent: dict[str, object] = {}
        self.estimate = None
        self.must_use_next_reply = True
        close_at_exit(self)

    def changed_messages(self, bus) -> dict:
        """Return the input messages written since they were last sent (CPU copies); the node keeps the others."""
        messages = {}
        for key in sorted(self.requires):
            if not bus.has(key):
                continue
            message = bus.read(key, self)
            if self.sent.get(key) is not message:
                messages[key] = tensors_to(message, "cpu")
                self.sent[key] = message
        return messages

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Reset the node's environments (with the inputs written so far: a filter reads its start state at reset).

        The next reply is used whatever its timing: there is no estimate of the new episode to hold yet.
        """
        self.transport.request(("reset", env_ids.tolist(), self.changed_messages(self.context.bus)))
        self.timing.reset(env_ids)
        self.must_use_next_reply = True

    def step(self, bus, dt: float) -> None:
        """Send the changed inputs, take the node's estimate (or keep the last one on a deadline miss)."""
        messages = self.changed_messages(bus)
        self.transport.publish_clock(float(self.context.clock.time[0]))
        estimate, round_trip = self.transport.request(("step", messages, dt))
        on_time = self.deadline.reply_is_on_time(round_trip)
        self.timing.add(on_time, 1e3 * round_trip)
        if on_time or self.must_use_next_reply or self.estimate is None:
            self.estimate = tensors_to(estimate, self.device)
            self.must_use_next_reply = False
        bus.write(MessageKey.NAVIGATION_ESTIMATE, self.estimate, self)

    def drain_decision_records(self) -> dict[str, Tensor]:
        """Return the per-episode timing records (replies, deadline misses, longest round trip)."""
        return self.timing.drain()

    def close(self) -> None:
        """Stop the node."""
        if getattr(self, "transport", None) is not None:
            self.transport.close()
            self.transport = None
