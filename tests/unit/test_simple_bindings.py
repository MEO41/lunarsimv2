# SPDX-License-Identifier: BSD-3-Clause
"""The pure-torch bindings of milestone 0: none-bindings, truth navigation, fixed start."""

import torch
from helpers import make_binding_context, make_truth_state

import lunarsim.slots  # noqa: F401
from lunarsim.core.binding import Binding, SlotName
from lunarsim.core.message_types import MessageKey, NavigationStateLayout
from lunarsim.core.registry import get_binding_class, registered_binding_names


class _AnyWriter:
    provides = frozenset({MessageKey.TRUTH_STATE, MessageKey.COMMAND})
    requires = frozenset()


class _AnyReader:
    requires = frozenset(
        {MessageKey.NAVIGATION_ESTIMATE, MessageKey.EPISODE_EVENTS, MessageKey.ACTUATOR_WRENCH, MessageKey.START_STATE}
    )
    provides = frozenset()
    reads_truth = False


def _build(slot, name, parameters=None, number_of_environments=4):
    """Build a binding on the CPU and return it with its context."""
    context = make_binding_context(number_of_environments)
    return get_binding_class(slot, name)(parameters or {}, context), context


def test_every_environment_slot_has_a_registered_binding_that_follows_the_protocol():
    """Each of the eight environment slots has at least one binding with the Binding interface."""
    for slot in SlotName.INSIDE_ENVIRONMENT:
        names = registered_binding_names(slot)
        assert names, f"slot {slot} has no binding"
        for name in names:
            binding_class = get_binding_class(slot, name)
            assert binding_class.slot == slot
            assert isinstance(binding_class.requires, frozenset) and isinstance(binding_class.provides, frozenset)
    binding, _ = _build(SlotName.FAULTS, "none")
    assert isinstance(binding, Binding)


def test_truth_navigation_copies_the_state_in_layout_order():
    """nav: truth writes position, velocity, quaternion, body rates and mass with zero covariance."""
    binding, context = _build(SlotName.NAVIGATION, "truth")
    truth = make_truth_state()
    truth.velocity[:, 2] = -3.0
    truth.footpad_height_above_ground[:] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    context.bus.write(MessageKey.TRUTH_STATE, truth, _AnyWriter())
    binding.step(context.bus, 0.02)
    estimate = context.bus.read(MessageKey.NAVIGATION_ESTIMATE, _AnyReader())
    layout = NavigationStateLayout
    assert torch.equal(estimate.mean[:, layout.POSITION], truth.position)
    assert torch.equal(estimate.mean[:, layout.VELOCITY], truth.velocity)
    assert torch.equal(estimate.mean[:, layout.ORIENTATION_QUATERNION], truth.orientation_quaternion)
    assert torch.equal(estimate.mean[:, layout.MASS].squeeze(1), truth.mass)
    assert torch.equal(estimate.mean[:, layout.HEIGHT_ABOVE_GROUND].squeeze(1), truth.footpad_height_above_ground)
    assert float(estimate.covariance.abs().sum()) == 0.0


def test_none_bindings_write_zero_outputs():
    """actuators: none writes a zero wrench; metrics: none never terminates."""
    actuators, context = _build(SlotName.ACTUATORS, "none")
    actuators.step(context.bus, 0.005)
    wrench = context.bus.read(MessageKey.ACTUATOR_WRENCH, _AnyReader())
    assert float(wrench.force.abs().sum() + wrench.torque.abs().sum() + wrench.mass_flow_rate.abs().sum()) == 0.0
    metrics, context = _build(SlotName.METRICS, "none")
    metrics.step(context.bus, 0.02)
    assert not bool(context.bus.read(MessageKey.EPISODE_EVENTS, _AnyReader()).terminated.any())


def test_fixed_start_writes_the_configured_state():
    """start: fixed gives every environment the same position and velocity."""
    start, context = _build(SlotName.START, "fixed", {"position_m": [1.0, 2.0, 3.0], "velocity_mps": [0.0, 0.0, -1.0]})
    start.step(context.bus, 0.0)
    state = context.bus.read(MessageKey.START_STATE, _AnyReader())
    assert state.position.tolist() == [[1.0, 2.0, 3.0]] * 4
    assert state.velocity[:, 2].tolist() == [-1.0] * 4
