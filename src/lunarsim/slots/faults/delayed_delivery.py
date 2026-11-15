# SPDX-License-Identifier: BSD-3-Clause
"""``faults: delay`` - sensors whose samples arrive late (milestone 3, scenario S4).

For each listed sensor, every episode draws a delay from ``delay_ms`` (uniform); every sample
then arrives that delay plus a per-sample jitter (uniform, plus or minus ``jitter_ms``) after its
capture, rounded to whole control steps, with its true capture ``stamp``. If two samples would
arrive in the same control step, the newer one wins; a sample that would arrive before the one
already delivered is dropped (no out-of-order delivery). All numbers are scenario parameters
(design S4: Doppler 50-200 ms, jitter 20 ms).

The binding is truth-side like ``faults: schedule`` (it writes a ``fault_status`` too, with the
sensors flagged as faulty from the start, so ``metrics: fault_response`` works unchanged).
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import Measurement, MessageKey
from lunarsim.core.registry import register_binding
from lunarsim.slots.faults.fault_schedule import FAULT_CODE, TARGET_NAMES, TARGETS, FaultStatus


@register_binding(SlotName.FAULTS, "delay")
class DelayedDelivery(BindingBase):
    """Delivers the listed sensors' samples late, with their capture stamps."""

    requires = frozenset({MessageKey.TRUTH_STATE, *(key for key, _ in TARGETS.values())})
    provides = frozenset({MessageKey.FAULT_STATUS, *(key for key, _ in TARGETS.values())})
    reads_truth = True

    def __init__(self, parameters, context) -> None:
        """Read the delay ranges per sensor and size the delivery buffers."""
        super().__init__(parameters, context)
        self.dt = float(context.shared_data["control_step_duration"])
        self.delays = {str(name): [float(value) / 1000.0 for value in bounds] for name, bounds in dict(self.parameters["delay_ms"]).items()}
        unknown = set(self.delays) - set(TARGET_NAMES)
        if unknown:
            raise ValueError(f"Unknown delayed sensors {sorted(unknown)}; known: {TARGET_NAMES}")
        self.jitter = float(self.parameters.get("jitter_ms", 0.0)) / 1000.0
        longest = max(high for _, high in self.delays.values()) + self.jitter
        self.length = int(round(longest / self.dt)) + 3
        count, device = self.number_of_environments, self.device
        self.delay = {name: torch.zeros(count, device=device) for name in self.delays}
        self.buffers: dict[str, tuple[Tensor, Tensor, Tensor, Tensor] | None] = {name: None for name in self.delays}
        self.last_delivered_stamp = {name: torch.full((count,), -1.0, device=device) for name in self.delays}
        self.step_counter = 0
        self.generator: torch.Generator | None = None

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Draw each sensor's delay for the episode and drop pending samples."""
        self.generator = gen
        for name, (low, high) in self.delays.items():
            self.delay[name][env_ids] = low + (high - low) * torch.rand(len(env_ids), generator=gen, device=self.device)
            self.last_delivered_stamp[name][env_ids] = -1.0
            if self.buffers[name] is not None:
                self.buffers[name][1][:, env_ids] = False

    def step(self, bus, dt: float) -> None:
        """Queue this step's new samples at their arrival step and deliver what arrives now."""
        if self.generator is None:
            self.generator = torch.Generator(device=self.device)
        count = self.number_of_environments
        now = self.step_counter % self.length
        for name in self.delays:
            key, _ = TARGETS[name]
            measurement: Measurement = bus.read(key, self)
            if self.buffers[name] is None:
                dimension = measurement.value.shape[1]
                self.buffers[name] = (
                    torch.full((self.length, count, dimension), float("nan"), device=self.device),
                    torch.zeros(self.length, count, dtype=torch.bool, device=self.device),
                    torch.full((self.length, count), -1.0, device=self.device),
                    torch.ones(self.length, count, dimension, device=self.device),
                )
            values, valid, stamps, covariances = self.buffers[name]
            jitter = self.jitter * (2.0 * torch.rand(count, generator=self.generator, device=self.device) - 1.0)
            steps_late = torch.clamp(torch.round((self.delay[name] + jitter) / dt).long(), 1, self.length - 1)
            arrival = (now + steps_late) % self.length
            environments = torch.arange(count, device=self.device)
            new = measurement.valid
            # Newer samples overwrite older ones that would arrive in the same step.
            replace = new & (~valid[arrival, environments] | (measurement.stamp >= stamps[arrival, environments]))
            values[arrival[replace], environments[replace]] = measurement.value[replace]
            stamps[arrival[replace], environments[replace]] = measurement.stamp[replace]
            covariances[arrival[replace], environments[replace]] = measurement.covariance[replace]
            valid[arrival[replace], environments[replace]] = True
            delivered_valid = valid[now] & (stamps[now] > self.last_delivered_stamp[name])
            self.last_delivered_stamp[name] = torch.where(delivered_valid, stamps[now], self.last_delivered_stamp[name])
            delivered = Measurement(
                value=values[now].clone(), valid=delivered_valid, stamp=stamps[now].clone(), covariance=covariances[now].clone()
            )
            valid[now] = False
            bus.write(key, delivered, self)
        self.step_counter += 1
        delayed_target = TARGET_NAMES.index(next(iter(self.delays)))
        bus.write(
            MessageKey.FAULT_STATUS,
            FaultStatus(
                fault_type=torch.full((count,), FAULT_CODE["delay_spike"], dtype=torch.long, device=self.device),
                target=torch.full((count,), delayed_target, dtype=torch.long, device=self.device),
                onset_altitude=torch.full((count,), float("inf"), device=self.device),
                active=torch.ones(count, dtype=torch.bool, device=self.device),
                onset_time=torch.zeros(count, device=self.device),
                corrupting_now=torch.ones(count, dtype=torch.bool, device=self.device),
            ),
            self,
        )
