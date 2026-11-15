# SPDX-License-Identifier: BSD-3-Clause
"""``faults: schedule`` - seeded sensor faults for the failure study (milestone 3, scenario S3).

Every episode, each environment draws one fault from the configured classes (or none, with
probability ``no_fault_fraction``): a type, a target sensor, an onset altitude and a size. The
fault starts when the footpad height above the ground first falls below the onset altitude (at
once if the lander starts below it) and lasts to the end of the episode. A fault hits the whole
target sensor (all its beams). Types:

| Class | Type | What the navigation receives after the onset |
| --- | --- | --- |
| intermittent | ``dropout_burst`` | bursts (Poisson events) in which no sample arrives (``valid`` False) |
| intermittent | ``delay_spike`` | bursts in which the samples arrive late: the sample of ``delay`` seconds ago, with its old ``stamp`` |
| intermittent | ``outlier_spike`` | single samples with a large error (random sign) |
| persistent | ``bias`` | a constant error |
| persistent | ``drift`` | an error that grows linearly with time since the onset |
| persistent | ``stuck`` | the last value before the onset, again and again, reported as new |
| persistent | ``scale`` | values times a factor |
| persistent | ``total_loss`` | nothing (``valid`` False, NaN values) |

Sizes are given per channel kind: ``range`` channels in m, ``velocity`` channels in m/s. All
sizes, rates and lengths are **model choices** (``configs/faults/schedule.yaml``); no public
source gives fault statistics for these sensors.

The binding is truth-side (it reads the footpad height for the onset). It writes the faulted
measurements under the sensors' own keys, and a truth-side ``fault_status`` message for the
metrics; the navigation never reads that message. The draws use the faults slot's generator,
so the same seed gives the same faults for every agent. ``set_fixed_faults`` replays a fixed
list (held-out test grid).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import Measurement, MessageKey, TruthState
from lunarsim.core.registry import register_binding
from lunarsim.slots.sensors.terminal_suite import DOPPLER_VELOCIMETER_KEY, RADAR_ALTIMETER_KEY

FAULT_TYPES = (
    "none",
    "dropout_burst",
    "delay_spike",
    "outlier_spike",
    "bias",
    "drift",
    "stuck",
    "scale",
    "total_loss",
)
FAULT_CODE = {name: code for code, name in enumerate(FAULT_TYPES)}
FAULT_CLASSES = {
    "intermittent": ("dropout_burst", "delay_spike", "outlier_spike"),
    "persistent": ("bias", "drift", "stuck", "scale", "total_loss"),
}
# Target sensors: message key and the kind of each value channel.
TARGETS = {
    "radar_altimeter": (RADAR_ALTIMETER_KEY, ("range",)),
    "doppler_velocimeter": (DOPPLER_VELOCIMETER_KEY, ("velocity",) * 3 + ("range",) * 3),
}
TARGET_NAMES = tuple(TARGETS)
# Size entries of the config per fault type: (range channels, velocity channels).
SIZE_KEYS = {
    "bias": ("bias_range_m", "bias_velocity_mps"),
    "outlier_spike": ("outlier_range_m", "outlier_velocity_mps"),
    "drift": ("drift_range_mps", "drift_velocity_mps2"),
}


@dataclass
class FaultStatus:
    """Truth-side description of every environment's fault (for the metrics)."""

    fault_type: Tensor  # (N,) long, FAULT_CODE
    target: Tensor  # (N,) long, index into TARGET_NAMES (-1 without a fault)
    onset_altitude: Tensor  # (N,) m
    active: Tensor  # (N,) bool, the fault has started
    onset_time: Tensor  # (N,) s since the episode start, -1 before the onset
    corrupting_now: Tensor  # (N,) bool, the fault changes the delivered data in this control step


@register_binding(SlotName.FAULTS, "schedule")
class FaultSchedule(BindingBase):
    """Draws one fault per episode and applies it to the target sensor's messages."""

    requires = frozenset({MessageKey.TRUTH_STATE, *(key for key, _ in TARGETS.values())})
    provides = frozenset({MessageKey.FAULT_STATUS, *(key for key, _ in TARGETS.values())})
    reads_truth = True

    def __init__(self, parameters, context) -> None:
        """Read the fault menu and allocate the per-environment fault state and sample buffers."""
        super().__init__(parameters, context)
        count, device = self.number_of_environments, self.device
        get = self.parameters.get
        self.targets = [str(name) for name in get("targets", list(TARGET_NAMES))]
        unknown = set(self.targets) - set(TARGET_NAMES)
        if unknown:
            raise ValueError(f"Unknown fault targets {sorted(unknown)}; known: {TARGET_NAMES}")
        self.types = []
        for class_name in get("classes", list(FAULT_CLASSES)):
            self.types += list(get(class_name, {}).get("types", FAULT_CLASSES[class_name]))
        self.no_fault_fraction = float(get("no_fault_fraction", 0.0))
        self.onset_range = [float(value) for value in get("onset_altitude_m", [5.0, 40.0])]
        self.sizes = dict(get("sizes"))
        self.intermittent = dict(get("intermittent"))
        self.dt = float(context.shared_data["control_step_duration"])
        maximum_delay = float(self.intermittent["delay_s"][1])
        self.buffer_length = int(round(maximum_delay / self.dt)) + 2

        def zeros(*shape, dtype=torch.float32):
            """Allocate a per-environment tensor."""
            return torch.zeros(count, *shape, dtype=dtype, device=device)

        self.fault_type = zeros(dtype=torch.long)
        self.target = zeros(dtype=torch.long) - 1
        self.onset_altitude = zeros()
        self.active = zeros(dtype=torch.bool)
        self.onset_time = zeros() - 1.0
        self.size = zeros()  # bias, outlier, drift rate: in units of the channel kind (see sizes)
        self.size_velocity = zeros()
        self.scale = zeros() + 1.0
        self.event_rate = zeros()
        self.burst_length = zeros()
        self.delay_steps = zeros(dtype=torch.long)
        self.event_end_time = zeros() - 1.0
        self.outlier_pending = zeros(dtype=torch.bool)
        self.stuck_value = {name: None for name in self.targets}
        self.buffers = {name: None for name in self.targets}  # ring buffers of (value, valid, stamp)
        self.buffer_index = 0
        self.generator: torch.Generator | None = None
        self.fixed_faults: dict[str, Tensor] | None = None
        self.corrupting_now = zeros(dtype=torch.bool)

    # ---- drawing ----

    def _uniform(self, bounds, count: int) -> Tensor:
        """Draw ``count`` uniform values in ``bounds`` from the slot's generator."""
        low, high = (float(value) for value in bounds)
        return low + (high - low) * torch.rand(count, generator=self.generator, device=self.device)

    def set_fixed_faults(self, fault_type: Tensor, target: Tensor, onset_altitude: Tensor, size_fraction: Tensor) -> None:
        """Use the given faults at every reset (held-out grid) instead of drawing them.

        Per environment: the fault code (``FAULT_CODE``), the target index (``TARGET_NAMES``), the
        onset altitude (m) and the size as a position inside the configured range (0 to 1).
        Signs, event times and delays still come from the slot's generator (same seed, same draws).
        """
        self.fixed_faults = {
            "fault_type": fault_type.to(self.device).long(),
            "target": target.to(self.device).long(),
            "onset_altitude": onset_altitude.to(self.device).float(),
            "size_fraction": size_fraction.to(self.device).float(),
        }

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Draw (or take) the fault of every reset environment and clear its state."""
        self.generator = gen
        count = len(env_ids)
        if self.fixed_faults is not None:
            fixed = self.fixed_faults
            self.fault_type[env_ids] = fixed["fault_type"][env_ids]
            self.target[env_ids] = fixed["target"][env_ids]
            self.onset_altitude[env_ids] = fixed["onset_altitude"][env_ids]
            drawn_size = fixed["size_fraction"][env_ids]
        else:
            type_index = torch.randint(len(self.types), (count,), generator=gen, device=self.device)
            codes = torch.tensor([FAULT_CODE[name] for name in self.types], device=self.device)[type_index]
            no_fault = torch.rand(count, generator=gen, device=self.device) < self.no_fault_fraction
            self.fault_type[env_ids] = torch.where(no_fault, torch.zeros_like(codes), codes)
            target_index = torch.randint(len(self.targets), (count,), generator=gen, device=self.device)
            all_targets = torch.tensor([TARGET_NAMES.index(name) for name in self.targets], device=self.device)
            self.target[env_ids] = torch.where(no_fault, torch.full_like(target_index, -1), all_targets[target_index])
            self.onset_altitude[env_ids] = self._uniform(self.onset_range, count)
            drawn_size = None
        self._draw_sizes(env_ids, drawn_size)
        self.active[env_ids] = False
        self.onset_time[env_ids] = -1.0
        self.event_end_time[env_ids] = -1.0
        self.outlier_pending[env_ids] = False
        for name in self.targets:
            if self.buffers[name] is not None:
                values, valid, stamps = self.buffers[name]
                valid[:, env_ids] = False
                self.stuck_value[name][env_ids] = float("nan")

    def _draw_sizes(self, env_ids: Tensor, size_fraction: Tensor | None) -> None:
        """Draw size, sign, scale, event rate, burst length and delay of the given environments.

        Each size is drawn uniformly inside its configured range; a fixed fault gives the
        position inside the range instead (``size_fraction``, 0 = low end, 1 = high end).
        """
        count = len(env_ids)
        fault_type = self.fault_type[env_ids]
        fraction = torch.rand(count, generator=self.generator, device=self.device) if size_fraction is None else size_fraction
        sign = torch.where(torch.rand(count, generator=self.generator, device=self.device) < 0.5, -1.0, 1.0)
        size = torch.zeros(count, device=self.device)
        size_velocity = torch.zeros(count, device=self.device)
        for name, (range_key, velocity_key) in SIZE_KEYS.items():
            chosen = fault_type == FAULT_CODE[name]
            low, high = (float(value) for value in self.sizes[range_key])
            size = torch.where(chosen, sign * (low + fraction * (high - low)), size)
            low, high = (float(value) for value in self.sizes[velocity_key])
            size_velocity = torch.where(chosen, sign * (low + fraction * (high - low)), size_velocity)
        low, high = (float(value) for value in self.sizes["scale_deviation"])
        scale = 1.0 + sign * (low + fraction * (high - low))
        self.scale[env_ids] = torch.where(fault_type == FAULT_CODE["scale"], scale, torch.ones_like(scale))
        self.size[env_ids] = size
        self.size_velocity[env_ids] = size_velocity
        intermittent = self.intermittent
        self.event_rate[env_ids] = self._uniform(intermittent["event_rate_per_s"], count)
        self.burst_length[env_ids] = self._uniform(intermittent["burst_length_s"], count)
        delay = self._uniform(intermittent["delay_s"], count)
        self.delay_steps[env_ids] = torch.clamp(torch.round(delay / self.dt).long(), 1, self.buffer_length - 1)

    # ---- applying ----

    def _remember(self, name: str, measurement: Measurement) -> None:
        """Keep the sensor's output of this step in the ring buffer (for delays and stuck values)."""
        if self.buffers[name] is None:
            count, dimension = measurement.value.shape
            self.buffers[name] = (
                torch.full((self.buffer_length, count, dimension), float("nan"), device=self.device),
                torch.zeros(self.buffer_length, count, dtype=torch.bool, device=self.device),
                torch.full((self.buffer_length, count), -1.0, device=self.device),
            )
            self.stuck_value[name] = torch.full_like(measurement.value, float("nan"))
        values, valid, stamps = self.buffers[name]
        values[self.buffer_index] = measurement.value
        valid[self.buffer_index] = measurement.valid
        stamps[self.buffer_index] = measurement.stamp

    def _delayed(self, name: str) -> tuple[Tensor, Tensor, Tensor]:
        """Return the (value, valid, stamp) of each environment's own delay ago."""
        values, valid, stamps = self.buffers[name]
        index = (self.buffer_index - self.delay_steps) % self.buffer_length  # (N,)
        environments = torch.arange(self.number_of_environments, device=self.device)
        return values[index, environments], valid[index, environments], stamps[index, environments]

    def step(self, bus, dt: float) -> None:
        """Start faults at their onset, advance the intermittent events, and corrupt the target messages."""
        truth: TruthState = bus.read(MessageKey.TRUTH_STATE, self)
        time = truth.time
        has_fault = self.fault_type != FAULT_CODE["none"]
        starting = has_fault & ~self.active & (truth.footpad_height_above_ground <= self.onset_altitude)
        self.onset_time = torch.where(starting, time, self.onset_time)
        self.active |= starting

        intermittent_burst = (self.fault_type == FAULT_CODE["dropout_burst"]) | (self.fault_type == FAULT_CODE["delay_spike"])
        in_event = time < self.event_end_time
        new_event = self.active & ~in_event & (
            torch.rand(self.number_of_environments, generator=self._generator(), device=self.device) < self.event_rate * dt
        )
        self.event_end_time = torch.where(new_event & intermittent_burst, time + self.burst_length, self.event_end_time)
        self.outlier_pending |= new_event & (self.fault_type == FAULT_CODE["outlier_spike"])
        in_burst = self.active & intermittent_burst & (time < self.event_end_time)

        corrupting = torch.zeros_like(self.active)
        for name in self.targets:
            key, kinds = TARGETS[name]
            measurement: Measurement = bus.read(key, self)
            self._remember(name, measurement)
            target_index = TARGET_NAMES.index(name)
            hit = self.active & (self.target == target_index)
            if not bool(hit.any()):
                self.stuck_value[name] = torch.where(measurement.valid.unsqueeze(1), measurement.value, self.stuck_value[name])
                continue
            # A sensor that sticks before its first sample sticks on its first sample.
            unset = torch.isnan(self.stuck_value[name]).all(dim=1) & measurement.valid
            self.stuck_value[name] = torch.where(unset.unsqueeze(1), measurement.value, self.stuck_value[name])
            value, valid, stamp = measurement.value.clone(), measurement.valid.clone(), measurement.stamp.clone()
            is_velocity = torch.tensor([kind == "velocity" for kind in kinds], device=self.device)
            channel_size = torch.where(is_velocity, self.size_velocity.unsqueeze(1), self.size.unsqueeze(1))
            code = self.fault_type

            def where(condition: Tensor, new: Tensor, old: Tensor) -> Tensor:
                """Replace rows where ``condition`` (per environment)."""
                return torch.where(condition.unsqueeze(1) if new.dim() == 2 else condition, new, old)

            bias = hit & (code == FAULT_CODE["bias"])
            value = where(bias, value + channel_size, value)
            drift = hit & (code == FAULT_CODE["drift"])
            elapsed = (time - self.onset_time).clamp_min(0.0).unsqueeze(1)
            value = where(drift, value + channel_size * elapsed, value)
            scale = hit & (code == FAULT_CODE["scale"])
            value = where(scale, value * self.scale.unsqueeze(1), value)
            stuck = hit & (code == FAULT_CODE["stuck"])
            value = where(stuck, self.stuck_value[name], value)
            outlier = hit & self.outlier_pending & valid & (code == FAULT_CODE["outlier_spike"])
            value = where(outlier, value + channel_size, value)
            self.outlier_pending &= ~outlier
            dropout = hit & in_burst & (code == FAULT_CODE["dropout_burst"])
            loss = hit & (code == FAULT_CODE["total_loss"])
            valid = valid & ~(dropout | loss)
            value = where(loss, torch.full_like(value, float("nan")), value)
            delayed = hit & in_burst & (code == FAULT_CODE["delay_spike"])
            if bool(delayed.any()):
                old_value, old_valid, old_stamp = self._delayed(name)
                value, valid, stamp = where(delayed, old_value, value), where(delayed, old_valid, valid), where(delayed, old_stamp, stamp)
            corrupting |= bias | drift | scale | stuck | outlier | dropout | loss | delayed
            # The stuck value is the last good value before the onset.
            self.stuck_value[name] = torch.where((measurement.valid & ~hit).unsqueeze(1), measurement.value, self.stuck_value[name])
            covariance = measurement.covariance
            bus.write(key, Measurement(value=value, valid=valid, stamp=stamp, covariance=covariance), self)
        self.corrupting_now = corrupting
        self.buffer_index = (self.buffer_index + 1) % self.buffer_length
        bus.write(
            MessageKey.FAULT_STATUS,
            FaultStatus(
                fault_type=self.fault_type.clone(), target=self.target.clone(), onset_altitude=self.onset_altitude.clone(),
                active=self.active.clone(), onset_time=self.onset_time.clone(), corrupting_now=corrupting.clone(),
            ),
            self,
        )

    def _generator(self) -> torch.Generator:
        """Return the slot's generator (a fresh one before the first reset)."""
        if self.generator is None:
            self.generator = torch.Generator(device=self.device)
        return self.generator
