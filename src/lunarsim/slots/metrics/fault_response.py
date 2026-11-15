# SPDX-License-Identifier: BSD-3-Clause
"""``metrics: fault_response`` - the touchdown metric plus how the navigation met the fault (milestone 3).

Extra columns of every episode record (scenario S3, "time to detect"):

- ``fault_type`` (code, ``FAULT_TYPES``), ``fault_target`` (index into ``TARGET_NAMES``, -1 none),
  ``fault_onset_altitude_m``, ``fault_onset_time_s`` (-1 if it never started),
  ``fault_present_at_start`` (the fault was active from the first control step);
- ``fault_detected``: the navigation declared the faulted sensor failed after the onset;
  ``fault_detection_delay_s``: time from the onset to that declaration (-1 if never);
- ``false_alarm``: the navigation declared a sensor failed that was not faulted, or before the
  onset;
- ``fault_corrupting_steps``: control steps in which the fault changed the delivered data.

Detection is read from the health columns of the navigation estimate (failed flag per aided
sensor, see ``nav: ekf`` and ``nav: ekf_fdi``); a filter that never declares failures never
detects. Needs ``faults: schedule`` (message ``fault_status``).
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey
from lunarsim.core.registry import register_binding
from lunarsim.slots.faults.fault_schedule import TARGET_NAMES
from lunarsim.slots.metrics.touchdown import TouchdownMetrics
from lunarsim.slots.navigation.ekf import AIDED_SENSORS

# Health column of the failed flag of each fault target.
FAILED_COLUMN = {TARGET_NAMES.index(name): 2 * AIDED_SENSORS.index(name) for name in TARGET_NAMES}


@register_binding(SlotName.METRICS, "fault_response")
class FaultResponseMetrics(TouchdownMetrics):
    """Touchdown judgement plus fault onset, detection delay and false alarms."""

    requires = TouchdownMetrics.requires | {MessageKey.FAULT_STATUS}

    def __init__(self, parameters, context) -> None:
        """Allocate the per-episode fault records."""
        super().__init__(parameters, context)
        count, device = self.number_of_environments, self.device
        self.fault_type = torch.zeros(count, dtype=torch.long, device=device)
        self.fault_target = torch.full((count,), -1, dtype=torch.long, device=device)
        self.fault_onset_altitude = torch.zeros(count, device=device)
        self.fault_onset_time = torch.full((count,), -1.0, device=device)
        self.present_at_start = torch.zeros(count, dtype=torch.bool, device=device)
        self.detection_time = torch.full((count,), -1.0, device=device)
        self.false_alarm = torch.zeros(count, dtype=torch.bool, device=device)
        self.corrupting_steps = torch.zeros(count, device=device)
        self.steps_in_episode = torch.zeros(count, device=device)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Clear the fault records of the given environments."""
        super().reset(env_ids, gen)
        self.fault_onset_time[env_ids] = -1.0
        self.present_at_start[env_ids] = False
        self.detection_time[env_ids] = -1.0
        self.false_alarm[env_ids] = False
        self.corrupting_steps[env_ids] = 0.0
        self.steps_in_episode[env_ids] = 0.0

    def step(self, bus, dt: float) -> None:
        """Follow the fault and the navigation's failure flags, then judge the touchdown."""
        status = bus.read(MessageKey.FAULT_STATUS, self)
        health = bus.read(MessageKey.NAVIGATION_ESTIMATE, self).health
        self.fault_type, self.fault_target, self.fault_onset_altitude = status.fault_type, status.target, status.onset_altitude
        self.fault_onset_time = status.onset_time
        self.steps_in_episode += 1.0
        self.present_at_start |= status.active & (self.steps_in_episode <= 1.0)
        self.corrupting_steps += status.corrupting_now.float()
        if health.shape[1] >= 2 * len(AIDED_SENSORS):
            failed = health[:, 0::2] > 0.5  # (N, aided sensors)
            target_column = torch.zeros_like(failed)
            for target, column in FAILED_COLUMN.items():
                target_column[:, column // 2] |= status.target == target
            detected_now = (failed & target_column).any(dim=1) & status.active & (self.detection_time < 0.0)
            # the time on the bus is the previous step's estimate: use this step's truth time
            self.detection_time = torch.where(detected_now, self.previous_time_for_detection(), self.detection_time)
            alarm = (failed & ~target_column).any(dim=1) | ((failed & target_column).any(dim=1) & ~status.active)
            self.false_alarm |= alarm
        super().step(bus, dt)

    def previous_time_for_detection(self) -> Tensor:
        """Return the time of the control step whose estimate is on the bus (the previous step)."""
        truth = self.context.bus.read(MessageKey.TRUTH_STATE, self)
        return (truth.time - self.context.clock.control_step_duration).clamp_min(0.0)

    def extra_record_columns(self) -> dict[str, Tensor]:
        """Return the fault columns of the episode records."""
        detected = self.detection_time >= 0.0
        delay = torch.where(detected, self.detection_time - self.fault_onset_time, torch.full_like(self.detection_time, -1.0))
        return {
            "fault_type": self.fault_type,
            "fault_target": self.fault_target,
            "fault_onset_altitude_m": self.fault_onset_altitude,
            "fault_onset_time_s": self.fault_onset_time,
            "fault_present_at_start": self.present_at_start,
            "fault_detected": detected,
            "fault_detection_delay_s": delay,
            "false_alarm": self.false_alarm,
            "fault_corrupting_steps": self.corrupting_steps,
        }
