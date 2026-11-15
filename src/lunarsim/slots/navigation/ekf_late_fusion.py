# SPDX-License-Identifier: BSD-3-Clause
"""``nav: ekf_late_fusion`` - the EKF that applies late samples at their capture time (milestone 3, S4).

The design: "Keeps a buffer of past states. A late measurement is applied at its capture time,
then the filter replays to now." Every control step the filter stores its inputs (the IMU sample
and the altimeter and Doppler messages) and its state after the step, for the last
``buffer_seconds``. A sample is late when its capture ``stamp`` is older than the current step
by more than half a step. Then:

1. the sample is written into the input slot of its capture step (as an extra sample there);
2. the filter goes back to its state just before the oldest capture step among the late samples;
3. it replays every stored step up to now (propagation, the on-time samples of that step and
   any late samples inserted there).

Replays use the stored inputs only, so an environment without late samples gets the same result
it had; the replay runs for all environments at once (batched), from the oldest capture step.
A late sample is used only if it is newer than the last sample already fused from that sensor
(out-of-order older samples are dropped), and samples older than the buffer are dropped too.
With no late data the filter is exactly ``ekf``. Parameter: ``late_fusion.buffer_seconds``.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.binding import SlotName
from lunarsim.core.message_types import MessageKey, SurfaceGeometry
from lunarsim.core.registry import register_binding
from lunarsim.slots.navigation.ekf import ExtendedKalmanFilterNavigation
from lunarsim.slots.sensors.terminal_suite import DOPPLER_VELOCIMETER_KEY, IMU_KEY, RADAR_ALTIMETER_KEY

SENSOR_KEYS = (RADAR_ALTIMETER_KEY, DOPPLER_VELOCIMETER_KEY)


class StoredMessage:
    """One message to fuse in a stored step; ``has_samples`` is False when no environment has a valid sample."""

    def __init__(self, value: Tensor, valid: Tensor, stamp: Tensor, covariance: Tensor, has_samples: bool) -> None:
        """Keep the tensors and the flag."""
        self.value, self.valid, self.stamp, self.covariance, self.has_samples = value, valid, stamp, covariance, has_samples

    def merge(self, value: Tensor, valid: Tensor, stamp: Tensor, covariance: Tensor) -> None:
        """Add the samples of other environments (where ``valid``) to this message."""
        row = valid.unsqueeze(1)
        self.value = torch.where(row, value, self.value)
        self.covariance = torch.where(row, covariance, self.covariance)
        self.stamp = torch.where(valid, stamp, self.stamp)
        self.valid = self.valid | valid
        self.has_samples = True


class StepRecord:
    """Inputs of one control step: the IMU sample, and per aided sensor the on-time and the late message."""

    def __init__(self) -> None:
        """Start empty."""
        self.imu_value: Tensor | None = None
        self.imu_valid: Tensor | None = None
        # Per aided sensor: the on-time message and the late samples inserted into this step (one
        # merged message, or None), fused in this order.
        self.on_time: list[StoredMessage | None] = [None, None]
        self.late: list[StoredMessage | None] = [None, None]

    def messages(self, sensor: int) -> list[StoredMessage]:
        """Return the messages of a sensor that hold at least one valid sample."""
        return [message for message in (self.on_time[sensor], self.late[sensor]) if message is not None and message.has_samples]


@register_binding(SlotName.NAVIGATION, "ekf_late_fusion")
class LateFusionKalmanFilterNavigation(ExtendedKalmanFilterNavigation):
    """``ekf`` plus a buffer of past inputs and states, and replay for late samples."""

    def __init__(self, parameters, context) -> None:
        """Size the buffer from ``late_fusion.buffer_seconds`` and the control step."""
        super().__init__(parameters, context)
        dt = float(context.shared_data["control_step_duration"])
        self.dt = dt
        self.length = int(round(float(self.parameters.get("late_fusion", {}).get("buffer_seconds", 0.6)) / dt)) + 1
        self.records: list[StepRecord | None] = [None] * self.length
        self.snapshots: list[tuple | None] = [None] * self.length  # filter state after each stored step
        self.newest = -1  # index of the newest stored step (absolute step counter)
        self.replays = 0  # number of replays (for tests and logs)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Start the given environments; their stored past is no longer theirs."""
        super().reset(env_ids, gen)
        for record in self.records:
            if record is None:
                continue
            if record.imu_valid is not None:
                record.imu_valid[env_ids] = False
            for message in (*record.on_time, *record.late):
                if message is not None:
                    message.valid[env_ids] = False
        for index, snapshot in enumerate(self.snapshots):
            if snapshot is not None:
                self.snapshots[index] = self._overwrite(snapshot, env_ids)

    # ---- state snapshots ----

    def _snapshot(self) -> tuple:
        """Return a copy of everything the replay must restore."""
        state = self.filter
        return (
            state.position.clone(), state.velocity.clone(), state.orientation.clone(), state.accelerometer_bias.clone(),
            state.gyro_bias.clone(), state.scaled_covariance.clone(), self.body_rates.clone(), self.last_fused_stamp.clone(),
        )

    def _restore(self, snapshot: tuple, where: Tensor) -> None:
        """Restore the snapshot for the environments in ``where``."""
        state = self.filter
        names = ("position", "velocity", "orientation", "accelerometer_bias", "gyro_bias", "scaled_covariance")
        for name, saved in zip(names, snapshot[:6]):
            current = getattr(state, name)
            mask = where.view(-1, *([1] * (current.dim() - 1)))
            setattr(state, name, torch.where(mask, saved, current))
        self.body_rates = torch.where(where.unsqueeze(1), snapshot[6], self.body_rates)
        self.last_fused_stamp = torch.where(where.unsqueeze(1), snapshot[7], self.last_fused_stamp)

    @staticmethod
    def _merge(old: tuple | None, new: tuple, where: Tensor) -> tuple:
        """Return ``new`` for the environments in ``where`` and ``old`` for the others."""
        if old is None:
            return new
        return tuple(torch.where(where.view(-1, *([1] * (fresh.dim() - 1))), fresh, kept) for kept, fresh in zip(old, new))

    def _overwrite(self, snapshot: tuple, env_ids: Tensor) -> tuple:
        """Return the snapshot with the current (just reset) state of ``env_ids`` written into it."""
        current = self._snapshot()
        result = []
        for saved, now in zip(snapshot, current):
            saved = saved.clone()
            saved[env_ids] = now[env_ids]
            result.append(saved)
        return tuple(result)

    # ---- the step ----

    def step(self, bus, dt: float) -> None:
        """Store this step's inputs, insert late samples at their capture steps, replay if needed."""
        geometry: SurfaceGeometry = bus.read(MessageKey.SURFACE_GEOMETRY, self)
        self.newest += 1
        slot = self.newest % self.length
        record = StepRecord()
        imu = bus.read(IMU_KEY, self)
        record.imu_value, record.imu_valid = imu.value.clone(), imu.valid.clone()
        oldest_depth = torch.zeros(self.number_of_environments, dtype=torch.long, device=self.device)
        late_samples = []
        for sensor, key in enumerate(SENSOR_KEYS):
            message = bus.read(key, self)
            age = self.time - message.stamp
            depth = torch.round(age / dt).long()
            late = message.valid & (depth >= 1)
            usable = late & (depth <= min(self.newest - 1, self.length - 1)) & (message.stamp > self.last_fused_stamp[:, sensor])
            on_time = message.valid & ~late
            record.on_time[sensor] = StoredMessage(
                message.value.clone(), on_time, message.stamp.clone(), message.covariance.clone(), bool(on_time.any())
            )
            if bool(usable.any()):
                late_samples.append((sensor, message, depth, usable))
                oldest_depth = torch.maximum(oldest_depth, torch.where(usable, depth, torch.zeros_like(depth)))
        self.records[slot] = record

        if not late_samples:
            self._run_step(record, geometry, dt)
        else:
            depth = int(oldest_depth.max())
            for sensor, message, sample_depth, usable in late_samples:
                for back in range(1, depth + 1):
                    chosen = usable & (sample_depth == back)
                    capture = self.records[(self.newest - back) % self.length]
                    if capture.late[sensor] is None:
                        capture.late[sensor] = StoredMessage(
                            message.value.clone(), chosen, message.stamp.clone(), message.covariance.clone(), True
                        )
                    else:
                        capture.late[sensor].merge(message.value, chosen, message.stamp, message.covariance)
            replaying = oldest_depth > 0
            start_snapshot = self.snapshots[(self.newest - depth - 1) % self.length]
            if start_snapshot is not None:
                self._restore(start_snapshot, replaying)
            current_time = self.time.clone()
            self.time = self.time - depth * dt
            for back in range(depth, -1, -1):
                index = (self.newest - back) % self.length
                self._run_step(self.records[index], geometry, dt, replay_mask=replaying if back > 0 else None)
                if back > 0:
                    # Only the replaying environments get a new past; the others keep theirs.
                    self.snapshots[index] = self._merge(self.snapshots[index], self._snapshot(), replaying)
                    self.time = self.time + dt
            self.time = current_time
            self.replays += 1
        self.snapshots[slot] = self._snapshot()
        self.finish_step(bus, geometry, dt)

    def _run_step(self, record: StepRecord, geometry: SurfaceGeometry, dt: float, replay_mask: Tensor | None = None) -> None:
        """Propagate with the stored IMU sample and fuse the stored messages of one step.

        During a replay (``replay_mask``), only the replaying environments change.
        """
        def mask(valid: Tensor) -> Tensor:
            """Restrict a validity mask to the replaying environments during a replay."""
            return valid if replay_mask is None else valid & replay_mask

        self.propagate_with(record.imu_value, mask(record.imu_valid), dt)
        # Messages without any valid sample are skipped (in S4 every Doppler sample is late, so
        # the on-time Doppler message of a stored step is empty).
        for message in record.messages(0):
            self.fuse_altimeter(geometry, message.value, mask(message.valid), message.stamp, message.covariance)
        for message in record.messages(1):
            self.fuse_doppler(geometry, message.value, mask(message.valid), message.stamp, message.covariance)
