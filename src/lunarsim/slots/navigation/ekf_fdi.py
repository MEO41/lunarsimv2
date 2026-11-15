# SPDX-License-Identifier: BSD-3-Clause
"""``nav: ekf_fdi`` - the EKF with fault detection and isolation (milestone 3, design: chi-square gate).

Each new sample of an aided sensor (radar altimeter, Doppler velocimeter) is tested before it
is fused:

1. **Age check**: a sample whose capture time is older than ``maximum_sample_age_s`` is dropped
   (late data; ``ekf_late_fusion`` would use it at its capture time instead).
2. **Chi-square gate**: the normalised innovation squared NIS = y^T S^-1 y of the sample's used
   values is compared with the chi-square quantile ``gate_probability`` for that many values. A
   sample above it is rejected.
3. **Isolation**: ``failure_after_rejections`` rejected samples in a row declare the sensor
   failed; a failed sensor is still tested every sample but not fused, and
   ``recovery_after_acceptances`` passing samples in a row declare it healthy again.
4. **Loss**: no valid sample for ``loss_after_s`` while a ground return is expected (predicted
   beam inside the sensor's range limits) also declares the sensor failed. Near the ground the
   altimeter (3.05 m) and the Doppler (2 m) stop by design; that is not a failure.

The health columns of the estimate give, per aided sensor, the failed flag (0 or 1) and the data
age (time since the capture of the last fused sample). All thresholds are **model choices**
(``configs/nav/ekf_fdi.yaml``).
"""

from __future__ import annotations

import torch
from scipy.stats import chi2
from torch import Tensor

from lunarsim.core.binding import SlotName
from lunarsim.core.registry import register_binding
from lunarsim.slots.navigation.ekf import AIDED_SENSORS, ExtendedKalmanFilterNavigation

MAXIMUM_VALUES_PER_SAMPLE = 6


@register_binding(SlotName.NAVIGATION, "ekf_fdi")
class FaultDetectingKalmanFilterNavigation(ExtendedKalmanFilterNavigation):
    """The ``ekf`` filter plus an age check, a chi-square gate and per-sensor isolation."""

    def __init__(self, parameters, context) -> None:
        """Read the detection settings and build the chi-square thresholds."""
        super().__init__(parameters, context)
        detection = dict(self.parameters["detection"])
        probability = float(detection["gate_probability"])
        thresholds = [float("inf")] + [float(chi2.ppf(probability, dof)) for dof in range(1, MAXIMUM_VALUES_PER_SAMPLE + 1)]
        self.thresholds = torch.tensor(thresholds, device=self.device)
        self.failure_after = int(detection["failure_after_rejections"])
        self.recovery_after = int(detection["recovery_after_acceptances"])
        self.maximum_sample_age = float(detection["maximum_sample_age_s"])
        self.loss_after = float(detection["loss_after_s"])
        count, sensors = self.number_of_environments, len(AIDED_SENSORS)
        self.rejections_in_a_row = torch.zeros(count, sensors, dtype=torch.long, device=self.device)
        self.acceptances_in_a_row = torch.zeros(count, sensors, dtype=torch.long, device=self.device)
        self.failed = torch.zeros(count, sensors, dtype=torch.bool, device=self.device)
        self.last_valid_time = torch.zeros(count, sensors, device=self.device)
        self.last_normalised_innovation = torch.zeros(count, sensors, device=self.device)

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Start the given environments with every sensor healthy."""
        super().reset(env_ids, gen)
        self.rejections_in_a_row[env_ids] = 0
        self.acceptances_in_a_row[env_ids] = 0
        self.failed[env_ids] = False
        self.last_valid_time[env_ids] = 0.0
        self.last_normalised_innovation[env_ids] = 0.0

    def declared_failed(self) -> Tensor:
        """Return the failed flags as floats (N, aided sensors)."""
        return self.failed.float()

    def fuse(
        self, sensor: int, residual: Tensor, jacobian: Tensor, variance: Tensor, use: Tensor, stamp: Tensor, sample: Tensor, expected: Tensor
    ) -> None:
        """Age check, gate, update the isolation counters, and fuse only accepted samples of healthy sensors."""
        fresh = (self.time - stamp) <= self.maximum_sample_age + 1e-6
        tested = use.any(dim=1) & fresh
        nis, dof = self.filter.normalised_innovation(residual, jacobian, variance, use)
        passes = nis <= self.thresholds[dof.clamp(max=MAXIMUM_VALUES_PER_SAMPLE)]
        self.last_normalised_innovation[:, sensor] = torch.where(tested, nis, self.last_normalised_innovation[:, sensor])
        accepted, rejected = tested & passes, tested & ~passes
        self.rejections_in_a_row[:, sensor] = torch.where(rejected, self.rejections_in_a_row[:, sensor] + 1, torch.where(accepted, 0, self.rejections_in_a_row[:, sensor]))
        self.acceptances_in_a_row[:, sensor] = torch.where(accepted, self.acceptances_in_a_row[:, sensor] + 1, torch.where(rejected, 0, self.acceptances_in_a_row[:, sensor]))
        self.last_valid_time[:, sensor] = torch.where(sample | ~expected, self.time, self.last_valid_time[:, sensor])
        lost = (self.time - self.last_valid_time[:, sensor]) > self.loss_after
        becomes_failed = (self.rejections_in_a_row[:, sensor] >= self.failure_after) | lost
        recovers = self.failed[:, sensor] & (self.acceptances_in_a_row[:, sensor] >= self.recovery_after) & ~lost
        self.failed[:, sensor] = (self.failed[:, sensor] | becomes_failed) & ~recovers
        fused = accepted & ~self.failed[:, sensor]
        super().fuse(sensor, residual, jacobian, variance, use & fused.unsqueeze(1), stamp, sample, expected)
