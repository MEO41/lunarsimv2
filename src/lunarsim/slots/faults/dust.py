# SPDX-License-Identifier: BSD-3-Clause
"""``faults: dust`` - blowing dust from the plume degrades the optical sensors (scenario S8, milestone 5).

Design S8: "the surface sends dust density to the faults slot. The dust fault then degrades LiDAR
and camera data. Thus, a physical effect becomes a sensor fault with no change to the sensor
bindings." The dust density (0..1) comes from ``regolith_state`` (written by the surface in the
previous step: the dust already in the air when the sensors sample).

- **flash LiDAR**: each pixel return is lost with probability ``density * lidar_dropout_at_full_dust``
  or, with probability ``density * lidar_false_return_at_full_dust``, replaced by a short return
  from the dust cloud (uniform between the sensor's minimum range and the true range);
- **Doppler velocimeter**: the velocimeter of S2 is a Doppler *lidar* (NDL, optical), so each
  beam's sample is lost with probability ``density * doppler_dropout_at_full_dust``;
- **radar altimeter**: unaffected (radar sees through the dust; Apollo's landing radar did).

All probabilities are model choices until the L2 and L3 levels give numbers (design S8). Apollo
11 shows the effect is real: "we were beginning to get a transparent sheet of moving dust that
obscured visibility a little bit [...] As we got lower, the visibility continued to decrease"
(Armstrong, technical debrief, Apollo Lunar Surface Journal, after 102:45:17). The stated
measurement variances are left unchanged: the filter does not know about the dust.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import Measurement, MessageKey, RegolithState
from lunarsim.core.registry import register_binding
from lunarsim.slots.sensors.terminal_suite import DOPPLER_VELOCIMETER_KEY, LIDAR_KEY


@register_binding(SlotName.FAULTS, "dust")
class DustFault(BindingBase):
    """LiDAR dropouts and false returns, and Doppler-lidar dropouts, scaled by the dust density."""

    requires = frozenset({MessageKey.REGOLITH_STATE, LIDAR_KEY, DOPPLER_VELOCIMETER_KEY})
    provides = frozenset({LIDAR_KEY, DOPPLER_VELOCIMETER_KEY})
    reads_truth = False

    def __init__(self, parameters, context) -> None:
        """Read the degradation strengths and the LiDAR minimum range."""
        super().__init__(parameters, context)
        get = self.parameters.get
        self.lidar_dropout = float(get("lidar_dropout_at_full_dust", 0.9))
        self.lidar_false_return = float(get("lidar_false_return_at_full_dust", 0.05))
        self.doppler_dropout = float(get("doppler_dropout_at_full_dust", 0.5))
        lidar = (context.shared_data.get("sensor_suite_parameters") or {}).get("lidar") or {}
        self.lidar_minimum_range = float(lidar.get("minimum_range_m", 0.0))
        self.generator: torch.Generator | None = None

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Use the faults slot's generator."""
        self.generator = gen

    def step(self, bus, dt: float) -> None:
        """Degrade this step's LiDAR and Doppler messages by the dust density."""
        if self.generator is None:
            self.generator = torch.Generator(device=self.device)
        regolith: RegolithState = bus.read(MessageKey.REGOLITH_STATE, self)
        density = regolith.dust_density.clamp(0.0, 1.0)
        bus.write(LIDAR_KEY, self.degrade_lidar(bus.read(LIDAR_KEY, self), density), self)
        bus.write(DOPPLER_VELOCIMETER_KEY, self.degrade_doppler(bus.read(DOPPLER_VELOCIMETER_KEY, self), density), self)

    def degrade_lidar(self, message: Measurement, density: Tensor) -> Measurement:
        """Return the LiDAR message with pixels lost or replaced by short dust returns."""
        if message.value.shape[1] == 0 or not bool((density > 0).any()):
            return message
        draw = torch.rand(message.value.shape, generator=self.generator, device=message.value.device)
        lose = draw < density.unsqueeze(1) * self.lidar_dropout
        false_return = ~lose & (draw < density.unsqueeze(1) * (self.lidar_dropout + self.lidar_false_return))
        short = self.lidar_minimum_range + torch.rand(message.value.shape, generator=self.generator, device=message.value.device) * (
            torch.nan_to_num(message.value, nan=0.0) - self.lidar_minimum_range
        ).clamp_min(0.0)
        value = torch.where(lose, torch.full_like(message.value, float("nan")), message.value)
        value = torch.where(false_return & torch.isfinite(message.value), short, value)
        valid = message.valid & torch.isfinite(value).any(dim=1)
        return Measurement(value=value, valid=valid, stamp=message.stamp, covariance=message.covariance)

    def degrade_doppler(self, message: Measurement, density: Tensor) -> Measurement:
        """Return the Doppler message with beams lost (velocity and range of a beam go together)."""
        if not bool((density > 0).any()):
            return message
        beams = message.value.shape[1] // 2
        lose = torch.rand(message.value.shape[0], beams, generator=self.generator, device=message.value.device) < density.unsqueeze(1) * self.doppler_dropout
        lose_both = torch.cat((lose, lose), dim=1)
        value = torch.where(lose_both, torch.full_like(message.value, float("nan")), message.value)
        valid = message.valid & torch.isfinite(value).any(dim=1)
        return Measurement(value=value, valid=valid, stamp=message.stamp, covariance=message.covariance)
