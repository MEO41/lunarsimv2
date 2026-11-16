# SPDX-License-Identifier: BSD-3-Clause
"""``nav: truth_noise`` - the true state plus a set navigation error (design S5 variant; orbit phases of S7).

Design S5: "Change ``nav`` to ``truth_noise`` with the same error level as TRN, to see if TRN
error structure matters or only its size." The estimate is the true state plus, per axis:

- a bias drawn once per episode (``position_bias_m``, ``velocity_bias_mps``, one sigma);
- white noise drawn every control step (``position_noise_m``, ``velocity_noise_mps``).

Optional ``vertical_position_bias_m``: the bias along the local vertical gets its own sigma (the
radial direction from the Moon's centre in MCI and MCMF, +z in SITE); the other two directions keep
``position_bias_m``. Use it once an altimeter fixes the height (Apollo's landing radar updated the
altitude from about 39,000 ft, [D-6846] Table I).

Attitude, body rates and mass are passed on unchanged. The state keeps its frame and its dtype
(float64 in the orbit frames), so guidance in MCI or MCMF keeps its precision. Like ``nav: truth``
it reads ``TruthState`` and is a reference binding, not flight software.
"""

from __future__ import annotations

import torch
from torch import Tensor

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.message_types import Frame, MessageKey, NavEstimate, NavigationStateLayout, TruthState
from lunarsim.core.registry import register_binding


def truth_vector(truth: TruthState, dtype) -> Tensor:
    """Return the true state in NavigationStateLayout order, (N, 16), in ``dtype``."""
    return torch.cat(
        (
            truth.position.to(dtype), truth.velocity.to(dtype), truth.orientation_quaternion.to(dtype),
            truth.angular_velocity.to(dtype), truth.mass.unsqueeze(1).to(dtype),
            truth.footpad_height_above_ground.unsqueeze(1).to(dtype), truth.surface_probe_contact.unsqueeze(1).to(dtype),
        ),
        dim=1,
    )


@register_binding(SlotName.NAVIGATION, "truth_noise")
class TruthNoiseNavigation(BindingBase):
    """True state plus a per-episode bias and per-step white noise on position and velocity."""

    requires = frozenset({MessageKey.TRUTH_STATE})
    provides = frozenset({MessageKey.NAVIGATION_ESTIMATE})
    reads_truth = True
    health_dimension = 0

    def __init__(self, parameters, context) -> None:
        """Read the error sizes and allocate the biases."""
        super().__init__(parameters, context)
        get = self.parameters.get
        self.position_bias_sigma = float(get("position_bias_m", 0.0))
        self.velocity_bias_sigma = float(get("velocity_bias_mps", 0.0))
        self.position_noise_sigma = float(get("position_noise_m", 0.0))
        self.velocity_noise_sigma = float(get("velocity_noise_mps", 0.0))
        vertical = get("vertical_position_bias_m")
        self.vertical_bias_sigma = None if vertical is None else float(vertical)
        count = self.number_of_environments
        self.vertical_bias = torch.zeros(count, dtype=torch.float64, device=self.device)
        self.position_bias = torch.zeros(count, 3, dtype=torch.float64, device=self.device)
        self.velocity_bias = torch.zeros(count, 3, dtype=torch.float64, device=self.device)
        self.generator: torch.Generator | None = None

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Draw new biases for ``env_ids``."""
        self.generator = gen
        count = len(env_ids)
        self.position_bias[env_ids] = (self.position_bias_sigma * torch.randn(count, 3, generator=gen, device=self.device)).to(torch.float64)
        self.velocity_bias[env_ids] = (self.velocity_bias_sigma * torch.randn(count, 3, generator=gen, device=self.device)).to(torch.float64)
        if self.vertical_bias_sigma is not None:
            self.vertical_bias[env_ids] = (self.vertical_bias_sigma * torch.randn(count, generator=gen, device=self.device)).to(torch.float64)

    def position_bias_now(self, truth: TruthState) -> Tensor:
        """Return the position bias (N, 3, float64), with its local-vertical part replaced when ``vertical_position_bias_m`` is set."""
        if self.vertical_bias_sigma is None:
            return self.position_bias
        if truth.frame is Frame.SITE:
            up = torch.zeros_like(self.position_bias)
            up[:, 2] = 1.0
        else:
            up = torch.nn.functional.normalize(truth.position.to(torch.float64), dim=1)
        along = (self.position_bias * up).sum(dim=1, keepdim=True)
        return self.position_bias - along * up + self.vertical_bias.unsqueeze(1) * up

    def step(self, bus, dt: float) -> None:
        """Write the true state plus bias and noise, in the truth's frame and dtype (float32 at the site)."""
        truth: TruthState = bus.read(MessageKey.TRUTH_STATE, self)
        if self.generator is None:
            self.generator = torch.Generator(device=self.device)
        dtype = torch.float32 if truth.frame is Frame.SITE else torch.float64
        mean = truth_vector(truth, dtype)
        count = mean.shape[0]
        layout = NavigationStateLayout
        position_noise = self.position_noise_sigma * torch.randn(count, 3, generator=self.generator, device=self.device)
        velocity_noise = self.velocity_noise_sigma * torch.randn(count, 3, generator=self.generator, device=self.device)
        mean[:, layout.POSITION] += (self.position_bias_now(truth) + position_noise.to(torch.float64)).to(dtype)
        mean[:, layout.VELOCITY] += (self.velocity_bias + velocity_noise.to(torch.float64)).to(dtype)
        variance = torch.zeros(count, layout.DIMENSION, dtype=dtype, device=self.device)
        variance[:, layout.POSITION] = self.position_bias_sigma**2 + self.position_noise_sigma**2
        variance[:, layout.VELOCITY] = self.velocity_bias_sigma**2 + self.velocity_noise_sigma**2
        estimate = NavEstimate(mean=mean, covariance=variance, health=torch.zeros(count, 0, dtype=dtype, device=self.device))
        bus.write(MessageKey.NAVIGATION_ESTIMATE, estimate, self)
