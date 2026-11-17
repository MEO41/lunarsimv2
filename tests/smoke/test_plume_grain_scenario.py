# SPDX-License-Identifier: BSD-3-Clause
"""M5 L2 smoke test (GPU): S8 with the plume grains in shadow.

Four landers fly 400 control steps (8 s) of ``s8_plume_check_l2``. Checks: the regolith state
carries the grain state; landers whose nozzle is low enough have eroded soil and launched
particles; in shadow mode the dust fault still gets L1's dust density; the grain particles stay
within the cap; the slot-loop check passed (no flight-side binding reads truth).
"""

import torch
from isaac_test_support import build_agent, make_environment

from lunarsim.core.message_types import MessageKey
from lunarsim.tools.episode_runner import run_agent_in_environment


class _RegolithReader:
    """Test-side reader of the regolith state (truth side, like the dust metric)."""

    requires = frozenset({MessageKey.REGOLITH_STATE})
    provides = frozenset()
    reads_truth = True


def test_grains_run_in_shadow_of_the_l1_plume():
    """The grains erode and fly while L1 keeps feeding the dust fault."""
    environment = make_environment("s8_plume_check_l2", ["env.num_envs=4"])
    agent = build_agent(environment)
    reader = _RegolithReader()
    seen = {"eroded": torch.zeros(4, device=environment.device), "particles": 0.0, "published_dust": 0.0, "grain_dust": 0.0}

    def check(step_index: int) -> None:
        """Track the grain state; check that the published dust is L1's."""
        regolith = environment.bus.read(MessageKey.REGOLITH_STATE, reader)
        grains = regolith.grains
        assert grains is not None
        seen["eroded"] = torch.maximum(seen["eroded"], grains.eroded_mass)
        seen["particles"] = max(seen["particles"], float(grains.particles.max()))
        seen["published_dust"] = max(seen["published_dust"], float(regolith.dust_density.max()))
        seen["grain_dust"] = max(seen["grain_dust"], float(grains.dust_density.max()))

    run_agent_in_environment(environment, agent, number_of_steps=400, after_step=check)
    surface = environment.bindings["surface"]
    capacity = surface.grain_model.capacity
    environment.close()
    assert bool((seen["eroded"] > 0.0).any())
    assert 0.0 < seen["particles"] <= capacity
    # Shadow mode: the published dust is L1's (it saturates near the ground), not the grains' (thin cloud).
    assert seen["published_dust"] > 0.5 > seen["grain_dust"]
