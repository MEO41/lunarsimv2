# SPDX-License-Identifier: BSD-3-Clause
"""M5 L2 smoke test (GPU): S8 with the plume grains in shadow.

Four landers fly 400 control steps (8 s) of ``s8_plume_check_l2``. Checks: the regolith state
carries the grain state; landers whose nozzle is low enough have eroded soil and launched
particles; in shadow mode the dust fault still gets L1's dust density; the grain particles stay
within the cap; the slot-loop check passed (no flight-side binding reads truth).
"""

import numpy
import torch
from isaac_test_support import build_agent, make_environment

from lunarsim.core.message_types import MessageKey
from lunarsim.tools.episode_runner import run_agent_in_environment
from lunarsim.tools.replay_recording import ReplayRecorder


class _RegolithReader:
    """Test-side reader of the regolith state (truth side, like the dust metric)."""

    requires = frozenset({MessageKey.REGOLITH_STATE})
    provides = frozenset()
    reads_truth = True


def test_grains_run_in_shadow_of_the_l1_plume(tmp_path):
    """The grains erode and fly while L1 keeps feeding the dust fault; the replay recorder keeps the chosen landers' grains."""
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

    recorder = ReplayRecorder(environment, every_n_steps=2, grain_landers=[0, 2])

    def check_and_record(step_index: int) -> None:
        """Run the checks, then record the replay frame (the recorder does not use the command)."""
        check(step_index)
        recorder.record(step_index, None)

    run_agent_in_environment(environment, agent, number_of_steps=400, after_step=check_and_record)
    surface = environment.bindings["surface"]
    capacity = surface.grain_model.capacity
    environment.close()
    assert bool((seen["eroded"] > 0.0).any())
    assert 0.0 < seen["particles"] <= capacity
    # Shadow mode: the published dust is L1's (it saturates near the ground), not the grains' (thin cloud).
    assert seen["published_dust"] > 0.5 > seen["grain_dust"]
    # The replay holds the grains of the chosen landers in the asked order, within the cap.
    path = recorder.save(tmp_path / "grains.npz", [2, 0], "s8_plume_check_l2", [], ["a", "b"], ["", ""], frames_per_second=25.0)
    replay = numpy.load(path)
    columns = [str(name) for name in replay["columns"]]
    offsets, frames = replay["grain_offsets"], replay["frames"]
    assert offsets.shape[1:] == (2, capacity, 4) and replay["grain_bins"].shape == offsets.shape[:3]
    live = numpy.isfinite(offsets[..., 0]).sum(axis=2)
    assert live.max() <= capacity
    assert numpy.array_equal(live, frames[:, :, columns.index("grain_particles")].astype(int))
    assert numpy.nanmin(offsets[..., 3]) >= 0.0  # live parcels are above the ground
