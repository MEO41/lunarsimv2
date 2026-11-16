# SPDX-License-Identifier: BSD-3-Clause
"""``start: from_log`` - start every environment from a state saved at a gate by an earlier phase (S5, S7).

Design S7: "The saved states connect S7 to the other scenarios. S5 can start from the states saved
at PDI. S1, S3 and S8 can start from the states saved at the low gate. Thus, short scenarios use
the true dispersion of the full mission, not a guessed box."

The log is the episode table of a phase that ended at a gate (``metrics: phase_gate``): one row per
environment with ``gate_frame_code`` and the ``gate_`` state columns (float64). Environment ``i`` takes
row ``i`` (rows repeat when the log is shorter). ``convert_to`` changes the frame on the way:

- ``keep``: the saved frame (the orbit plant converts it to its own frame itself);
- ``SITE``: the **handoff** to the PhysX site plant, float64 Moon-fixed to float32 site frame
  (design: "the handoff converts the state from the Moon-fixed frame to the site frame and from
  float64 to float32. A check stops the run if position, velocity or energy jump at the
  handoff"). The converted state is converted back and compared with the saved one; if the
  position, velocity or specific energy differ by more than ``handoff_tolerance`` the run stops.
  The report is kept in ``handoff_report``.

Parameters: ``log_path``, ``convert_to`` (``keep`` or ``SITE``), ``site``, ``handoff_tolerance``
(``position_m`` 0.01, ``velocity_mps`` 1e-4, ``energy_fraction`` 1e-7), ``nav_error``.
"""

from __future__ import annotations

from pathlib import Path

import pandas
import torch
from torch import Tensor

from lunarsim.core.binding import BindingBase, SlotName
from lunarsim.core.frames import convert_truth_state, specific_orbital_energy
from lunarsim.core.message_types import Frame, MessageKey, TruthState
from lunarsim.core.registry import register_binding
from lunarsim.slots.plant.orbit_float64 import site_from_parameters
from lunarsim.slots.start.initial_navigation_error import InitialNavigationError

FRAME_BY_VALUE = {frame.value: frame for frame in Frame}
STATE_COLUMNS = {
    "position": ("gate_position_x_m", "gate_position_y_m", "gate_position_z_m"),
    "velocity": ("gate_velocity_x_mps", "gate_velocity_y_mps", "gate_velocity_z_mps"),
    "orientation": ("gate_orientation_w", "gate_orientation_x", "gate_orientation_y", "gate_orientation_z"),
    "rates": ("gate_body_rate_x_rad_s", "gate_body_rate_y_rad_s", "gate_body_rate_z_rad_s"),
}


class HandoffError(RuntimeError):
    """The state jumped at the handoff by more than the tolerance."""


def load_gate_states(path: Path | str, count: int, device) -> TruthState:
    """Return the saved gate states of a log file as a float64 TruthState with ``count`` rows (rows repeat if needed)."""
    return gate_states_from_table(pandas.read_parquet(path), count, device, source=str(path))


def gate_states_from_table(table: pandas.DataFrame, count: int, device, source: str = "table") -> TruthState:
    """Return the gate states of a phase-gate table as a float64 TruthState with ``count`` rows."""
    path = source
    table = table.sort_values("environment_index").reset_index(drop=True)
    if "gate_reached" in table:
        table = table[table["gate_reached"]].reset_index(drop=True)
    if table.empty:
        raise ValueError(f"{path}: no environment reached the gate")
    from lunarsim.slots.metrics.phase_gate import FRAME_CODES

    frames = {FRAME_CODES[int(code)] for code in table["gate_frame_code"]}
    if len(frames) != 1:
        raise ValueError(f"{path}: states in more than one frame {sorted(frames)}")
    rows = torch.arange(count) % len(table)
    def column_block(names):
        """Return the given columns as a float64 tensor (count, len(names))."""
        values = torch.tensor(table[list(names)].to_numpy(), dtype=torch.float64)
        return values[rows].to(device)
    time = torch.tensor(table["gate_time_s"].to_numpy(), dtype=torch.float64)[rows].to(device)
    mass = torch.tensor(table["gate_mass_kg"].to_numpy(), dtype=torch.float64)[rows].to(device)
    return TruthState(
        frame=FRAME_BY_VALUE[frames.pop()],
        time=time,
        position=column_block(STATE_COLUMNS["position"]),
        velocity=column_block(STATE_COLUMNS["velocity"]),
        orientation_quaternion=torch.nn.functional.normalize(column_block(STATE_COLUMNS["orientation"]), dim=1),
        angular_velocity=column_block(STATE_COLUMNS["rates"]),
        mass=mass,
        contact=torch.zeros(count, 0, dtype=torch.float64, device=device),
    )


@register_binding(SlotName.START, "from_log")
class FromLogStart(BindingBase):
    """Start states read from a gate log, optionally handed off to the site frame (see the module text)."""

    provides = frozenset({MessageKey.START_STATE, MessageKey.INITIAL_NAVIGATION_STATE})

    def __init__(self, parameters, context) -> None:
        """Load the log, convert it, and check the handoff once."""
        super().__init__(parameters, context)
        get = self.parameters.get
        if not get("log_path"):
            raise ValueError("start: from_log needs log_path (the gate log of the previous phase)")
        self.site = site_from_parameters(get("site"))
        saved = load_gate_states(get("log_path"), self.number_of_environments, self.device)
        self.convert_to = str(get("convert_to", "keep"))
        self.handoff_report: dict | None = None
        if self.convert_to == "SITE":
            self.states = self._handoff(saved, dict(get("handoff_tolerance") or {}))
        elif self.convert_to == "keep":
            self.states = saved
        else:
            raise ValueError("start.convert_to must be 'keep' or 'SITE'")
        preset = context.shared_data["lander_preset"]
        self.center_of_mass_height = preset.center_of_mass_height_above_footpad_plane
        self.navigation_error = InitialNavigationError(get("nav_error"), self.number_of_environments, self.device)

    def _handoff(self, saved: TruthState, tolerance: dict) -> TruthState:
        """Convert float64 Moon-fixed states to float32 site states and check that nothing jumps."""
        site_state = convert_truth_state(saved, Frame.SITE, self.site)
        single = TruthState(
            frame=Frame.SITE, time=site_state.time.to(torch.float32), position=site_state.position.to(torch.float32),
            velocity=site_state.velocity.to(torch.float32), orientation_quaternion=site_state.orientation_quaternion.to(torch.float32),
            angular_velocity=site_state.angular_velocity.to(torch.float32), mass=site_state.mass.to(torch.float32),
            contact=torch.zeros(saved.position.shape[0], 0, device=self.device),
        )
        back = convert_truth_state(
            TruthState(
                frame=Frame.SITE, time=saved.time, position=single.position.to(torch.float64), velocity=single.velocity.to(torch.float64),
                orientation_quaternion=single.orientation_quaternion.to(torch.float64), angular_velocity=single.angular_velocity.to(torch.float64),
                mass=single.mass.to(torch.float64), contact=saved.contact,
            ),
            saved.frame, self.site,
        )
        reference = convert_truth_state(saved, saved.frame, self.site)
        position_jump = float((back.position - reference.position).norm(dim=1).max())
        velocity_jump = float((back.velocity - reference.velocity).norm(dim=1).max())
        energy_before = specific_orbital_energy(reference, self.site)
        energy_after = specific_orbital_energy(back, self.site)
        energy_jump = float(((energy_after - energy_before) / energy_before.abs()).abs().max())
        self.handoff_report = {"position_jump_m": position_jump, "velocity_jump_mps": velocity_jump, "energy_jump_fraction": energy_jump}
        limits = {"position_m": 0.01, "velocity_mps": 1.0e-4, "energy_fraction": 1.0e-7}
        limits.update({name: float(value) for name, value in tolerance.items()})
        if position_jump > limits["position_m"] or velocity_jump > limits["velocity_mps"] or energy_jump > limits["energy_fraction"]:
            raise HandoffError(f"state jumped at the handoff: {self.handoff_report} (limits {limits})")
        return single

    def reset(self, env_ids: Tensor, gen: torch.Generator) -> None:
        """Draw the navigation error of ``env_ids``; the start states themselves are fixed by the log."""
        self.navigation_error.reset(env_ids, gen)

    def step(self, bus, dt: float) -> None:
        """Write the logged start states (time 0 in the site frame, the saved time otherwise)."""
        state = self.states
        if state.frame is Frame.SITE:
            state = TruthState(
                frame=Frame.SITE, time=torch.zeros_like(state.time), position=state.position.clone(), velocity=state.velocity.clone(),
                orientation_quaternion=state.orientation_quaternion.clone(), angular_velocity=state.angular_velocity.clone(),
                mass=state.mass.clone(), contact=state.contact.clone(),
            )
        bus.write(MessageKey.START_STATE, state, self)
        self.navigation_error.write(bus, state, self)
