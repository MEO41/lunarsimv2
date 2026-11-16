# SPDX-License-Identifier: BSD-3-Clause
"""Helpers of the phase manager (milestone 6): mission files, gate summaries, decision-record merge.

A mission (``configs/missions/<name>.yaml``) is a list of phases, each a scenario file. The runner
(``scripts/run_mission.py``) runs them in order for all environments; each phase ends every
environment at its gate (``metrics: phase_gate``) and the next phase starts from the saved gate
states (``start: from_log``). This module has the parts that need no simulator.
"""

from __future__ import annotations

from pathlib import Path

import numpy
import pandas
import torch
import yaml

from lunarsim.core.frames import SiteLocation, convert_truth_state
from lunarsim.core.message_types import Frame

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def load_mission(name: str) -> dict:
    """Return the mission file ``configs/missions/<name>.yaml`` as a dictionary (checks the phase list)."""
    path = REPOSITORY_ROOT / "configs" / "missions" / f"{name}.yaml"
    with open(path) as file:
        mission = yaml.safe_load(file)
    phases = mission.get("phases") or []
    if not phases or any("name" not in phase or "scenario" not in phase for phase in phases):
        raise ValueError(f"{path}: every phase needs a name and a scenario")
    return mission


def merge_agent_decision_records(episodes: pandas.DataFrame, agent) -> pandas.DataFrame:
    """Add the agent's per-episode decision records (``drain_decision_records``) to the episode table."""
    if not hasattr(agent, "drain_decision_records") or episodes.empty:
        return episodes
    records = agent.drain_decision_records()
    if not records:
        return episodes
    decisions = pandas.DataFrame({name: values.numpy() for name, values in records.items()})
    merged = episodes.merge(decisions, on=["environment_index", "episode_number"], how="left")
    if {"decision_mass_estimate_kg", "start_mass_kg", "propellant_used_kg"} <= set(merged.columns):
        merged["divert_propellant_estimate_kg"] = merged["decision_mass_estimate_kg"] - (merged["start_mass_kg"] - merged["propellant_used_kg"])
    return merged


def gate_states_in_site_frame(table: pandas.DataFrame, site: SiteLocation) -> numpy.ndarray | None:
    """Return the gate positions and velocities of a phase-gate table in the site frame, (rows, 6), or None."""
    if "gate_frame_code" not in table or table.empty:
        return None
    from lunarsim.slots.start.from_log import gate_states_from_table

    reached = int(table["gate_reached"].sum()) if "gate_reached" in table else len(table)
    if reached == 0:
        return None
    state = gate_states_from_table(table, reached, "cpu")
    site_state = convert_truth_state(state, Frame.SITE, site)
    return torch.cat((site_state.position, site_state.velocity), dim=1).numpy()


def summarize_phase(name: str, table: pandas.DataFrame, site: SiteLocation) -> dict:
    """Return the S7 measures of one phase: gate reach, dispersion at the gate, propellant, time."""
    summary = {"phase": name, "landers": int(len(table))}
    if "gate_reached" in table:
        summary["gate_reached_fraction"] = float(table["gate_reached"].mean())
        summary["propellant_used_kg_mean"] = float(table["propellant_used_kg"].mean())
        summary["elapsed_s_mean"] = float(table["elapsed_s"].mean())
        states = gate_states_in_site_frame(table, site)
        if states is not None and len(states) > 1:
            mean, spread = states.mean(axis=0), states.std(axis=0, ddof=1)
            labels = ("east_m", "north_m", "up_m", "east_speed_mps", "north_speed_mps", "up_speed_mps")
            summary["gate_mean"] = {label: float(value) for label, value in zip(labels, mean)}
            summary["gate_dispersion_1sigma"] = {label: float(value) for label, value in zip(labels, spread)}
    else:
        summary["success_fraction"] = float(table["success"].mean())
        for column in ("propellant_used_kg", "touchdown_vertical_speed_mps", "divert_distance_m", "touchdown_site_safe"):
            if column in table:
                summary[f"{column}_mean"] = float(table[column].astype(float).mean())
        if {"touchdown_position_x_m", "touchdown_position_y_m"} <= set(table.columns):
            summary["landing_error_from_site_m_mean"] = float(numpy.hypot(table["touchdown_position_x_m"], table["touchdown_position_y_m"]).mean())
    return summary
