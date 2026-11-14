# SPDX-License-Identifier: BSD-3-Clause
"""Group the failed episodes of a run by cause, using only the run files."""

from __future__ import annotations

import numpy
import pandas

from lunarsim.core.physical_constants import MOON_SURFACE_GRAVITY


def lowest_possible_contact_speed(episodes: pandas.DataFrame, manifest: dict) -> pandas.Series:
    """Return the descent speed at contact that an ideal lander would still have, per episode.

    Ideal means: upright, full throttle from the first instant, no thrust lag. With the largest
    net deceleration ``a = F_max / m - g`` the speed at the ground is ``sqrt(v0^2 - 2 a h)``
    (zero if the lander can stop above the ground). No controller can do better than this.
    """
    lander = manifest["resolved_scenario"]["lander"]
    maximum_thrust = (
        lander["descent_engine"]["rated_thrust_n"]["value"] * lander["descent_engine"]["maximum_throttle_fraction"]["value"]
    )
    deceleration = maximum_thrust / episodes["start_mass_kg"] - MOON_SURFACE_GRAVITY
    squared = episodes["start_vertical_speed_mps"].clip(upper=0.0) ** 2 - 2.0 * deceleration * episodes["start_altitude_m"]
    return numpy.sqrt(squared.clip(lower=0.0))


def add_failure_groups(episodes: pandas.DataFrame, manifest: dict) -> pandas.DataFrame:
    """Return a copy of the episode table with the columns used to group failures.

    ``failure_group`` is one of:

    - ``success``
    - ``impossible start``: even the ideal lander above would hit the ground faster than the
      vertical touchdown limit.
    - ``low and fast start``: the start is below 20 m with more than 3 m/s of descent rate.
      Stopping the descent uses most of the thrust and leaves a few seconds for the lateral
      and attitude corrections.
    - ``other``: any other failed episode.
    """
    limit = manifest["resolved_scenario"]["lander"]["touchdown_limits"]["maximum_vertical_speed_mps"]["value"]
    table = episodes.copy()
    table["start_horizontal_speed_mps"] = numpy.hypot(
        table["start_horizontal_velocity_x_mps"], table["start_horizontal_velocity_y_mps"]
    )
    table["lowest_possible_contact_speed_mps"] = lowest_possible_contact_speed(table, manifest)
    impossible = table["lowest_possible_contact_speed_mps"] > limit
    low_and_fast = (table["start_altitude_m"] < 20.0) & (table["start_vertical_speed_mps"] < -3.0)
    group = numpy.where(impossible, "impossible start", numpy.where(low_and_fast, "low and fast start", "other"))
    table["failure_group"] = numpy.where(table["success"], "success", group)
    return table


def failure_table(episodes: pandas.DataFrame, manifest: dict) -> pandas.DataFrame:
    """Return the count of episodes per failure group and outcome (rows: group, columns: outcome)."""
    table = add_failure_groups(episodes, manifest)
    return pandas.crosstab(table["failure_group"], table["outcome"], margins=True, margins_name="all")
