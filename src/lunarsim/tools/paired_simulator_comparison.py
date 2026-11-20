# SPDX-License-Identifier: BSD-3-Clause
"""Compare two evaluation runs that flew the same start states in two simulators (paired test T3).

Typical use: the same agent and the same ``seed`` evaluated once with ``plant=analytic_rigid_body``
and once with ``plant=physx_site``. Each environment's first episode is one paired observation.
The comparison gives

- the success rate of each run with its 95 % Wilson interval;
- the agreement table (both succeed, only the first, only the second, both fail) and the exact
  McNemar test of the two discordant counts (do the simulators judge the same starts differently?);
- the outcome cross table (outcome in the first run against the outcome in the second);
- per-start differences of the touchdown values (vertical and horizontal speed, tilt, tilt rate,
  touchdown point, time, propellant), over the starts that touched down in both runs.

No simulator is needed; it reads ``episodes.parquet`` files.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy
import pandas

from lunarsim.tools.agent_comparison import first_episodes

PAIRING_COLUMNS = (
    "start_altitude_m",
    "start_vertical_speed_mps",
    "start_horizontal_velocity_x_mps",
    "start_horizontal_velocity_y_mps",
    "start_tilt_deg",
)
TOUCHDOWN_DIFFERENCE_COLUMNS = (
    "touchdown_vertical_speed_mps",
    "touchdown_horizontal_speed_mps",
    "touchdown_tilt_deg",
    "touchdown_tilt_rate_deg_per_s",
    "touchdown_time_s",
    "propellant_used_kg",
)


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Return the Wilson score interval (low, high) of a success proportion; 95 % by default."""
    if trials == 0:
        return float("nan"), float("nan")
    proportion = successes / trials
    denominator = 1.0 + z * z / trials
    centre = (proportion + z * z / (2.0 * trials)) / denominator
    half_width = z * math.sqrt(proportion * (1.0 - proportion) / trials + z * z / (4.0 * trials * trials)) / denominator
    return max(0.0, centre - half_width), min(1.0, centre + half_width)


def mcnemar_exact_p_value(only_first: int, only_second: int) -> float:
    """Return the two-sided exact McNemar p-value: a binomial test of the discordant pairs at one half."""
    discordant = only_first + only_second
    if discordant == 0:
        return 1.0
    smaller = min(only_first, only_second)
    tail = sum(math.comb(discordant, k) for k in range(smaller + 1))
    return min(1.0, 2.0 * tail / 2**discordant)


def paired_first_episodes(first_run: Path | str, second_run: Path | str, tolerance: float = 1e-3) -> pandas.DataFrame:
    """Return one row per environment with the first episode of each run side by side (suffixes _first, _second).

    Raises ValueError if the runs have different environment counts or did not start from the
    same states (the tolerance is in the units of the start columns).
    """
    first = first_episodes(first_run)
    second = first_episodes(second_run)
    if len(first) != len(second):
        raise ValueError(f"The runs have {len(first)} and {len(second)} first episodes; they must fly the same starts")
    merged = first.merge(second, on="environment_index", suffixes=("_first", "_second"))
    for column in PAIRING_COLUMNS:
        if f"{column}_first" not in merged:
            continue
        gap = (merged[f"{column}_first"] - merged[f"{column}_second"]).abs().max()
        if gap > tolerance:
            raise ValueError(f"The runs started from different states: {column} differs by up to {gap:.4g}")
    return merged


def compare_paired_runs(first_run: Path | str, second_run: Path | str, labels: tuple[str, str] = ("first", "second")) -> dict:
    """Return the paired comparison as a dictionary of plain numbers and tables (see the module docstring)."""
    merged = paired_first_episodes(first_run, second_run)
    first_success = merged["success_first"].astype(bool)
    second_success = merged["success_second"].astype(bool)
    trials = len(merged)
    summary = {"pairs": trials, "labels": list(labels)}
    for label, success in zip(labels, (first_success, second_success), strict=True):
        count = int(success.sum())
        low, high = wilson_interval(count, trials)
        summary[f"{label}_success_rate"] = count / trials
        summary[f"{label}_success_wilson_95"] = (low, high)
    both = int((first_success & second_success).sum())
    only_first = int((first_success & ~second_success).sum())
    only_second = int((~first_success & second_success).sum())
    neither = int((~first_success & ~second_success).sum())
    summary["agreement"] = {
        "both_succeed": both,
        f"only_{labels[0]}_succeeds": only_first,
        f"only_{labels[1]}_succeeds": only_second,
        "both_fail": neither,
    }
    summary["outcome_agreement_rate"] = float((merged["outcome_first"] == merged["outcome_second"]).mean())
    summary["mcnemar_exact_p_value"] = mcnemar_exact_p_value(only_first, only_second)
    summary["outcome_cross_table"] = pandas.crosstab(
        merged["outcome_first"].rename(labels[0]), merged["outcome_second"].rename(labels[1])
    )

    touched = merged["had_contact_first"].astype(bool) & merged["had_contact_second"].astype(bool)
    rows = []
    for column in TOUCHDOWN_DIFFERENCE_COLUMNS:
        if f"{column}_first" not in merged:
            continue
        difference = (merged.loc[touched, f"{column}_second"] - merged.loc[touched, f"{column}_first"]).to_numpy()
        rows.append(
            {
                "value": column,
                f"{labels[0]} median": float(merged.loc[touched, f"{column}_first"].median()),
                f"{labels[1]} median": float(merged.loc[touched, f"{column}_second"].median()),
                "mean difference": float(numpy.mean(difference)) if len(difference) else float("nan"),
                "RMS difference": float(numpy.sqrt(numpy.mean(difference**2))) if len(difference) else float("nan"),
                "largest |difference|": float(numpy.max(numpy.abs(difference))) if len(difference) else float("nan"),
            }
        )
    touchdown_shift = numpy.hypot(
        merged.loc[touched, "touchdown_position_x_m_second"] - merged.loc[touched, "touchdown_position_x_m_first"],
        merged.loc[touched, "touchdown_position_y_m_second"] - merged.loc[touched, "touchdown_position_y_m_first"],
    ).to_numpy()
    rows.append(
        {
            "value": "touchdown point distance (m)",
            f"{labels[0]} median": float("nan"),
            f"{labels[1]} median": float("nan"),
            "mean difference": float(numpy.mean(touchdown_shift)) if len(touchdown_shift) else float("nan"),
            "RMS difference": float(numpy.sqrt(numpy.mean(touchdown_shift**2))) if len(touchdown_shift) else float("nan"),
            "largest |difference|": float(numpy.max(touchdown_shift)) if len(touchdown_shift) else float("nan"),
        }
    )
    summary["pairs_with_touchdown_in_both"] = int(touched.sum())
    summary["touchdown_differences"] = pandas.DataFrame(rows)
    return summary


def format_paired_comparison(summary: dict) -> str:
    """Return the comparison as plain text for the terminal."""
    first, second = summary["labels"]
    lines = [f"pairs: {summary['pairs']}"]
    for label in (first, second):
        low, high = summary[f"{label}_success_wilson_95"]
        lines.append(f"{label} success: {100 * summary[f'{label}_success_rate']:.2f} %  (95 % Wilson {100 * low:.2f} to {100 * high:.2f} %)")
    lines.append("agreement: " + ", ".join(f"{name} {count}" for name, count in summary["agreement"].items()))
    lines.append(f"same outcome code: {100 * summary['outcome_agreement_rate']:.2f} %")
    lines.append(f"exact McNemar p-value: {summary['mcnemar_exact_p_value']:.4g}")
    lines.append("")
    lines.append(summary["outcome_cross_table"].to_string())
    lines.append("")
    lines.append(f"touchdown differences ({second} minus {first}), {summary['pairs_with_touchdown_in_both']} pairs that touched down in both:")
    lines.append(summary["touchdown_differences"].round(4).to_string(index=False))
    return "\n".join(lines)
