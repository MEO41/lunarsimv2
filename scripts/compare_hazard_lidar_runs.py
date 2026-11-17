# SPDX-License-Identifier: BSD-3-Clause
"""Compare S6 runs that differ only in the LiDAR (or the map): one table of site-safety numbers.

Example (no Isaac needed; the runs are ``scripts/evaluate_agent.py`` evaluations of the S6
scenarios with the same seed and number of landers)::

    python scripts/compare_hazard_lidar_runs.py \
        runs=results/s6_lidar_specification_seed0,results/s6_lidar_measured_noise_seed0 \
        reference=results/s6_lidar_specification_seed0 \
        nominal=results/s6_nominal_target_seed0 output=results/s6_lidar_comparison

Runs with the same seed start every lander from the same state, so the table also compares them
lander by lander against ``reference`` (how many landers only one of the two put on a safe site).
``nominal`` is a run of the same scenario with site selection switched off
(``agent.decision_altitude_m=-100``): every lander flies to the nominal target, and the extra
propellant of a diverting lander over its own nominal flight is the divert propellant.

Per run: landing success; safe touchdown on the true ground with its 95 % Wilson interval; nominal
target safe; how often the first decision (80 m) found no safe site; redesignations; divert
distance; propellant; divert propellant; decision time; slope and relief under the footpads; and
which criterion each unsafe touchdown fails (LM design surface, [D-6850] Rogers, NASA TN D-6850
p. 3, the limits of ``configs/nav/hazard_map.yaml``). Writes ``comparison.yaml`` and
``comparison.md`` into ``output``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy
import pandas
import yaml

# [D-6850] Rogers, NASA TN D-6850 (1972) p. 3: LM design surface (as in configs/nav/hazard_map.yaml).
MAXIMUM_MEAN_SLOPE_DEG = 6.0
MAXIMUM_EFFECTIVE_SLOPE_DEG = 12.0
MAXIMUM_RELIEF_M = 0.6096
WILSON_Z_95 = 1.959964


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    """Return the 95 % Wilson score interval (low, high) of a proportion, as fractions."""
    if total == 0:
        return (math.nan, math.nan)
    proportion = successes / total
    denominator = 1.0 + WILSON_Z_95**2 / total
    centre = (proportion + WILSON_Z_95**2 / (2 * total)) / denominator
    half_width = WILSON_Z_95 * math.sqrt(proportion * (1 - proportion) / total + WILSON_Z_95**2 / (4 * total**2)) / denominator
    return (centre - half_width, centre + half_width)


def read_episodes(run: Path) -> pandas.DataFrame:
    """Return a run's episode table, one row per lander, sorted by environment index (rows are stored in finish order)."""
    return pandas.read_parquet(run / "episodes.parquet").sort_values(["environment_index", "episode_number"]).reset_index(drop=True)


def mean_and_maximum(values: pandas.Series, digits: int = 3) -> dict:
    """Return the NaN-safe mean and maximum of a column, rounded."""
    values = values.astype(float)
    if not numpy.isfinite(values).any():
        return {"mean": None, "max": None}
    return {"mean": round(float(numpy.nanmean(values)), digits), "max": round(float(numpy.nanmax(values)), digits)}


def unsafe_causes(episodes: pandas.DataFrame) -> dict:
    """Count the unsafe touchdowns by the criterion they fail (one touchdown can fail several)."""
    unsafe = episodes[~episodes["touchdown_site_safe"].astype(bool)]
    return {
        "unsafe_touchdowns": int(len(unsafe)),
        "relief_over_limit": int((unsafe["touchdown_relief_m"] > MAXIMUM_RELIEF_M).sum()),
        "mean_slope_over_limit": int((unsafe["touchdown_mean_slope_deg"] > MAXIMUM_MEAN_SLOPE_DEG).sum()),
        "effective_slope_over_limit": int((unsafe["touchdown_effective_slope_deg"] > MAXIMUM_EFFECTIVE_SLOPE_DEG).sum()),
    }


def summarise_run(run: Path, reference: pandas.DataFrame | None, nominal: pandas.DataFrame | None) -> dict:
    """Return the comparison numbers of one run (see the module docstring)."""
    episodes = read_episodes(run)
    count = int(len(episodes))
    safe = episodes["touchdown_site_safe"].astype(bool)
    low, high = wilson_interval(int(safe.sum()), count)
    summary = {
        "run": run.name,
        "landers": count,
        "landing_success_percent": round(100.0 * float(episodes["success"].mean()), 1),
        "safe_touchdown_percent": round(100.0 * float(safe.mean()), 1),
        "safe_touchdown_95_percent_interval": [round(100.0 * low, 1), round(100.0 * high, 1)],
        "nominal_target_safe_percent": round(100.0 * float(episodes["nominal_site_safe"].astype(bool).mean()), 1),
        "no_safe_site_at_first_decision_percent": round(100.0 * float(episodes["no_safe_site"].astype(bool).mean()), 1) if "no_safe_site" in episodes else None,
        "redesignations": mean_and_maximum(episodes["redesignations"], 2) if "redesignations" in episodes else None,
        "divert_distance_m": mean_and_maximum(episodes["divert_distance_m"], 2),
        "propellant_used_kg": mean_and_maximum(episodes["propellant_used_kg"], 1),
        "propellant_from_first_decision_kg": mean_and_maximum(episodes["divert_propellant_estimate_kg"], 1) if "divert_propellant_estimate_kg" in episodes else None,
        "first_decision_time_s": mean_and_maximum(episodes["decision_time_s"], 2) if "decision_time_s" in episodes else None,
        "landing_duration_s": mean_and_maximum(episodes["touchdown_time_s"], 1),
        "under_footpads": {
            "mean_slope_deg": mean_and_maximum(episodes["touchdown_mean_slope_deg"], 2),
            "effective_slope_deg": mean_and_maximum(episodes["touchdown_effective_slope_deg"], 2),
            "relief_m": mean_and_maximum(episodes["touchdown_relief_m"], 3),
        },
        "unsafe_causes": unsafe_causes(episodes),
    }
    if nominal is not None and len(nominal) == count and (nominal["start_altitude_m"].values == episodes["start_altitude_m"].values).all():
        extra = episodes["propellant_used_kg"].astype(float).values - nominal["propellant_used_kg"].astype(float).values
        summary["divert_propellant_over_nominal_flight_kg"] = {
            "mean": round(float(numpy.mean(extra)), 1), "min": round(float(numpy.min(extra)), 1), "max": round(float(numpy.max(extra)), 1),
        }
    if reference is not None and len(reference) == count and (reference["start_altitude_m"].values == episodes["start_altitude_m"].values).all():
        reference_safe = reference["touchdown_site_safe"].astype(bool).values
        summary["against_reference"] = {
            "safe_here_only": int((safe.values & ~reference_safe).sum()),
            "safe_in_reference_only": int((~safe.values & reference_safe).sum()),
        }
    return summary


def markdown_table(summaries: list[dict]) -> str:
    """Return the headline numbers of every run as a Markdown table."""
    header = (
        "| Run | Landed % | Safe touchdown % (95 % interval) | No safe site at 80 m % | Redesignations | Divert m | "
        "Divert propellant kg | Relief under pads m (mean / max) | Unsafe: relief / slope | Paired vs reference (+ / -) |\n"
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    )
    rows = []
    for summary in summaries:
        interval = summary["safe_touchdown_95_percent_interval"]
        divert_propellant = summary.get("divert_propellant_over_nominal_flight_kg", {}).get("mean", "-")
        paired = summary.get("against_reference")
        causes = summary["unsafe_causes"]
        relief = summary["under_footpads"]["relief_m"]
        rows.append(
            f"| {summary['run']} | {summary['landing_success_percent']} | {summary['safe_touchdown_percent']} ({interval[0]}-{interval[1]}) | "
            f"{summary['no_safe_site_at_first_decision_percent'] if summary['no_safe_site_at_first_decision_percent'] is not None else '-'} | "
            f"{summary['redesignations']['mean'] if summary['redesignations'] else '-'} | {summary['divert_distance_m']['mean']} | {divert_propellant} | "
            f"{relief['mean']} / {relief['max']} | {causes['relief_over_limit']} / {max(causes['mean_slope_over_limit'], causes['effective_slope_over_limit'])} | "
            f"{'+%d / -%d' % (paired['safe_here_only'], paired['safe_in_reference_only']) if paired else '-'} |"
        )
    return header + "\n".join(rows) + "\n"


def main() -> None:
    """Read the runs and write comparison.yaml and comparison.md."""
    options = dict(argument.partition("=")[::2] for argument in sys.argv[1:])
    if "runs" not in options:
        raise SystemExit("give runs=<run directory>,<run directory>,... (optional reference=, nominal=, output=)")
    runs = [Path(path) for path in options["runs"].split(",") if path]
    reference = read_episodes(Path(options["reference"])) if options.get("reference") else None
    nominal = read_episodes(Path(options["nominal"])) if options.get("nominal") else None
    output = Path(options.get("output", "results/s6_lidar_comparison"))
    output.mkdir(parents=True, exist_ok=True)
    summaries = [summarise_run(run, reference, nominal) for run in runs]
    comparison = {
        "reference": Path(options["reference"]).name if options.get("reference") else None,
        "nominal": Path(options["nominal"]).name if options.get("nominal") else None,
        "criteria": {"maximum_mean_slope_deg": MAXIMUM_MEAN_SLOPE_DEG, "maximum_effective_slope_deg": MAXIMUM_EFFECTIVE_SLOPE_DEG, "maximum_relief_m": MAXIMUM_RELIEF_M, "source": "[D-6850] Rogers, NASA TN D-6850 p. 3"},
        "runs": summaries,
    }
    (output / "comparison.yaml").write_text(yaml.safe_dump(comparison, sort_keys=False))
    (output / "comparison.md").write_text(markdown_table(summaries))
    print(markdown_table(summaries))


if __name__ == "__main__":
    main()
