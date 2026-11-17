# SPDX-License-Identifier: BSD-3-Clause
"""Compare the L1 plume with the L2 grains on the same landings (milestone 5 gate: "the L1 vs L2 check is written up").

Example (no Isaac needed; the run is ``python scripts/evaluate_agent.py scenario=s8_plume_check_l2
num_envs=64 seed=0 run_name=s8_plume_check_l2_seed0``)::

    python scripts/compare_plume_levels.py run=runs/s8_plume_check_l2_seed0

The dust metric records both levels on each landing (grains in shadow). Writes ``comparison.yaml``
in the run directory: per level the eroded mass, erosion start height, time with the sensors
blinded and peak dust; for the grains also the peak optical depth and the peak lofted number
density; and the published reference values:

- Apollo 11 eroded volume 36-57 L (Mason 1970, as cited in Metzger, Lane and Immer 2008, p. 6),
  as mass at the surface bulk density 1.30 g/cm^3 (Lunar Sourcebook p. 494, as cited in NTRS
  20220014634); only landings that start above 100 ft are compared with it;
- lofted particle density 1e8 to 1e13 per m^3 from the Apollo 11 and 16 videos (Immer, Metzger,
  Hintze et al. 2008, arXiv 2104.07669);
- dust first noticed "something less than 100 feet" up (Armstrong, ALSJ, Apollo 11).
"""

import sys
from pathlib import Path

import numpy
import pandas
import yaml

SURFACE_BULK_DENSITY_KG_PER_M3 = 1300.0
APOLLO_11_ERODED_LITRES = (36.0, 57.0)
APOLLO_LOFTED_DENSITY_PER_M3 = (1e8, 1e13)
ONSET_HEIGHT_M = 30.48


def spread(values: pandas.Series) -> dict:
    """Return the mean, the smallest and the largest value (NaN-safe, rounded)."""
    values = values.astype(float)
    return {"mean": round(float(numpy.nanmean(values)), 4), "min": round(float(numpy.nanmin(values)), 4), "max": round(float(numpy.nanmax(values)), 4)}


def main() -> None:
    """Read the run's episodes and write comparison.yaml."""
    options = dict(argument.partition("=")[::2] for argument in sys.argv[1:])
    if "run" not in options:
        raise SystemExit("give run=<run directory>")
    run = Path(options["run"])
    episodes = pandas.read_parquet(run / "episodes.parquet")
    if "grain_eroded_mass_kg" not in episodes:
        raise SystemExit("this run has no grain columns: use scenario s8_plume_check_l2")
    high = episodes[episodes["start_altitude_m"] >= ONSET_HEIGHT_M]
    eroded_mass = tuple(SURFACE_BULK_DENSITY_KG_PER_M3 * litres / 1000.0 for litres in APOLLO_11_ERODED_LITRES)
    comparison = {
        "run": run.name,
        "landings": int(len(episodes)),
        "landing_success_fraction": round(float(episodes["success"].mean()), 4),
        "landings_starting_above_100_ft": int(len(high)),
        "L1": {
            "eroded_mass_kg": spread(episodes["ejecta_mass_kg"]),
            "eroded_mass_kg_from_above_100_ft": spread(high["ejecta_mass_kg"]) if len(high) else None,
            "erosion_start_nozzle_height_m": spread(episodes["erosion_start_nozzle_height_m"]),
            "blinded_time_s": spread(episodes["dust_blinded_time_s"]),
            "peak_dust_density": spread(episodes["peak_dust_density"]),
        },
        "L2_grains": {
            "eroded_mass_kg": spread(episodes["grain_eroded_mass_kg"]),
            "eroded_mass_kg_from_above_100_ft": spread(high["grain_eroded_mass_kg"]) if len(high) else None,
            "erosion_start_nozzle_height_m": spread(episodes["grain_erosion_start_nozzle_height_m"]),
            "blinded_time_s": spread(episodes["grain_dust_blinded_time_s"]),
            "peak_dust_density": spread(episodes["grain_peak_dust_density"]),
            "peak_optical_depth": spread(episodes["grain_peak_optical_depth"]),
            "peak_number_density_per_m3": spread(episodes["grain_peak_number_density_per_m3"]),
        },
        "references": {
            "apollo_11_eroded_mass_kg": [round(value, 1) for value in eroded_mass],
            "apollo_lofted_number_density_per_m3": list(APOLLO_LOFTED_DENSITY_PER_M3),
            "apollo_11_dust_noticed_below_m": ONSET_HEIGHT_M,
        },
    }
    with open(run / "comparison.yaml", "w") as file:
        yaml.safe_dump(comparison, file, sort_keys=False)
    print(yaml.safe_dump(comparison, sort_keys=False))


if __name__ == "__main__":
    main()
