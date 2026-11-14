# SPDX-License-Identifier: BSD-3-Clause
"""Make docs/figures/apollo17_terrain.png: the measured site and the detail patch, as shaded relief.

Usage (needs the height maps, see scripts/download_height_maps.py)::

    python docs/figures/make_apollo17_terrain_figure.py

Left: the measured 2 m height map, 1.2 km around the Apollo 17 lunar module, with the detail
patch outlined. Right: the 160 m detail patch with the drawn craters and rocks, and the square
in which landing targets are drawn. Both panels use the seed-0 terrain of the scenario.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy  # noqa: E402
import yaml  # noqa: E402
from matplotlib.colors import LightSource  # noqa: E402

from lunarsim.core.binding import BindingContext  # noqa: E402
from lunarsim.core.message_bus import MessageBus  # noqa: E402
from lunarsim.core.simulation_clock import SimulationClock  # noqa: E402
from lunarsim.env.scenario_loader import default_config_directory  # noqa: E402
from lunarsim.slots.surface.measured_terrain import MeasuredTerrainSurface  # noqa: E402

OUTPUT = Path(__file__).resolve().parent / "apollo17_terrain.png"


def build_terrain():
    """Build the seed-0 terrain of the Apollo 17 surface config on the CPU; return terrain and parameters."""
    parameters = yaml.safe_load((default_config_directory() / "surface" / "apollo17_taurus_littrow.yaml").read_text())
    parameters.pop("binding")
    context = BindingContext(
        number_of_environments=1, device="cpu", clock=SimulationClock(1, 0.005, 4, device="cpu"), bus=MessageBus()
    )
    surface = MeasuredTerrainSurface(parameters, context)
    return surface.build_terrain(0), parameters


def shaded(heights: numpy.ndarray, cell_size: float) -> numpy.ndarray:
    """Return a grey shaded-relief image (sun from the east, 15 degrees up, as during the Apollo 17 EVAs)."""
    light = LightSource(azdeg=90.0, altdeg=15.0)
    return light.hillshade(heights, vert_exag=1.0, dx=cell_size, dy=cell_size)


def main() -> None:
    """Draw both panels and save the PNG."""
    terrain, parameters = build_terrain()
    outer = terrain.outer.heights.numpy()
    detail = terrain.detail.heights.numpy()
    outer_half = parameters["outer_half_size_m"]
    detail_half = parameters["detail_half_size_m"]
    target_half = parameters["target_half_size_m"]

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10})
    figure, (left, right) = plt.subplots(1, 2, figsize=(10.0, 5.2), constrained_layout=True)
    left.imshow(
        shaded(outer, terrain.outer.cell_size), cmap="gray", origin="lower",
        extent=(-outer_half, outer_half, -outer_half, outer_half), vmin=0.0, vmax=1.0,
    )
    contours = left.contour(
        numpy.linspace(-outer_half, outer_half, outer.shape[1]), numpy.linspace(-outer_half, outer_half, outer.shape[0]),
        outer, levels=numpy.arange(-40, 25, 5), colors="#1f6fb2", linewidths=0.4, alpha=0.7,
    )
    left.clabel(contours, fmt="%d m", fontsize=6)
    left.add_patch(plt.Rectangle((-detail_half, -detail_half), 2 * detail_half, 2 * detail_half, fill=False, color="#d1495b", lw=1.2))
    left.plot([0], [0], marker="+", color="#d1495b", ms=8)
    left.set_title("a   Measured site, 1.2 km (LROC NAC 2 m map)", loc="left")
    left.set_xlabel("east (m)")
    left.set_ylabel("north (m)")

    right.imshow(
        shaded(detail, terrain.detail.cell_size), cmap="gray", origin="lower",
        extent=(-detail_half, detail_half, -detail_half, detail_half), vmin=0.0, vmax=1.0,
    )
    right.add_patch(plt.Rectangle((-target_half, -target_half), 2 * target_half, 2 * target_half, fill=False, color="#d1495b", lw=1.0, ls="--"))
    footpad_circle = plt.Circle((0.0, 0.0), 4.256, fill=False, color="#edae49", lw=1.2)
    right.add_patch(footpad_circle)
    craters, rocks = terrain.craters, terrain.rocks
    right.set_title(f"b   Detail patch, 160 m: {len(craters.diameter)} craters, {len(rocks.diameter)} rocks", loc="left")
    right.set_xlabel("east (m)")
    figure.savefig(OUTPUT, dpi=150)
    print(OUTPUT)


if __name__ == "__main__":
    main()
