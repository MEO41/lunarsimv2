# SPDX-License-Identifier: BSD-3-Clause
"""Draw the overview figure of the slot loop as built up to milestone 3 (sensors, faults, filters, learned agents).

Run from the repository root::

    python docs/figures/make_slot_loop_figure.py <path to pipefig scripts directory>

The drawing library ``pipefig`` is not part of this repository (it comes with the
pipeline-figure skill). The lander picture is made by ``scripts/render_lander_image.py``.
Slot names, binding names and message names are the ones in the code and scenario files.
"""

import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
import pipefig as pf  # noqa: E402

FIGURE_DIRECTORY = Path(__file__).resolve().parent
diagram = pf.Diagram(width=183, height_mm=98, venue="generic")

# ---- 1 scenario -------------------------------------------------------------------------
diagram.panel("scenario", 2, 2, 38, 64, title="1  Scenario", tint="grey")
diagram.box("preset", 5, 11, 32, 12, "Lander preset", role="input", sub="Apollo LM, cited sources")
diagram.box("file", 5, 30, 32, 12, "Scenario file", role="input", sub="s3_faults.yaml")
diagram.box("check", 5, 49, 32, 12, "Registry check", role="process", sub="requires, provides, truth")
diagram.arrow("preset", "file", out="bottom", inn="top")
diagram.arrow("file", "check", out="bottom", inn="top")

# ---- 2 slot loop ------------------------------------------------------------------------
diagram.panel("loop", 44, 2, 96, 64, title="2  LanderEnv: slot loop in the DirectRLEnv hooks, N envs on the GPU", tint="blue")
columns = (47, 79, 111)
rows = (12, 30, 49)
width, height = 26, 12
diagram.box("sensors", columns[0], rows[0], width, height, "Sensors", role="model", sub="terminal_suite")
diagram.box("faults", columns[1], rows[0], width, height, "Faults", role="model", sub="schedule, delay")
diagram.box("nav", columns[2], rows[0], width, height, "Navigation", role="process", sub="ekf, ekf_fdi")
diagram.box("plant", columns[0], rows[1], width, height, "Plant", role="model", sub="physx_site")
diagram.box("surface", columns[1], rows[1], width, height, "Surface", role="model", sub="flat, Apollo 17")
diagram.box("actuators", columns[2], rows[1], width, height, "Actuators", role="process", sub="engine_rcs_pulse")
diagram.box("start", columns[0], rows[2], width, height, "Start", role="model", sub="box + nav error")
diagram.box("metrics", columns[2], rows[2], width, height, "Metrics", role="eval", sub="fault_response")
diagram.image("render", str(FIGURE_DIRECTORY / "lander_at_rest.png"), columns[1], 47.2, width, crop=(250, 170, 1050, 662))

diagram.arrow("check", (44.0, 55.0), out="right")
diagram.arrow("start", "plant", out="top", inn="bottom", fo=0.3, fi=0.3)
diagram.arrow("plant", "sensors", out="top", inn="bottom", label="TruthState", label_side="right")
diagram.arrow("sensors", "faults", out="right", inn="left")
diagram.arrow("faults", "nav", out="right", inn="left")
diagram.arrow("actuators", "surface", out="left", inn="right")
diagram.arrow("surface", "plant", out="left", inn="right")
diagram.arrow("plant", "metrics", out="bottom", inn="top", fo=0.8, via=[(67.8, 44.5), (124.0, 44.5)])

# ---- 3 agent ----------------------------------------------------------------------------
diagram.panel("agent_panel", 144, 2, 37, 64, title="3  Agent, outside the env", tint="red")
diagram.box("agent", 147, 21, 31, 14, "Agent", role="novel", bold=True, sub="zem_zev, ppo_*")
diagram.box("training", 147, 48, 31, 13, "PPO training", role="learned", sub="mlp, gru, attention")
diagram.arrow("training", "agent", out="top", inn="bottom", fo=0.82, fi=0.82, flow="train", ls="--")
diagram.arrow("nav", "agent", out="right", inn="top", flow="forward", label="Observation", label_side="above", label_t=0.72, via=[(162.5, 18.0)])
diagram.arrow("agent", "actuators", out="bottom", inn="right", flow="feedback", label="Command", label_side="below", label_t=0.3, via=[(162.5, 36.0)], fo=0.5)

# ---- 4 run files ------------------------------------------------------------------------
diagram.panel("run", 66, 70, 115, 26, title="4  Run files", tint="green")
diagram.box("manifest", 69, 79, 34, 13, "manifest.yaml", role="eval", sub="config, git hash, seeds")
diagram.box("episodes", 107, 79, 34, 13, "episodes.parquet", role="eval", sub="one row per episode")
diagram.box("figures", 145, 79, 33, 13, "lunarsim CLI", role="eval", sub="figures, compare")
diagram.arrow("metrics", "episodes", out="bottom", inn="top")
diagram.arrow("episodes", "figures", out="right", inn="left")

diagram.legend(
    [
        ("scenario input", dict(patch="input")),
        ("truth-side slot", dict(patch="model")),
        ("flight-side slot", dict(patch="process")),
        ("studied slot", dict(patch="novel")),
        ("trained network", dict(patch="learned")),
        ("metrics and records", dict(patch="eval")),
        ("messages on the bus", dict(flow="data")),
        ("Observation", dict(flow="forward")),
        ("Command", dict(flow="feedback")),
        ("training (rsl_rl)", dict(flow="train", ls="--")),
    ],
    2,
    70,
    w=62,
    ncol=2,
    colw=31,
)
diagram.save(str(FIGURE_DIRECTORY / "slot_loop_overview"))
