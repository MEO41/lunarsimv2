# SPDX-License-Identifier: BSD-3-Clause
"""Render one picture of the lander at rest on the ground.

Examples::

    python scripts/render_lander_image.py --enable_cameras output=docs/figures/lander_at_rest.png
    python scripts/render_lander_image.py --enable_cameras replay=runs/contact_check/contacts.npz clip=1 \
        output=docs/figures/lander_at_rest.png

With ``replay=`` the lander is drawn on the measured Apollo 17 terrain, at rest where it settled in
clip ``clip`` (1-based) of a recording made by ``scripts/contact_check.py``, under a sun 15 degrees
above the eastern horizon; no physics runs. Without it, on flat ground: the lander is dropped from 0.3 m with the engine off, settles for 6 s, and a camera takes one
picture. The picture shows the simple collision shapes that the simulator uses (a cylinder for
the descent stage, a box for the ascent stage, a dish per footpad) and a thin visual cylinder for
each strut. The surface probes and the nozzle have no shape and do not show.
"""

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render the lander at rest.")
AppLauncher.add_app_launcher_args(parser)
arguments, other_arguments = parser.parse_known_args()
arguments.enable_cameras = True
simulation_app = AppLauncher(arguments).app

import numpy  # noqa: E402
import omni.replicator.core as replicator  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402

from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario  # noqa: E402


CAMERA_DIRECTION_FROM_LANDER = numpy.array([-0.63, -0.77])  # from the south-west, sun from the right
CAMERA_DISTANCE_M = 21.0
CAMERA_HEIGHT_ABOVE_LANDER_M = 4.0


def render_on_terrain(replay_path: Path, clip: int, output_path: Path) -> None:
    """Draw the lander at rest on the measured terrain from a recorded clip and save one picture."""
    from lunarsim.tools.replay_rendering import ReplayRenderer
    from pxr import UsdGeom

    replay = numpy.load(replay_path)
    columns = {str(name): index for index, name in enumerate(replay["columns"])}
    renderer = ReplayRenderer(
        simulation_app, str(replay["scenario_name"]), [str(value) for value in replay["scenario_overrides"]],
        replay["frames"].shape[1], 1280, 800,
    )
    index = clip - 1
    last = int(numpy.flatnonzero(replay["valid"][:, index]).max())
    row = replay["frames"][last, index]
    renderer.place(index, row, columns)
    for root, *_ in renderer.arrows[index]:  # a picture of the lander, not of the contact forces
        UsdGeom.Imageable(root).MakeInvisible()
    centre = row[:3].astype(numpy.float64)
    eye = numpy.array([*(centre[:2] + CAMERA_DISTANCE_M * CAMERA_DIRECTION_FROM_LANDER), centre[2] + CAMERA_HEIGHT_ABOVE_LANDER_M])
    renderer.look(eye, centre - numpy.array([0.0, 0.0, 1.2]))
    pixels = renderer.capture(settle_updates=40)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels).save(output_path)
    print(f"[lunarsim] wrote {output_path} (clip {clip}: {replay['clip_titles'][index]})", flush=True)


def main() -> None:
    """Save one camera picture of the lander at rest, on flat ground or (with ``replay=``) on the terrain."""
    options = dict(argument.split("=", 1) for argument in other_arguments if "=" in argument)
    output_path = Path(options.get("output", "docs/figures/lander_at_rest.png"))
    if "replay" in options:
        render_on_terrain(Path(options["replay"]), int(options.get("clip", "1")), output_path)
        return
    scenario = load_scenario(
        "s1_terminal_truth",
        ["num_envs=1", "metrics=none", "actuators.initial_throttle=0.0", "start.altitude_m=[0.3,0.3]",
         "start.v_vertical_mps=[0,0]", "start.v_horizontal_mps=[0,0]", "start.tilt_deg=[0,0]"],
    )
    environment = LanderEnv(build_lander_env_cfg(scenario, device=arguments.device))
    # A second, low light from the side so the shapes show their form.
    side_light = sim_utils.DistantLightCfg(intensity=500.0, color=(1.0, 0.97, 0.9))
    side_light.func("/World/SideLight", side_light, orientation=(0.35, 0.35, 0.15, 0.85))
    environment.reset()
    actions = torch.zeros(1, environment.cfg.action_space, device=environment.device)
    for _ in range(300):
        environment.step(actions)

    camera = replicator.create.camera(position=(22.0, -19.0, 7.0), look_at=(0.0, 0.0, 2.6), focal_length=24.0)
    render_product = replicator.create.render_product(camera, (1280, 800))
    annotator = replicator.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([render_product])
    pixels = numpy.zeros((0,))
    for _ in range(60):
        # The renderer needs some frames before the first picture is ready.
        environment.sim.render()
        simulation_app.update()
        pixels = numpy.asarray(annotator.get_data())
        if pixels.size > 0 and pixels[..., :3].mean() > 1.0:
            break
    if pixels.size == 0:
        replicator.orchestrator.step(rt_subframes=8)
        pixels = numpy.asarray(annotator.get_data())
    for _ in range(20):  # a few more frames so the image has converged
        environment.sim.render()
        simulation_app.update()
    pixels = numpy.asarray(annotator.get_data())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels[..., :3].astype(numpy.uint8)).save(output_path)
    print(f"[lunarsim] wrote {output_path} shape={pixels.shape} mean={pixels[..., :3].mean():.1f}", flush=True)
    environment.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
