# SPDX-License-Identifier: BSD-3-Clause
"""Make a video of chosen landings on the Apollo 17 terrain: some successes and some failures.

Two steps, each in its own process (the second one needs the renderer):

    python scripts/record_landing_videos.py record                     # fly 3072 landings, save poses
    python scripts/record_landing_videos.py render                     # replay 10 of them on camera

Options (``key=value``): ``run_name`` (default ``landing_videos``), ``landings`` (3072),
``seed`` (0), ``successes`` (5), ``failures`` (5), ``width`` (1280), ``height`` (720).

**record** flies ``landings`` episodes of ``apollo17_terminal_truth`` with ZEM/ZEV (no camera,
full speed) and keeps, for every lander, its world pose, strut strokes, speed and height above
ground at 25 frames per second until its first episode ends. It then picks the clips: failures
with ``body_contact`` first, then other failure causes, and successes spread over the ground
roughness under the footprint. Writes ``runs/<run_name>/replay.npz`` and ``episodes.parquet``.

**render** rebuilds the same terrain (same seed) on a plain stage without physics, puts each
chosen lander back on its recorded pose (body and strut strokes) frame by frame, films it with a camera that follows it, writes a text line on every frame
and joins the clips with ffmpeg into ``runs/<run_name>/landings.mp4``. A clip starts 12 s before
the first contact (or at the start) and holds the last frame for 1.5 s.
"""

import argparse
import math
import shutil
import subprocess
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record and render landing videos.")
parser.add_argument("mode", choices=("record", "render"))
AppLauncher.add_app_launcher_args(parser)
arguments, script_arguments = parser.parse_known_args()
# The camera renderer is needed only to render; this Isaac Lab has no command-line flag for it.
arguments.enable_cameras = arguments.mode == "render"
simulation_app = AppLauncher(arguments).app

import numpy  # noqa: E402
import torch  # noqa: E402

from lunarsim.core.binding import SlotName  # noqa: E402
from lunarsim.core.message_types import MessageKey  # noqa: E402
from lunarsim.core.rotation_math import tilt_angle_from_vertical  # noqa: E402
from lunarsim.env.lander_env import LanderEnv  # noqa: E402
from lunarsim.env.lander_env_cfg import build_lander_env_cfg  # noqa: E402
from lunarsim.env.scenario_loader import load_scenario  # noqa: E402
from lunarsim.tools.episode_records import EPISODES_FILE_NAME  # noqa: E402
from lunarsim.tools.episode_runner import build_agent, collect_episodes  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FRAMES_PER_SECOND = 25
SECONDS_BEFORE_CONTACT = 12.0
HOLD_SECONDS = 1.5
# Per-frame values kept for every lander, in this column order.
FRAME_COLUMNS = (
    "position_x", "position_y", "position_z",  # world, m
    "quaternion_w", "quaternion_x", "quaternion_y", "quaternion_z",
    "strut_1", "strut_2", "strut_3", "strut_4",  # m
    "speed_vertical", "speed_horizontal",  # m/s
    "height_above_ground", "tilt_deg", "time",
)


def read_options() -> dict[str, str]:
    """Return the ``key=value`` options with their defaults."""
    options = {"run_name": "landing_videos", "landings": "3072", "seed": "0", "successes": "5", "failures": "5",
               "width": "1280", "height": "720", "preview": ""}
    for argument in script_arguments:
        key, _, value = argument.partition("=")
        if key not in options:
            raise SystemExit(f"unknown option '{key}'")
        options[key] = value
    return options


class PoseRecorder:
    """Keeps the first episode of every environment at 25 frames per second (every second control step)."""

    slot = SlotName.METRICS
    requires = frozenset({MessageKey.TRUTH_STATE, MessageKey.SURFACE_GEOMETRY})
    provides = frozenset()
    reads_truth = True

    def __init__(self, environment) -> None:
        """Prepare the frame list and the per-environment 'still in the first episode' flags."""
        self.environment = environment
        self.plant = environment.bindings[SlotName.PLANT]
        count = environment.num_envs
        self.still_recording = torch.ones(count, dtype=torch.bool, device=environment.device)
        self.previous_time = torch.full((count,), -1.0, device=environment.device)
        self.frames: list[numpy.ndarray] = []
        self.valid: list[numpy.ndarray] = []
        self.site_origin = None

    def record(self, step_index: int, command) -> None:
        """Store one frame of every environment that is still in its first episode."""
        truth = self.environment.bus.read(MessageKey.TRUTH_STATE, self)
        geometry = self.environment.bus.read(MessageKey.SURFACE_GEOMETRY, self)
        if self.site_origin is None:
            self.site_origin = geometry.site_origin.clone()
        # The episode time jumps back when an environment resets: its first episode is over.
        self.still_recording &= truth.time > self.previous_time
        self.previous_time = truth.time.clone()
        if step_index % 2 == 1:
            return
        world = truth.position + self.site_origin
        frame = torch.cat(
            (
                world,
                truth.orientation_quaternion,
                self.plant.strut_compression(),
                truth.velocity[:, 2:3],
                truth.velocity[:, :2].norm(dim=1, keepdim=True),
                truth.footpad_height_above_ground.unsqueeze(1),
                torch.rad2deg(tilt_angle_from_vertical(truth.orientation_quaternion)).unsqueeze(1),
                truth.time.unsqueeze(1),
            ),
            dim=1,
        )
        self.frames.append(frame.cpu().numpy().astype(numpy.float32))
        self.valid.append(self.still_recording.cpu().numpy())


def choose_clips(episodes, successes: int, failures: int, random: numpy.random.Generator) -> list[int]:
    """Return environment indices: failures (body_contact first) and successes spread over ground relief."""
    failed = episodes[~episodes["success"]]
    order = ["body_contact"] + [name for name in failed["outcome"].value_counts().index if name != "body_contact"]
    chosen_failures: list[int] = []
    for outcome in order:
        candidates = failed[failed["outcome"] == outcome]["environment_index"].tolist()
        random.shuffle(candidates)
        chosen_failures += candidates[: failures - len(chosen_failures)]
        if len(chosen_failures) >= failures:
            break
    succeeded = episodes[episodes["success"]].sort_values("touchdown_ground_relief_m")
    positions = numpy.linspace(0, len(succeeded) - 1, successes).round().astype(int)
    chosen_successes = succeeded.iloc[positions]["environment_index"].tolist()
    return chosen_successes + chosen_failures


def record(options: dict[str, str]) -> None:
    """Fly the landings, keep all poses, pick the clips and save them."""
    count = int(options["landings"])
    scenario = load_scenario("apollo17_terminal_truth", [f"num_envs={count}", f"seed={options['seed']}"])
    environment = LanderEnv(build_lander_env_cfg(scenario, device=arguments.device))
    agent = build_agent(environment)
    recorder = PoseRecorder(environment)
    episodes = collect_episodes(
        environment, agent, episodes_per_environment=1,
        maximum_steps=int(1.2 * environment.max_episode_length) + 10, after_step=recorder.record,
    )
    frames = numpy.stack(recorder.frames)  # (frames, envs, columns)
    valid = numpy.stack(recorder.valid)  # (frames, envs)
    clips = choose_clips(episodes, int(options["successes"]), int(options["failures"]), numpy.random.default_rng(int(options["seed"])))
    run_directory = REPOSITORY_ROOT / "runs" / options["run_name"]
    run_directory.mkdir(parents=True, exist_ok=True)
    episodes.to_parquet(run_directory / EPISODES_FILE_NAME, index=False)
    table = episodes.set_index("environment_index")
    numpy.savez_compressed(
        run_directory / "replay.npz",
        frames=frames[:, clips],
        valid=valid[:, clips],
        environment_index=numpy.array(clips),
        site_origin=recorder.site_origin.cpu().numpy()[clips],
        outcome=numpy.array([table.loc[index, "outcome"] for index in clips]),
        contact_time=numpy.array([table.loc[index, "touchdown_time_s"] for index in clips]),
        relief=numpy.array([table.loc[index, "touchdown_ground_relief_m"] for index in clips]),
        slope=numpy.array([table.loc[index, "touchdown_ground_slope_deg"] for index in clips]),
        body_force=numpy.array([table.loc[index, "peak_body_contact_force_n"] for index in clips]),
        nozzle_clearance=numpy.array([table.loc[index, "minimum_nozzle_clearance_m"] for index in clips]),
        seed=int(options["seed"]),
        columns=numpy.array(FRAME_COLUMNS),
    )
    print(f"[videos] {len(episodes)} landings, success rate {episodes['success'].mean():.4f}")
    print(episodes["outcome"].value_counts().to_string())
    print(f"[videos] clips: {[(index, table.loc[index, 'outcome']) for index in clips]}")
    print(f"[videos] wrote {run_directory / 'replay.npz'}")
    environment.close()


def look_at_matrix(eye, target):
    """Return the USD camera transform that looks from ``eye`` at ``target`` with z up."""
    from pxr import Gf

    return Gf.Matrix4d().SetLookAt(Gf.Vec3d(*map(float, eye)), Gf.Vec3d(*map(float, target)), Gf.Vec3d(0, 0, 1)).GetInverse()


def caption_lines(clip_number: int, total: int, outcome: str, values: dict, extra: dict) -> list[str]:
    """Return the text lines written on a frame."""
    title = "SUCCESS" if outcome == "success" else f"FAILURE: {outcome}"
    lines = [
        f"{clip_number}/{total}  {title}",
        f"t = {values['time']:5.1f} s   height above ground {values['height_above_ground']:6.2f} m   "
        f"vertical {values['speed_vertical']:+5.2f} m/s   horizontal {values['speed_horizontal']:4.2f} m/s   tilt {values['tilt_deg']:4.1f} deg",
        f"ground under footprint at contact: slope {extra['slope']:.1f} deg, relief {extra['relief']:.2f} m   "
        f"peak body force {extra['body_force']:.0f} N   lowest nozzle clearance {extra['nozzle_clearance']:.2f} m",
    ]
    return lines


def build_render_stage(seed: int, clip_count: int):
    """Make a plain stage (no physics) with the scenario's terrain, a low sun and one lander per clip.

    Returns (stage, lander prim paths, base leg translations). Only the renderer reads this
    stage, so the landers are moved by setting their transforms directly.
    """
    import omni.usd
    from pxr import Gf, UsdGeom, UsdLux, Vt

    from lunarsim.assets.lander_builder import write_lander_usd
    from lunarsim.assets.lander_preset import lander_preset_from_dictionary
    from lunarsim.core.binding import BindingContext
    from lunarsim.core.message_bus import MessageBus
    from lunarsim.core.simulation_clock import SimulationClock
    from lunarsim.slots.surface.measured_terrain import MeasuredTerrainSurface

    scenario = load_scenario("apollo17_terminal_truth", [f"seed={seed}"])
    preset = lander_preset_from_dictionary(scenario.lander)
    context = omni.usd.get_context()
    context.new_stage()
    stage = context.get_stage()
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    UsdGeom.Xform.Define(stage, "/World")
    surface = MeasuredTerrainSurface(
        scenario.slots["surface"].parameters,
        BindingContext(number_of_environments=1, device="cpu", clock=SimulationClock(1, 0.005, 4, device="cpu"), bus=MessageBus()),
    )
    terrain = surface.build_terrain(scenario.root_seed)
    UsdGeom.Xform.Define(stage, "/World/ground")
    for name, (vertices, triangles) in terrain.meshes().items():
        mesh = UsdGeom.Mesh.Define(stage, f"/World/ground/{name}")
        mesh.CreatePointsAttr(Vt.Vec3fArray.FromNumpy(vertices.astype(numpy.float32)))
        mesh.CreateFaceVertexCountsAttr(Vt.IntArray.FromNumpy(numpy.full(len(triangles), 3, dtype=numpy.int32)))
        mesh.CreateFaceVertexIndicesAttr(Vt.IntArray.FromNumpy(triangles.astype(numpy.int32).ravel()))
        mesh.CreateSubdivisionSchemeAttr(UsdGeom.Tokens.none)
        mesh.CreateDisplayColorAttr([Gf.Vec3f(0.30, 0.29, 0.27)])  # regolith grey
    # Sun 15 degrees above the eastern horizon (Apollo 17 surface operations), and a faint sky
    # fill so that the shadows are not fully black. A distant light shines along its -z axis.
    sun = UsdLux.DistantLight.Define(stage, "/World/Sun")
    sun.CreateIntensityAttr(2500.0)
    sun.CreateAngleAttr(0.53)
    sun.CreateColorAttr(Gf.Vec3f(1.0, 0.98, 0.95))
    UsdGeom.Xformable(sun).AddRotateYOp().Set(90.0 - 15.0)
    fill = UsdLux.DomeLight.Define(stage, "/World/Fill")
    fill.CreateIntensityAttr(25.0)
    import tempfile

    lander_path = write_lander_usd(preset, Path(tempfile.gettempdir()) / "lunarsim_assets")
    lander_paths, leg_bases = [], []
    for clip in range(clip_count):
        path = f"/World/Lander_{clip}"
        prim = stage.DefinePrim(path, "Xform")
        prim.GetReferences().AddReference(str(lander_path))
        xformable = UsdGeom.Xformable(prim)
        xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(Gf.Vec3d(0.0, 0.0, -1000.0))
        xformable.AddOrientOp(UsdGeom.XformOp.PrecisionDouble)
        lander_paths.append(path)
    for index in range(preset.number_of_legs):
        leg_bases.append(preset.leg_positions[index])
    return stage, lander_paths, leg_bases, preset


def render(options: dict[str, str]) -> None:
    """Replay the chosen landings on camera and write the MP4."""
    import omni.replicator.core as replicator
    from PIL import Image, ImageDraw, ImageFont
    from pxr import Gf, UsdGeom

    from lunarsim.assets.lander_builder import leg_link_name

    run_directory = REPOSITORY_ROOT / "runs" / options["run_name"]
    replay = numpy.load(run_directory / "replay.npz")
    frames, valid = replay["frames"], replay["valid"]
    columns = {name: index for index, name in enumerate(replay["columns"])}
    clip_count = frames.shape[1]
    stage, lander_paths, leg_bases, preset = build_render_stage(int(replay["seed"]), clip_count)
    landers = [stage.GetPrimAtPath(path) for path in lander_paths]
    leg_translate = [
        [UsdGeom.Xformable(stage.GetPrimAtPath(f"{path}/{leg_link_name(index)}")).GetOrderedXformOps()[0] for index in range(len(leg_bases))]
        for path in lander_paths
    ]
    camera = UsdGeom.Camera.Define(stage, "/World/VideoCamera")
    camera.CreateFocalLengthAttr(28.0)
    camera.CreateHorizontalApertureAttr(36.0)
    camera.CreateVerticalApertureAttr(36.0 * int(options["height"]) / int(options["width"]))
    camera.CreateClippingRangeAttr(Gf.Vec2f(0.1, 5000.0))
    transform = UsdGeom.Xformable(camera).AddTransformOp()
    render_product = replicator.create.render_product("/World/VideoCamera", (int(options["width"]), int(options["height"])))
    annotator = replicator.AnnotatorRegistry.get_annotator("rgb")
    annotator.attach([render_product])
    font_path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    font = ImageFont.truetype(str(font_path), 18) if font_path.is_file() else ImageFont.load_default()
    frame_directory = run_directory / "frames"
    if frame_directory.exists():
        shutil.rmtree(frame_directory)
    frame_directory.mkdir(parents=True)
    for _ in range(30):  # let the stage load and the renderer start
        simulation_app.update()

    def show(pose: numpy.ndarray, active: int) -> None:
        """Put lander ``active`` on its recorded pose (body and strut strokes) and hide the others."""
        for clip, prim in enumerate(landers):
            translate_op, orient_op = UsdGeom.Xformable(prim).GetOrderedXformOps()
            if clip != active:
                translate_op.Set(Gf.Vec3d(0.0, 0.0, -1000.0))
                continue
            translate_op.Set(Gf.Vec3d(*map(float, pose[0:3])))
            orient_op.Set(Gf.Quatd(float(pose[3]), Gf.Vec3d(float(pose[4]), float(pose[5]), float(pose[6]))))
            for index, (x, y, z) in enumerate(leg_bases):
                leg_translate[clip][index].Set(Gf.Vec3d(x, y, z + float(pose[columns[f"strut_{index + 1}"]])))

    def capture() -> numpy.ndarray:
        """Render the current stage and return the camera image (the first ones may need more updates)."""
        for _ in range(60):
            simulation_app.update()
            pixels = numpy.asarray(annotator.get_data())
            if pixels.size > 0:
                return pixels
        raise RuntimeError("the camera gave no image after 60 updates")

    written = 0
    for clip in range(clip_count):
        outcome = str(replay["outcome"][clip])
        extra = {name: float(replay[name][clip]) for name in ("slope", "relief", "body_force", "nozzle_clearance")}
        last = int(numpy.flatnonzero(valid[:, clip]).max())
        contact_time = float(replay["contact_time"][clip])
        first = max(0, int((contact_time - SECONDS_BEFORE_CONTACT) * FRAMES_PER_SECOND)) if contact_time > 0 else 0
        indices = list(range(first, last + 1)) + [last] * int(HOLD_SECONDS * FRAMES_PER_SECOND)
        previous_target = None
        warm_up = 8
        for position, index in enumerate([indices[0]] * warm_up + indices):
            pose = frames[index, clip]
            target = pose[:3].astype(numpy.float64) - numpy.array([0.0, 0.0, 1.0])
            if previous_target is not None:
                target = 0.85 * previous_target + 0.15 * target  # smooth the camera
            previous_target = target
            eye = target + numpy.array([-14.0, -17.0, 5.0])
            transform.Set(look_at_matrix(eye, target))
            show(pose, clip)
            pixels = capture()
            if position < warm_up:
                continue  # let the renderer converge on the new view
            image = Image.fromarray(pixels[..., :3].astype(numpy.uint8))
            draw = ImageDraw.Draw(image)
            values = {name: float(pose[columns[name]]) for name in columns}
            lines = caption_lines(clip + 1, clip_count, outcome, values, extra)
            draw.rectangle((0, 0, image.width, 30 + 24 * len(lines)), fill=(0, 0, 0))
            for line_number, line in enumerate(lines):
                color = (255, 255, 255) if line_number else ((120, 220, 120) if outcome == "success" else (255, 110, 110))
                draw.text((14, 10 + 24 * line_number), line, fill=color, font=font)
            image.save(frame_directory / f"frame_{written:06d}.png")
            written += 1
            if options.get("preview") and written >= int(options["preview"]):
                print(f"[videos] preview: {written} frames in {frame_directory}")
                return
        print(f"[videos] clip {clip + 1}/{clip_count} ({outcome}) done, {written} frames so far", flush=True)

    video_path = run_directory / "landings.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FRAMES_PER_SECOND), "-i", str(frame_directory / "frame_%06d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(video_path)],
        check=True,
    )
    shutil.rmtree(frame_directory)
    print(f"[videos] wrote {video_path} ({written} frames, {written / FRAMES_PER_SECOND:.0f} s)")


if __name__ == "__main__":
    selected = read_options()
    if arguments.mode == "record":
        record(selected)
    else:
        render(selected)
    simulation_app.close()
