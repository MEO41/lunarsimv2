# SPDX-License-Identifier: BSD-3-Clause
"""Render a recorded replay (``.npz`` from ``lunarsim.tools.replay_recording``) as a video or as stills.

Usage::

    python scripts/render_replay.py replay=runs/drop_test_5m/flips.npz mode=video
    python scripts/render_replay.py replay=runs/contact_check/contacts.npz mode=stills

**video** writes ``<replay name>.mp4`` next to the replay: every clip from its start time to the
end of its first episode, plus 1.5 s of the last frame, followed by a camera that trails the
lander. **stills** writes one PNG per clip from the last recorded frame (the lander at rest): a
wide view on top and a close-up of each footpad below. When contacts were recorded, a green
arrow stands beside every footpad that carries more than 100 N, along the force from that leg's
contact sensor (1 m per 3 kN), and the captions give each pad's force and its gap to the ground.

When the replay holds plume grains (``ReplayRecorder(grain_landers=...)``), the video also draws
the grain parcels and puts a true-scale side view with both plume levels' readings to the right
of the 3D view (``lunarsim.tools.plume_grain_video``); the frame is then 1920 wide. Two styles:

- ``dust_style=streaks`` (default): as a camera would see the model's cloud. Each parcel is its
  path during the exposure (``exposure_s``, default 1/60 s, a model choice), in the sunlit
  regolith grey, with the opacity of its grains' extinction cross-section (Beer-Lambert per
  pixel), hidden behind the terrain and the lander (depth image). ``visibility_gain`` multiplies
  the optical depth; anything but 1 is written on every frame;
- ``dust_style=dots``: each parcel a dot coloured by grain size, ``grain_dot_angle_rad`` wide as
  seen from the camera (far larger than the grains).

``view=chase`` (default) follows the lander from outside; ``view=window`` is a camera fixed to the
lander that looks forward and down along the grain model's sensor line of sight, like the
Apollo crews' window films.

Options: ``replay``, ``mode`` (video or stills), ``width`` (1280), ``height`` (720),
``camera_distance_m`` (20), ``camera_height_m`` (5, above the lander), ``dust_style``, ``view``,
``exposure_s``, ``visibility_gain`` (1), ``grain_dot_angle_rad`` (0.005), ``output`` (default
``<replay name>.mp4``).
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render a recorded replay.")
AppLauncher.add_app_launcher_args(parser)
arguments, script_arguments = parser.parse_known_args()
arguments.enable_cameras = True  # this Isaac Lab has no command-line flag for it
simulation_app = AppLauncher(arguments).app

import numpy  # noqa: E402
from PIL import Image  # noqa: E402

from lunarsim.tools.replay_rendering import ARROW_SHOWN_ABOVE_NEWTON, ReplayRenderer, draw_caption  # noqa: E402

LEG_NAMES = ("leg 1 (+x, ladder)", "leg 2 (+y)", "leg 3 (-x)", "leg 4 (-y)")
HOLD_SECONDS = 1.5
CAMERA_DISTANCE_M = 20.0
CAMERA_HEIGHT_ABOVE_TARGET_M = 5.0
CAMERA_CLEARANCE_ABOVE_GROUND_M = 3.0
CAMERA_SIDE_ANGLE_RAD = 0.6  # look across the slope a little, not straight up it


def camera_eye(renderer: ReplayRenderer, target: numpy.ndarray, distance: float = CAMERA_DISTANCE_M,
               height: float = CAMERA_HEIGHT_ABOVE_TARGET_M) -> numpy.ndarray:
    """Return a camera position on the downhill side of ``target`` that stays above the ground.

    On flat ground the camera looks from the south-west. On a slope it looks from downhill and a
    little to the side, so the hill does not hide the lander, and it is raised until it is at
    least 3 m above the ground under it and under the point halfway to the target. ``distance``
    and ``height`` are the horizontal distance and the height above the target.
    """
    import torch

    point = torch.tensor([float(target[0])]), torch.tensor([float(target[1])])
    slope_x, slope_y = renderer.height_field.gradient_at(*point)
    downhill = -numpy.array([float(slope_x), float(slope_y)])
    if numpy.linalg.norm(downhill) < 0.05:
        downhill = numpy.array([-0.63, -0.77])
    downhill = downhill / numpy.linalg.norm(downhill)
    cosine, sine = numpy.cos(CAMERA_SIDE_ANGLE_RAD), numpy.sin(CAMERA_SIDE_ANGLE_RAD)
    direction = numpy.array([cosine * downhill[0] - sine * downhill[1], sine * downhill[0] + cosine * downhill[1]])
    eye = numpy.array([*(target[:2] + distance * direction), target[2] + height])
    for fraction in (1.0, 0.5):
        sample = target[:2] + fraction * distance * direction
        ground = float(renderer.height_field.height_at(torch.tensor([sample[0]]), torch.tensor([sample[1]])))
        eye[2] = max(eye[2], ground + CAMERA_CLEARANCE_ABOVE_GROUND_M)
    return eye
TOUCHING_GAP_LIMIT_M = 0.03  # contact offset 2 cm plus 1 cm


def read_options() -> dict[str, str]:
    """Return the ``key=value`` options with their defaults."""
    options = {"replay": "", "mode": "video", "width": "1280", "height": "720",
               "camera_distance_m": str(CAMERA_DISTANCE_M), "camera_height_m": str(CAMERA_HEIGHT_ABOVE_TARGET_M), "grain_dot_angle_rad": "0.005",
               "dust_style": "streaks", "view": "chase", "exposure_s": str(1.0 / 60.0), "visibility_gain": "1", "output": ""}
    for argument in script_arguments:
        key, _, value = argument.partition("=")
        if key not in options:
            raise SystemExit(f"unknown option '{key}'")
        options[key] = value
    if not options["replay"]:
        raise SystemExit("give replay=<file.npz>")
    return options


def pad_report(row: numpy.ndarray, columns: dict[str, int]) -> list[tuple[str, float, float, str]]:
    """Return (name, force N, gap m, verdict) per footpad; the verdict compares force and gap."""
    report = []
    for leg in range(4):
        force = float(numpy.linalg.norm([row[columns[f"leg_{leg + 1}_force_{axis}"]] for axis in "xyz"]))
        gap = float(row[columns[f"leg_{leg + 1}_gap"]])
        if force > ARROW_SHOWN_ABOVE_NEWTON:
            verdict = "touching, gap agrees" if abs(gap) < TOUCHING_GAP_LIMIT_M else "CHECK: force but gap"
        else:
            verdict = "free, gap agrees" if gap > -0.005 else "CHECK: no force but inside ground"
        report.append((LEG_NAMES[leg], force, gap, verdict))
    return report


def live_line(row: numpy.ndarray, columns: dict[str, int]) -> str:
    """Return the per-frame state line."""
    return (f"t = {row[columns['time']]:5.1f} s   height above ground {row[columns['height_above_ground']]:6.2f} m   "
            f"vertical {row[columns['speed_vertical']]:+5.2f} m/s   horizontal {row[columns['speed_horizontal']]:4.2f} m/s   "
            f"tilt {row[columns['tilt_deg']]:5.1f} deg   body force {row[columns['body_force']] / 1000:4.1f} kN")


def force_line(row: numpy.ndarray, columns: dict[str, int]) -> str:
    """Return the footpad forces of one frame, in kN."""
    forces = [numpy.linalg.norm([row[columns[f"leg_{leg}_force_{axis}"]] for axis in "xyz"]) / 1000 for leg in range(1, 5)]
    return "footpad forces (kN): " + "   ".join(f"leg {leg + 1}: {force:5.2f}" for leg, force in enumerate(forces))


def grain_readings(row: numpy.ndarray, columns: dict[str, int]) -> dict[str, float]:
    """Return the frame's plume readings of both levels and its time, for the side panel."""
    from lunarsim.tools.replay_recording import GRAIN_COLUMNS

    return {name: float(row[columns[name]]) for name in (*GRAIN_COLUMNS, "time")}


class GrainDrawing:
    """How one replay's grain parcels are drawn: the side panel, plus dots in the scene or streaks on the image."""

    def __init__(self, renderer: ReplayRenderer, replay, options: dict[str, str]) -> None:
        """Read the grain arrays and the drawing options; turn on the depth image for streaks."""
        from lunarsim.tools.plume_grain_video import GrainSidePanel

        self.renderer = renderer
        self.panel = GrainSidePanel(replay, width_px=renderer.height * 8 // 9, height_px=renderer.height)
        self.parcels, self.bins = replay["grain_parcels"], replay["grain_bins"]
        names = [str(name) for name in replay["grain_parcel_columns"]]
        self.velocity_columns = slice(names.index("velocity_x"), names.index("velocity_z") + 1)
        self.mass_column = names.index("mass")
        self.diameters = numpy.asarray(replay["grain_bin_diameters_m"], dtype=numpy.float64)
        self.particle_density = float(replay["grain_particle_density_kg_per_m3"])
        self.extinction_efficiency = float(replay["grain_extinction_efficiency"])
        self.style = options["dust_style"]
        self.dot_angle = float(options["grain_dot_angle_rad"])
        self.exposure = float(options["exposure_s"])
        self.gain = float(options["visibility_gain"])
        if self.style == "streaks":
            renderer.enable_depth()

    def live(self, index: int, clip: int, row: numpy.ndarray):
        """Return the live parcels of one frame: world positions, velocities, masses and bins."""
        parcels, bins = self.parcels[index, clip], self.bins[index, clip]
        alive = numpy.isfinite(parcels[:, 0])
        world = row[:3].astype(numpy.float64) + parcels[alive, :3]
        return world, parcels[alive, self.velocity_columns].astype(numpy.float64), parcels[alive, self.mass_column].astype(numpy.float64), bins[alive].astype(int)

    def before_capture(self, index: int, clip: int, row: numpy.ndarray) -> None:
        """Dots style: put the parcels in the scene as dots of the same on-screen size."""
        if self.style != "dots":
            return
        world, _, _, bins = self.live(index, clip, row)
        widths = self.dot_angle * numpy.linalg.norm(world - self.renderer.eye, axis=1)
        self.renderer.place_grains(clip, world, self.panel.colors[bins], widths)

    def after_capture(self, pixels: numpy.ndarray, index: int, clip: int, row: numpy.ndarray) -> numpy.ndarray:
        """Streaks style: draw each parcel's path during the exposure, with the opacity of its grain cross-section."""
        if self.style != "streaks":
            return pixels
        from lunarsim.tools.plume_grain_video import composite_dust_streaks, parcel_cross_sections

        world, velocity, mass, bins = self.live(index, clip, row)
        cross_sections = parcel_cross_sections(mass, self.diameters[bins], self.particle_density, self.extinction_efficiency)
        image, _ = composite_dust_streaks(pixels, self.renderer.capture_depth(), self.renderer.eye, self.renderer.target, self.renderer.focal_px,
                                          world - velocity * self.exposure, world, cross_sections, gain=self.gain)
        return image

    def caption(self) -> str:
        """Return the caption line that says how the grains are drawn."""
        if self.style == "dots":
            return (f"dots: grain parcels, all drawn {self.dot_angle * 1000:.0f} mrad wide whatever their distance "
                    "(real grains 4-512 um); colour = size, as on the right")
        boost = "faithful (x1)" if self.gain == 1.0 else f"DUST OPACITY x{self.gain:g} (boosted to be seen)"
        return (f"dust: each parcel's path in a 1/{1 / self.exposure:.0f} s exposure; opacity from its grains' "
                f"cross-section: {boost}")


def render_video(renderer: ReplayRenderer, replay, columns: dict[str, int], output: Path, options: dict[str, str]) -> None:
    """Render every clip (chase or window camera, with its grains and side panel if recorded) and join the frames into an MP4."""
    frames, valid = replay["frames"], replay["valid"]
    rate = float(replay["frames_per_second"])
    has_contacts = "leg_1_force_x" in columns
    distance, height = float(options["camera_distance_m"]), float(options["camera_height_m"])
    window = options["view"] == "window"
    grains = GrainDrawing(renderer, replay, options) if "grain_parcels" in replay else None
    panel = grains.panel if grains is not None else None
    depression = float(replay["grain_line_of_sight_depression_deg"]) if "grain_line_of_sight_depression_deg" in replay else 45.0
    directory = output.with_suffix("")
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    written = 0
    for clip in range(frames.shape[1]):
        for other in range(frames.shape[1]):
            renderer.hide(other)
        last = int(numpy.flatnonzero(valid[:, clip]).max())
        first = int(float(replay["clip_start_times"][clip]) * rate)
        indices = list(range(min(first, last), last + 1)) + [last] * int(HOLD_SECONDS * rate)
        target_smooth, eye_smooth = None, None
        for position, index in enumerate([indices[0]] * 8 + indices):
            row = frames[index, clip]
            if window:
                renderer.look(*renderer.window_view(row, depression))
            else:
                target = row[:3].astype(numpy.float64) - numpy.array([0.0, 0.0, 1.0])
                target_smooth = target if target_smooth is None else 0.85 * target_smooth + 0.15 * target
                eye = camera_eye(renderer, target_smooth, distance, height)
                eye_smooth = eye if eye_smooth is None else 0.9 * eye_smooth + 0.1 * eye
                renderer.look(eye_smooth, target_smooth)
            renderer.place(clip, row, columns)
            if grains is not None:
                grains.before_capture(index, clip, row)
            pixels = renderer.capture()
            if position < 8:
                continue  # let the renderer converge on the first view
            if grains is not None:
                pixels = grains.after_capture(pixels, index, clip, row)
            lines = [str(replay["clip_titles"][clip]), str(replay["clip_details"][clip]), live_line(row, columns)]
            if window:
                lines.append(f"WINDOW VIEW: camera fixed to the lander, looking forward and {depression:.0f} deg down (the sensor line of sight)")
            elif has_contacts:
                lines.append(force_line(row, columns))
            if grains is not None:
                lines.append(grains.caption())
            title_color = (120, 220, 120) if "SUCCESS" in lines[0] or "AT REST" in lines[0] else (255, 120, 120)
            image = draw_caption(Image.fromarray(pixels), lines, title_color, size=16 if panel is not None else 18)
            if panel is not None:
                side = Image.fromarray(panel.draw(grains.parcels[index, clip], grains.bins[index, clip], grain_readings(row, columns)))
                combined = Image.new("RGB", (image.width + side.width, image.height))
                combined.paste(image, (0, 0))
                combined.paste(side, (image.width, 0))
                image = combined
            image.save(directory / f"frame_{written:06d}.png")
            written += 1
        print(f"[render] clip {clip + 1}/{frames.shape[1]} done, {written} frames", flush=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{rate:g}", "-i", str(directory / "frame_%06d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(output)],
        check=True,
    )
    shutil.rmtree(directory)
    print(f"[render] wrote {output} ({written} frames, {written / rate:.0f} s)")


def render_stills(renderer: ReplayRenderer, replay, columns: dict[str, int], output_directory: Path) -> None:
    """Render, per clip, a wide view and four footpad close-ups of the last frame into one PNG."""
    frames, valid = replay["frames"], replay["valid"]
    output_directory.mkdir(parents=True, exist_ok=True)
    width, height = renderer.width, renderer.height
    summary = []
    for clip in range(frames.shape[1]):
        for other in range(frames.shape[1]):
            renderer.hide(other)
        last = int(numpy.flatnonzero(valid[:, clip]).max())
        row = frames[last, clip]
        renderer.place(clip, row, columns)
        centre = row[:3].astype(numpy.float64)
        wide_eye = centre + 0.8 * (camera_eye(renderer, centre) - centre)  # a little closer than in videos
        renderer.look(wide_eye, centre - numpy.array([0.0, 0.0, 2.0]))
        wide = Image.fromarray(renderer.capture(settle_updates=20))
        report = pad_report(row, columns)
        draw_caption(wide, [str(replay["clip_titles"][clip]), str(replay["clip_details"][clip]), live_line(row, columns)],
                     (120, 220, 120))
        sheet = Image.new("RGB", (width, height * 2), (0, 0, 0))
        sheet.paste(wide, (0, 0))
        for leg in range(4):
            leg_position = numpy.array([row[columns[f"leg_{leg + 1}_{axis}"]] for axis in "xyz"], dtype=numpy.float64)
            radial = leg_position[:2] - centre[:2]
            radial = radial / max(numpy.linalg.norm(radial), 1e-6)
            tangent = numpy.array([-radial[1], radial[0]])
            eye = leg_position + numpy.array([*(2.6 * radial + 1.6 * tangent), 0.6])
            renderer.look(eye, leg_position - numpy.array([0.0, 0.0, 0.25]))
            tile = Image.fromarray(renderer.capture(settle_updates=20)).resize((width // 2, height // 2))
            name, force, gap, verdict = report[leg]
            draw_caption(tile, [name, f"sensor {force / 1000:5.2f} kN   gap {100 * gap:+5.1f} cm", verdict],
                         (120, 220, 120) if not verdict.startswith("CHECK") else (255, 120, 120), size=15)
            sheet.paste(tile, ((leg % 2) * (width // 2), height + (leg // 2) * (height // 2)))
            summary.append((clip, str(replay["clip_titles"][clip]), name, force, gap, verdict))
        path = output_directory / f"contact_{clip + 1:02d}.png"
        sheet.save(path)
        print(f"[render] wrote {path}", flush=True)
    with open(output_directory / "contact_summary.csv", "w") as file:
        file.write("clip,title,leg,force_n,gap_m,verdict\n")
        for clip, title, name, force, gap, verdict in summary:
            file.write(f"{clip + 1},\"{title}\",\"{name}\",{force:.1f},{gap:.4f},{verdict}\n")
    checks = [entry for entry in summary if entry[5].startswith("CHECK")]
    print(f"[render] {len(summary)} footpads, {len(checks)} disagree between sensor and gap")


def main() -> None:
    """Load the replay, build the stage and render."""
    options = read_options()
    replay_path = Path(options["replay"])
    replay = numpy.load(replay_path)
    columns = {str(name): index for index, name in enumerate(replay["columns"])}
    renderer = ReplayRenderer(
        simulation_app, str(replay["scenario_name"]), [str(value) for value in replay["scenario_overrides"]],
        replay["frames"].shape[1], int(options["width"]), int(options["height"]),
    )
    if options["mode"] == "video":
        if options["dust_style"] not in ("dots", "streaks") or options["view"] not in ("chase", "window"):
            raise SystemExit("dust_style is dots or streaks; view is chase or window")
        output = Path(options["output"]) if options["output"] else replay_path.with_suffix(".mp4")
        render_video(renderer, replay, columns, output, options)
    else:
        render_stills(renderer, replay, columns, replay_path.parent / f"{replay_path.stem}_stills")


if __name__ == "__main__":
    main()
    simulation_app.close()
