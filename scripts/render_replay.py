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

Options: ``replay``, ``mode`` (video or stills), ``width`` (1280), ``height`` (720).
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


def camera_eye(renderer: ReplayRenderer, target: numpy.ndarray) -> numpy.ndarray:
    """Return a camera position on the downhill side of ``target`` that stays above the ground.

    On flat ground the camera looks from the south-west. On a slope it looks from downhill and a
    little to the side, so the hill does not hide the lander, and it is raised until it is at
    least 3 m above the ground under it and under the point halfway to the target.
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
    eye = numpy.array([*(target[:2] + CAMERA_DISTANCE_M * direction), target[2] + CAMERA_HEIGHT_ABOVE_TARGET_M])
    for fraction in (1.0, 0.5):
        sample = target[:2] + fraction * CAMERA_DISTANCE_M * direction
        ground = float(renderer.height_field.height_at(torch.tensor([sample[0]]), torch.tensor([sample[1]])))
        eye[2] = max(eye[2], ground + CAMERA_CLEARANCE_ABOVE_GROUND_M)
    return eye
TOUCHING_GAP_LIMIT_M = 0.03  # contact offset 2 cm plus 1 cm


def read_options() -> dict[str, str]:
    """Return the ``key=value`` options with their defaults."""
    options = {"replay": "", "mode": "video", "width": "1280", "height": "720"}
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


def render_video(renderer: ReplayRenderer, replay, columns: dict[str, int], output: Path) -> None:
    """Render every clip following the lander and join the frames into an MP4."""
    frames, valid = replay["frames"], replay["valid"]
    rate = float(replay["frames_per_second"])
    has_contacts = "leg_1_force_x" in columns
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
            target = row[:3].astype(numpy.float64) - numpy.array([0.0, 0.0, 1.0])
            target_smooth = target if target_smooth is None else 0.85 * target_smooth + 0.15 * target
            eye = camera_eye(renderer, target_smooth)
            eye_smooth = eye if eye_smooth is None else 0.9 * eye_smooth + 0.1 * eye
            renderer.look(eye_smooth, target_smooth)
            renderer.place(clip, row, columns)
            pixels = renderer.capture()
            if position < 8:
                continue  # let the renderer converge on the first view
            lines = [str(replay["clip_titles"][clip]), str(replay["clip_details"][clip]), live_line(row, columns)]
            if has_contacts:
                lines.append(force_line(row, columns))
            title_color = (120, 220, 120) if "SUCCESS" in lines[0] or "AT REST" in lines[0] else (255, 120, 120)
            draw_caption(Image.fromarray(pixels), lines, title_color).save(directory / f"frame_{written:06d}.png")
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
        render_video(renderer, replay, columns, replay_path.with_suffix(".mp4"))
    else:
        render_stills(renderer, replay, columns, replay_path.parent / f"{replay_path.stem}_stills")


if __name__ == "__main__":
    main()
    simulation_app.close()
