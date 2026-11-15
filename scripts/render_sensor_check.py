# SPDX-License-Identifier: BSD-3-Clause
"""Video of the visual sensor check: each free fall with its sensor beams and live sensor traces.

Usage (after ``scripts/sensor_check.py``)::

    python scripts/render_sensor_check.py

Left (1280 x 720): the lander replayed on the Apollo 17 terrain (physics-free stage, as in
``render_replay.py``) with the radar altimeter beam (yellow), the three Doppler beams (red,
green, purple, as in the plots) drawn from the centre of mass to the measured range, and the
LiDAR returns as magenta points on the ground at their measured ranges (every second pixel of
the 32 x 32 grid). Right (640 x 720): altimeter range, Doppler velocities, IMU vertical specific
force and the filter's height error with its 3-sigma band, with a cursor at the current time.
Writes ``runs/sensor_check/sensor_check.mp4``.
"""

import argparse
import shutil
import subprocess
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render the sensor check video.")
AppLauncher.add_app_launcher_args(parser)
arguments, _ = parser.parse_known_args()
arguments.enable_cameras = True
arguments.headless = True
simulation_app = AppLauncher(arguments).app

import io  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy  # noqa: E402
import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from lunarsim.tools.replay_rendering import ReplayRenderer, draw_caption, rotation_from_z_to  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RUN = REPOSITORY_ROOT / "runs" / "sensor_check"
ALTIMETER_COLOR = (0.95, 0.80, 0.10)
DOPPLER_COLORS = ((0.82, 0.29, 0.36), (0.18, 0.62, 0.36), (0.48, 0.31, 0.77))
LIDAR_COLOR = (0.95, 0.20, 0.85)
BEAM_RADIUS_M = 0.07
LIDAR_POINT_WIDTH_M = 0.35
PANEL_WIDTH, PANEL_HEIGHT = 640, 720
HOLD_SECONDS = 1.0


def quaternion_rotate(quaternion: numpy.ndarray, vectors: numpy.ndarray) -> numpy.ndarray:
    """Rotate body vectors (K, 3) into the world by one quaternion (w, x, y, z)."""
    w, x, y, z = quaternion
    rotation = numpy.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return vectors @ rotation.T


class BeamOverlay:
    """Cylinders for the altimeter and Doppler beams and a point cloud for the LiDAR returns."""

    def __init__(self, stage) -> None:
        """Create the hidden beam prims."""
        from pxr import Gf, UsdGeom

        self.stage = stage
        self.beams = []
        for index, color in enumerate((ALTIMETER_COLOR, *DOPPLER_COLORS)):
            path = f"/World/SensorBeams/beam_{index}"
            cylinder = UsdGeom.Cylinder.Define(stage, path)
            cylinder.CreateRadiusAttr(BEAM_RADIUS_M)
            cylinder.CreateHeightAttr(1.0)
            cylinder.CreateDisplayColorAttr([Gf.Vec3f(*color)])
            xformable = UsdGeom.Xformable(cylinder)
            operations = (xformable.AddTranslateOp(), xformable.AddOrientOp(), xformable.AddScaleOp())
            self.beams.append((cylinder, operations))
        self.points = UsdGeom.Points.Define(stage, "/World/SensorBeams/lidar")
        self.points.CreateDisplayColorAttr([Gf.Vec3f(*LIDAR_COLOR)])

    def set_beam(self, index: int, start: numpy.ndarray, end: numpy.ndarray | None) -> None:
        """Draw beam ``index`` from ``start`` to ``end``; None hides it (no return)."""
        from pxr import Gf, UsdGeom

        cylinder, (translate, orient, scale) = self.beams[index]
        if end is None or not numpy.all(numpy.isfinite(end)):
            UsdGeom.Imageable(cylinder).MakeInvisible()
            return
        UsdGeom.Imageable(cylinder).MakeVisible()
        vector = end - start
        length = float(numpy.linalg.norm(vector))
        translate.Set(Gf.Vec3d(*(0.5 * (start + end))))
        orient.Set(rotation_from_z_to(vector / max(length, 1e-6)))
        scale.Set(Gf.Vec3f(1.0, 1.0, length))

    def set_points(self, points: numpy.ndarray) -> None:
        """Show the LiDAR returns (K, 3); an empty array hides them."""
        from pxr import Vt

        points = points[numpy.all(numpy.isfinite(points), axis=1)]
        self.points.GetPointsAttr().Set(Vt.Vec3fArray.FromNumpy(points.astype(numpy.float32)))
        self.points.GetWidthsAttr().Set(Vt.FloatArray.FromNumpy(numpy.full(len(points), LIDAR_POINT_WIDTH_M, dtype=numpy.float32)))


def panel_image(sensors: dict, index: int):
    """Draw the right-hand trace panel of one lander; return (image, function time -> x pixel)."""
    keep = sensors["valid_record"][:, index] > 0.5
    time = sensors["time"][keep, index]
    dt = float(sensors["meta_control_step_s"])
    figure, axes = plt.subplots(4, 1, figsize=(PANEL_WIDTH / 100, PANEL_HEIGHT / 100), dpi=100, sharex=True)
    altimeter_valid = sensors["measured_altimeter_valid"][keep, index] > 0.5
    axes[0].plot(time, sensors["expected_altimeter_range"][keep, index, 0], color="#3d3d3d", linewidth=1.2)
    axes[0].plot(time[altimeter_valid], sensors["measured_altimeter"][keep, index, 0][altimeter_valid], ".", color="#e0a800", markersize=2)
    axes[0].set_title("altimeter range (m): line expected, dots measured", fontsize=8, loc="left")
    doppler_valid = sensors["measured_doppler_valid"][keep, index] > 0.5
    for beam, color in enumerate(("#d1495b", "#2e9e5b", "#7b4fc4")):
        axes[1].plot(time, sensors["expected_doppler_velocity"][keep, index, beam], color=color, linewidth=1.0)
        axes[1].plot(time[doppler_valid], sensors["measured_doppler"][keep, index, beam][doppler_valid], ".", color=color, markersize=1.5)
    axes[1].set_title("Doppler line-of-sight velocity (m/s), 3 beams", fontsize=8, loc="left")
    imu_valid = sensors["measured_imu_valid"][keep, index] > 0.5
    axes[2].plot(time, sensors["expected_specific_force"][keep, index, 2], color="#3d3d3d", linewidth=1.0)
    axes[2].plot(time[imu_valid], sensors["measured_imu"][keep, index, 2][imu_valid] / dt, ".", color="#1f6fd1", markersize=1.5)
    axes[2].set_yscale("symlog", linthresh=0.05)
    axes[2].set_title("IMU specific force, body z (m/s^2): 0 in free fall", fontsize=8, loc="left")
    error = sensors["estimated_position"][keep, index, 2] - sensors["true_position"][keep, index, 2]
    sigma = numpy.sqrt(numpy.maximum(sensors["estimated_covariance_diagonal"][keep, index, 2], 0.0))
    axes[3].plot(time, 100 * error, color="#1f6fd1", linewidth=1.0)
    axes[3].fill_between(time, -300 * sigma, 300 * sigma, color="#1f6fd1", alpha=0.15)
    axes[3].set_title("EKF height error (cm) with 3-sigma band", fontsize=8, loc="left")
    axes[3].set_xlabel("time since release (s)", fontsize=8)
    for axis in axes:
        axis.tick_params(labelsize=7)
        axis.grid(color="#dddddd", linewidth=0.5)
    figure.tight_layout()
    figure.canvas.draw()
    transform = axes[3].transData
    height = figure.bbox.height

    def time_to_pixel(value: float) -> float:
        """Return the panel's x pixel of a time."""
        return float(transform.transform((value, 0.0))[0])

    top = int(height - axes[0].get_window_extent().y1)
    bottom = int(height - axes[3].get_window_extent().y0)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100)
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB").resize((PANEL_WIDTH, PANEL_HEIGHT)), time_to_pixel, (top, bottom)


def main() -> None:
    """Render every lander's fall with beams and traces, then join the frames into an MP4."""
    sensors = dict(numpy.load(RUN / "sensors.npz"))
    replay = numpy.load(RUN / "replay.npz")
    columns = {str(name): index for index, name in enumerate(replay["columns"])}
    frames, valid = replay["frames"], replay["valid"]
    count = frames.shape[1]
    renderer = ReplayRenderer(simulation_app, str(replay["scenario_name"]), [str(v) for v in replay["scenario_overrides"]], count)
    overlay = BeamOverlay(renderer.stage)
    site_origin = replay["site_origin"]
    directions = {
        "altimeter": sensors["altimeter_directions_body"],
        "doppler": sensors["doppler_directions_body"],
        "lidar": sensors["lidar_directions_body"],
    }
    lidar_side = int(round(numpy.sqrt(directions["lidar"].shape[0])))
    lidar_subset = numpy.array([row * lidar_side + column for row in range(0, lidar_side, 2) for column in range(0, lidar_side, 2)])
    directory = RUN / "sensor_frames"
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    rate = float(replay["frames_per_second"])
    written = 0
    for clip in range(count):
        for other in range(count):
            renderer.hide(other)
        panel, time_to_pixel, (top, bottom) = panel_image(sensors, clip)
        sensor_time = sensors["time"][:, clip]
        sensor_keep = sensors["valid_record"][:, clip] > 0.5
        last = int(numpy.flatnonzero(valid[:, clip]).max())
        indices = list(range(0, last + 1)) + [last] * int(HOLD_SECONDS * rate)
        target_smooth, distance_smooth = None, None
        for position, frame_index in enumerate([0] * 8 + indices):
            row = frames[frame_index, clip]
            time = float(row[columns["time"]])
            step = int(numpy.flatnonzero(sensor_keep)[numpy.argmin(numpy.abs(sensor_time[sensor_keep] - time))])
            centre = row[:3].astype(numpy.float64)
            quaternion = row[3:7].astype(numpy.float64)
            renderer.place(clip, row, columns)
            altimeter_range = float(sensors["measured_altimeter"][step, clip, 0])
            altimeter_direction = quaternion_rotate(quaternion, directions["altimeter"])[0]
            overlay.set_beam(0, centre, centre + altimeter_range * altimeter_direction if numpy.isfinite(altimeter_range) else None)
            doppler_directions = quaternion_rotate(quaternion, directions["doppler"])
            for beam in range(3):
                doppler_range = float(sensors["measured_doppler"][step, clip, 3 + beam])
                overlay.set_beam(1 + beam, centre, centre + doppler_range * doppler_directions[beam] if numpy.isfinite(doppler_range) else None)
            lidar_directions = quaternion_rotate(quaternion, directions["lidar"][lidar_subset])
            lidar_ranges = sensors["measured_lidar"][step, clip, lidar_subset]
            lidar_points = centre + lidar_ranges[:, None] * lidar_directions
            overlay.set_points(lidar_points)
            # Frame the lander, the ground under it and the LiDAR spot ahead of it, and pull the
            # camera back as the height grows, so every beam stays in view down to the ground.
            ground_below = float(renderer.height_field.height_at(torch.tensor([centre[0]]), torch.tensor([centre[1]])))
            lidar_centre = numpy.nanmean(lidar_points, axis=0) if numpy.isfinite(lidar_points).any() else centre
            target = 0.5 * centre + 0.3 * numpy.array([centre[0], centre[1], ground_below]) + 0.2 * lidar_centre
            height = max(centre[2] - ground_below, 4.0)
            distance = 16.0 + 1.7 * height
            target_smooth = target if target_smooth is None else 0.85 * target_smooth + 0.15 * target
            distance_smooth = distance if position == 0 else 0.85 * distance_smooth + 0.15 * distance
            eye = target_smooth + distance_smooth * numpy.array([-0.47, -0.85, 0.22])
            ground = float(renderer.height_field.height_at(torch.tensor([eye[0]]), torch.tensor([eye[1]])))
            eye[2] = max(eye[2], ground + 3.0)
            renderer.look(eye, target_smooth)
            pixels = renderer.capture()
            if position < 8:
                continue
            left = draw_caption(
                Image.fromarray(pixels),
                [str(replay["clip_titles"][clip]), str(replay["clip_details"][clip]),
                 f"t = {time:4.1f} s   footpad height {row[columns['height_above_ground']]:6.2f} m   vertical {row[columns['speed_vertical']]:+6.2f} m/s   "
                 f"beams: altimeter yellow, Doppler red/green/purple, LiDAR points magenta"],
                (255, 210, 90),
            )
            right = panel.copy()
            cursor = time_to_pixel(time) * PANEL_WIDTH / panel.width
            ImageDraw.Draw(right).line([(cursor, top), (cursor, bottom)], fill=(220, 0, 0), width=2)
            sheet = Image.new("RGB", (left.width + PANEL_WIDTH, max(left.height, PANEL_HEIGHT)), (255, 255, 255))
            sheet.paste(left, (0, 0))
            sheet.paste(right, (left.width, 0))
            sheet.save(directory / f"frame_{written:06d}.png")
            written += 1
        print(f"[render] lander {clip + 1}/{count} done, {written} frames", flush=True)
    output = RUN / "sensor_check.mp4"
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", f"{rate:g}", "-i", str(directory / "frame_%06d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(output)],
        check=True,
    )
    shutil.rmtree(directory)
    print(f"[render] wrote {output} ({written} frames, {written / rate:.0f} s)")


if __name__ == "__main__":
    main()
    simulation_app.close()
