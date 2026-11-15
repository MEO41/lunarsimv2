# SPDX-License-Identifier: BSD-3-Clause
"""Plots of the visual sensor check (``scripts/sensor_check.py``); needs no simulator.

Writes into ``runs/sensor_check/figures/``:

- ``lander_<n>.png``: per lander, eight panels against time: footpad height (truth, filter,
  contact light), altimeter slant range, Doppler line-of-sight velocities and ranges, IMU
  specific force and body rates, and the filter's position and velocity errors with their
  plus/minus three-sigma bands. Measured values are dots (one per sample), expected values (from
  the true state, no noise) are lines.
- ``normalised_residuals.png``: for every sensor, (measured - expected) / stated sigma over all
  samples of all landers, against the standard normal curve. A sensor that matches its stated
  noise gives mean 0 and standard deviation 1.
- ``lidar_frames.png``: LiDAR range images of lander 2 at three moments: measured, expected, and
  the normalised difference.
"""

import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
TRUTH = "#3d3d3d"
ESTIMATE = "#1f6fd1"
BEAM_COLORS = ("#d1495b", "#2e9e5b", "#7b4fc4")
AXIS_COLORS = ("#d1495b", "#2e9e5b", "#1f6fd1")
MEASURED = "#e07b00"
PROBE_LENGTH_M = 1.727


def load(path: Path) -> dict:
    """Return the recording as a dictionary of arrays."""
    with numpy.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def lander_slice(data: dict, index: int) -> dict:
    """Return the steps of one lander's first episode."""
    keep = data["valid_record"][:, index] > 0.5
    return {name: values[keep, index] for name, values in data.items() if values.ndim >= 2 and values.shape[1] == data["valid_record"].shape[1] and not name.endswith("directions_body")}


def rotation_vector_error(estimated: numpy.ndarray, true: numpy.ndarray) -> numpy.ndarray:
    """Return the attitude error (rad) as the rotation vector of conj(estimated) * true, (steps, 3)."""
    w1, v1 = estimated[:, :1], -estimated[:, 1:]
    w2, v2 = true[:, :1], true[:, 1:]
    w = w1 * w2 - numpy.sum(v1 * v2, axis=1, keepdims=True)
    v = w1 * v2 + w2 * v1 + numpy.cross(v1, v2)
    sign = numpy.where(w < 0, -1.0, 1.0)
    w, v = w * sign, v * sign
    norm = numpy.linalg.norm(v, axis=1, keepdims=True)
    angle = 2.0 * numpy.arctan2(norm, w)
    return numpy.where(norm > 1e-9, v * angle / numpy.maximum(norm, 1e-12), 2.0 * v)


def style(axis, title: str, unit: str) -> None:
    """Apply the common look: title, unit label, light grid."""
    axis.set_title(title, fontsize=10, loc="left")
    axis.set_ylabel(unit, fontsize=9)
    axis.grid(color="#dddddd", linewidth=0.6)
    axis.tick_params(labelsize=8)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)


def plot_lander(lander: dict, name: str, dt: float, output: Path) -> None:
    """Draw the eight panels of one lander."""
    time = lander["time"]
    touch = numpy.flatnonzero(lander["true_leg_force"] > 100.0)
    touchdown = time[touch[0]] if touch.size else None
    figure, axes = plt.subplots(4, 2, figsize=(15, 13), sharex=True)
    figure.suptitle(f"Sensor check, free fall from 30 m (engine off, no jets): {name}", fontsize=13, x=0.01, ha="left")

    def mark(axis):
        """Mark the first footpad contact on a panel."""
        if touchdown is not None:
            axis.axvline(touchdown, color="#999999", linestyle=":", linewidth=1)

    axis = axes[0, 0]
    axis.plot(time, lander["true_footpad_height"], color=TRUTH, linewidth=1.6, label="truth")
    height_estimate = lander["estimated_position"][:, 2] - lander["true_position"][:, 2] + lander["true_footpad_height"]
    axis.plot(time, height_estimate, color=ESTIMATE, linewidth=1.0, linestyle="--", label="EKF estimate")
    light = lander["true_probe_contact"] > 0.5
    if light.any():
        first = time[numpy.argmax(light)]
        axis.axvspan(first, time[-1], color="#f2c94c", alpha=0.25, label="contact light on")
        axis.axhline(PROBE_LENGTH_M, color="#b8860b", linewidth=0.8, linestyle="-.", label="probe length 1.727 m")
    style(axis, "Footpad height above the ground", "m")
    axis.legend(fontsize=8, frameon=False)
    mark(axis)

    axis = axes[0, 1]
    valid = lander["measured_altimeter_valid"] > 0.5
    axis.plot(time, lander["expected_altimeter_range"][:, 0], color=TRUTH, linewidth=1.4, label="expected (true pose)")
    axis.plot(time[valid], lander["measured_altimeter"][valid, 0], ".", color=MEASURED, markersize=3, label="measured (20 Hz)")
    axis.axhline(3.05, color="#999999", linewidth=0.8, linestyle="-.", label="minimum range 3.05 m")
    style(axis, "Radar altimeter slant range (beam along body -z)", "m")
    axis.legend(fontsize=8, frameon=False)
    mark(axis)

    doppler_valid = lander["measured_doppler_valid"] > 0.5
    for panel, offset, title, unit in ((axes[1, 0], 0, "Doppler line-of-sight velocity, 3 beams", "m/s"), (axes[1, 1], 3, "Doppler slant ranges, 3 beams", "m")):
        expected = lander["expected_doppler_velocity"] if offset == 0 else lander["expected_doppler_range"]
        for beam in range(3):
            panel.plot(time, expected[:, beam], color=BEAM_COLORS[beam], linewidth=1.2, label=f"beam {beam + 1} expected")
            measured = lander["measured_doppler"][:, offset + beam]
            panel.plot(time[doppler_valid], measured[doppler_valid], ".", color=BEAM_COLORS[beam], markersize=2.5, alpha=0.7)
        style(panel, title + " (lines expected, dots measured)", unit)
        panel.legend(fontsize=8, frameon=False)
        mark(panel)

    imu_valid = lander["measured_imu_valid"] > 0.5
    axis = axes[2, 0]
    for k, label in enumerate("xyz"):
        axis.plot(time, lander["expected_specific_force"][:, k], color=AXIS_COLORS[k], linewidth=1.2, label=f"{label} expected")
        axis.plot(time[imu_valid], lander["measured_imu"][imu_valid, k] / dt, ".", color=AXIS_COLORS[k], markersize=1.5, alpha=0.6)
    axis.set_yscale("symlog", linthresh=0.05)
    style(axis, "IMU specific force, body frame (zero in free fall; spike at impact)", "m/s^2 (symlog)")
    axis.legend(fontsize=8, frameon=False, ncol=3)
    mark(axis)

    axis = axes[2, 1]
    for k, label in enumerate("xyz"):
        axis.plot(time, numpy.degrees(lander["true_body_rates"][:, k]), color=AXIS_COLORS[k], linewidth=1.2, label=f"{label} true")
        axis.plot(time[imu_valid], numpy.degrees(lander["measured_imu"][imu_valid, 3 + k] / dt), ".", color=AXIS_COLORS[k], markersize=1.5, alpha=0.6)
    style(axis, "IMU body rates (gyro delta-angle / step)", "deg/s")
    axis.legend(fontsize=8, frameon=False, ncol=3)
    mark(axis)

    sigma = numpy.sqrt(numpy.maximum(lander["estimated_covariance_diagonal"], 0.0))
    position_error = lander["estimated_position"] - lander["true_position"]
    velocity_error = lander["estimated_velocity"] - lander["true_velocity"]
    for panel, error, block, title, unit, scale in (
        (axes[3, 0], position_error, 0, "EKF position error with +/-3 sigma (x, y, z)", "m", 1.0),
        (axes[3, 1], velocity_error, 3, "EKF velocity error with +/-3 sigma (x, y, z)", "mm/s", 1000.0),
    ):
        for k, label in enumerate("xyz"):
            panel.plot(time, scale * error[:, k], color=AXIS_COLORS[k], linewidth=1.0, label=f"{label} error")
            panel.fill_between(time, -3 * scale * sigma[:, block + k], 3 * scale * sigma[:, block + k], color=AXIS_COLORS[k], alpha=0.10)
        style(panel, title, unit)
        panel.legend(fontsize=8, frameon=False, ncol=3)
        panel.set_xlabel("time since release (s)", fontsize=9)
        mark(panel)
    axes[3, 1].set_ylim(-40, 40)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output, dpi=110)
    plt.close(figure)


def normalised_residuals(data: dict, dt: float) -> dict[str, numpy.ndarray]:
    """Return (measured - expected) / stated sigma for every sensor, all landers together."""
    result = {name: [] for name in ("IMU delta-velocity", "IMU delta-angle", "altimeter range", "Doppler velocity", "Doppler range", "LiDAR range")}
    for index in range(data["valid_record"].shape[1]):
        lander = lander_slice(data, index)
        imu_valid = lander["measured_imu_valid"] > 0.5
        imu_valid[0] = False
        bias = lander["true_biases"]
        expected_dv = (lander["expected_specific_force"] + bias[:, :3]) * dt
        mean_rate = 0.5 * (lander["true_body_rates"][1:] + lander["true_body_rates"][:-1])
        expected_dtheta = numpy.vstack((lander["true_body_rates"][:1], mean_rate)) * dt + bias[:, 3:] * dt
        variance = lander["measured_imu_variance"]
        result["IMU delta-velocity"].append(((lander["measured_imu"][:, :3] - expected_dv) / numpy.sqrt(variance[:, :3]))[imu_valid].ravel())
        result["IMU delta-angle"].append(((lander["measured_imu"][:, 3:] - expected_dtheta) / numpy.sqrt(variance[:, 3:]))[imu_valid].ravel())
        for label, measured, expected, variance, valid in (
            ("altimeter range", lander["measured_altimeter"], lander["expected_altimeter_range"], lander["measured_altimeter_variance"], lander["measured_altimeter_valid"]),
            ("Doppler velocity", lander["measured_doppler"][:, :3], lander["expected_doppler_velocity"], lander["measured_doppler_variance"][:, :3], lander["measured_doppler_valid"]),
            ("Doppler range", lander["measured_doppler"][:, 3:], lander["expected_doppler_range"], lander["measured_doppler_variance"][:, 3:], lander["measured_doppler_valid"]),
            ("LiDAR range", lander.get("measured_lidar"), lander.get("expected_lidar_range"), lander.get("measured_lidar_variance"), lander.get("measured_lidar_valid")),
        ):
            if measured is None:
                continue
            values = (measured - expected) / numpy.sqrt(variance)
            values = values[valid > 0.5]
            result[label].append(values[numpy.isfinite(values)].ravel())
    return {name: numpy.concatenate(parts) if parts else numpy.array([]) for name, parts in result.items()}


def plot_residuals(residuals: dict[str, numpy.ndarray], output: Path) -> None:
    """Histograms of normalised residuals against the standard normal density."""
    figure, axes = plt.subplots(2, 3, figsize=(15, 8))
    grid = numpy.linspace(-5, 5, 400)
    normal = numpy.exp(-0.5 * grid**2) / math.sqrt(2 * math.pi)
    for axis, (name, values) in zip(axes.ravel(), residuals.items()):
        axis.hist(values, bins=numpy.linspace(-5, 5, 61), density=True, color="#9fb7d9", edgecolor="white", linewidth=0.4)
        axis.plot(grid, normal, color=TRUTH, linewidth=1.4, label="standard normal")
        mean, spread = (float(values.mean()), float(values.std())) if values.size else (float("nan"), float("nan"))
        style(axis, f"{name}\nmean {mean:+.3f}, standard deviation {spread:.3f}, {values.size:,} samples", "density")
        axis.set_xlabel("(measured - expected) / stated sigma", fontsize=9)
        axis.legend(fontsize=8, frameon=False)
    figure.suptitle("Do the sensors match their stated noise? (all four free falls)", fontsize=13, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output, dpi=110)
    plt.close(figure)


def plot_lidar(data: dict, index: int, output: Path) -> None:
    """LiDAR range images of one lander at three sample times."""
    lander = lander_slice(data, index)
    valid = numpy.flatnonzero(lander["measured_lidar_valid"] > 0.5)
    if valid.size == 0:
        return
    picks = valid[[0, valid.size // 2, int(valid.size * 0.85)]]
    side = int(round(math.sqrt(lander["measured_lidar"].shape[1])))
    figure, axes = plt.subplots(3, 3, figsize=(12, 11))
    for row, step in enumerate(picks):
        measured = lander["measured_lidar"][step].reshape(side, side)
        expected = lander["expected_lidar_range"][step].reshape(side, side)
        normalised = (measured - expected) / numpy.sqrt(lander["measured_lidar_variance"][step].reshape(side, side))
        limits = (numpy.nanmin(expected), numpy.nanmax(expected))
        for column, (image, title, colormap, vmin, vmax) in enumerate(
            ((measured, "measured range (m)", "viridis", *limits), (expected, "expected range (m)", "viridis", *limits), (normalised, "(measured - expected) / sigma", "RdBu_r", -3, 3))
        ):
            axis = axes[row, column]
            shown = axis.imshow(image, cmap=colormap, vmin=vmin, vmax=vmax, origin="lower")
            axis.set_title(f"t = {lander['time'][step]:.1f} s: {title}", fontsize=9, loc="left")
            axis.set_xticks([])
            axis.set_yticks([])
            figure.colorbar(shown, ax=axis, fraction=0.046)
    figure.suptitle("Flash LiDAR (32 x 32 pixels, 6 deg field of view, boresight 30 deg off -z), lander 2", fontsize=12, x=0.01, ha="left")
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output, dpi=110)
    plt.close(figure)


def main() -> None:
    """Make all figures of the sensor check."""
    run = Path(sys.argv[1]) if len(sys.argv) > 1 else REPOSITORY_ROOT / "runs" / "sensor_check"
    data = load(run / "sensors.npz")
    names = [str(name) for name in data["meta_names"]]
    dt = float(data["meta_control_step_s"])
    output = run / "figures"
    output.mkdir(exist_ok=True)
    for index, name in enumerate(names):
        plot_lander(lander_slice(data, index), name, dt, output / f"lander_{index + 1}.png")
    residuals = normalised_residuals(data, dt)
    plot_residuals(residuals, output / "normalised_residuals.png")
    plot_lidar(data, 1, output / "lidar_frames.png")
    for name, values in residuals.items():
        print(f"[sensor check] {name:20s} mean {values.mean():+.3f}  sd {values.std():.3f}  samples {values.size}")
    print(f"[sensor check] figures in {output}")


if __name__ == "__main__":
    main()
