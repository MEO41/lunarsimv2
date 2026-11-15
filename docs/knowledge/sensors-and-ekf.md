---
id: sensors-and-ekf
title: Sensors and EKF (milestone 2)
type: component
status: current
updated: 2026-11-15
summary: The terminal_suite sensor binding (IMU, radar altimeter, Doppler velocimeter, propellant gauge, probe switches), the ekf navigation, scenario S2 and the consistency results.
---

# Sensors and EKF (milestone 2)

## Files

| File | What it does |
| --- | --- |
| `src/lunarsim/slots/sensors/sensor_models.py` | `SampleSchedule` (rate per sensor), `InertialMeasurementUnit`, `RangeBeams` (altimeter and Doppler beams, ray cast on the ground lookup), `PropellantGauge`, `ContactProbeSwitch` |
| `src/lunarsim/slots/sensors/terminal_suite.py` | `sensors: terminal_suite`, one message per sensor (`measurement.imu`, `.radar_altimeter`, `.doppler_velocimeter`, `.propellant_gauge`, `.contact_probe`) |
| `src/lunarsim/slots/navigation/error_state_kalman_filter.py` | Batched 15-state error-state filter (Solà 2017), float64 covariance, Joseph update |
| `src/lunarsim/slots/navigation/ekf.py` | `nav: ekf`: IMU propagation, altimeter and Doppler updates, 16-value estimate |
| `src/lunarsim/slots/start/initial_navigation_error.py` | First estimate = start state + sampled error (`start.nav_error`), message `initial_navigation_state`, own generator so the true starts do not change |
| `configs/sensors/terminal_suite.yaml`, `configs/nav/ekf.yaml`, `configs/scenarios/s2_terminal_ekf.yaml` | Numbers with sources |

## Sensor numbers (all in `configs/sensors/terminal_suite.yaml` with sources)

| Sensor | Values | Source |
| --- | --- | --- |
| IMU | gyro bias 1°/h, ARW 0.07°/√h, accelerometer bias 300 µg, noise 35 µg/√Hz; delta-v and delta-angle per 20 ms | LN-200S datasheet (Northrop Grumman CS-23015) |
| Radar altimeter | 1 beam along −z, 3.05 m to 7.62 km, 20 Hz, σ = 0.16 m + 1.2e-4·range | TN D-6849 (one beam), Grumman GN&C reference (10–25,000 ft); noise and rate are the NDL values (**model choice**: Apollo altimeter accuracy not public) |
| Doppler velocimeter | 3 beams 25° off axis (**model choice**), 20 Hz, 2 m to 5 km, σ_v = 1.4 mm/s + 1.1e-6·range, σ_r = 0.16 m + 1.2e-4·range | NDL, IM-1 flight calibration (Karlgaard et al. 2025) |
| Propellant gauge | 3.5 % of the load at start, 1 Hz (rate **model choice**) | Apollo 10 DPS evaluation (NASA CR-101869) |
| Probe switches | contact light, every step | [[lander-preset-apollo-lm]] |
| Flash LiDAR (`lidar_cheap`, only in `sensors=terminal_suite_with_lidar`) | 32 × 32 of the 128 × 128 pixels (**model choice**), 6° field of view, 10 Hz, 2.8 m to 1 km, σ = 0.10 m + 1 % of range, boresight 30° off −z toward +x (**model choice**) | ASC flash LiDAR of OSIRIS-REx (Sornsin et al., ASC); ALHAT/Morpheus flash LiDAR (8 cm, but 150 m minimum range) as the alternative |

## Filter

Error state [dp, dv, dθ, accelerometer bias, gyro bias]; process noise from the IMU numbers;
bias random walks 1e-5 m/s²/√s and 0.01°/h/√s (**model choice**). Initial sigma 1 m, 0.2 m/s,
0.1° (design S2 for the first two), IMU bias repeatability for the biases. Onboard map = the
surface's ground lookup (perfect map; map errors are milestone 6). Outputs body rates from the
gyro minus bias, mass from the gauge, height from position against the map, contact light from
the switches.

## Results (2026-11-15, RTX 5060)

- Synthetic flight (CPU test, 200 landers, 15 s): mean NEES 9.44 (ideal 9); at 15 s, 95 % of
  landers within 1.1 cm vertical and 3.7 mm/s.
- **Recorded truth in S2 (GPU gate test, 256 landers, ~300,000 samples): mean NEES 9.78, median
  8.97**; at first contact 95 % within 0.9 cm vertical and 3.2 mm/s. The filter never reads truth.
- ZEM/ZEV on the estimate (S2, 1,024 landings, seed 0): **97.46 %** against 97.66 % on truth
  (S1, same seed and starts). Navigation error at touchdown: horizontal position 1.25 m mean
  (2.48 m p95) — the first-estimate error, which no S2 sensor observes; vertical 3.6 mm;
  velocity about 1 mm/s.

- S2 on the Apollo 17 terrain (`s2_apollo17_ekf`, 1,024 landings, seed 0): ZEM/ZEV **96.00 %** on
  the estimate against 96.58 % on truth, same 6 body strikes.

## Speed (RTX 5060, `zem_zev` in the loop, env steps per second)

| Scenario | 1,024 envs | 4,096 envs |
| --- | --- | --- |
| S1 truth | 55,063 | 129,102 |
| S2 sensors + EKF | 40,132 | 98,107 |
| S2 + LiDAR 32 × 32 | 39,177 | 87,072 |

## Visual sensor check (2026-11-15, user request: "see a free fall lander and how the sensors respond")

`scripts/sensor_check.py` drops four landers from 30 m on Apollo 17 terrain (engine off, no jets,
no control): flattest spot (upright, at rest), crater rim (tilted 10°, drifting), steepest
footprint (turning 3-5°/s), roughest spot (tilted, drifting, turning). `tools/sensor_recording.py`
keeps truth, the noise-free expected reading of every sensor (ray cast from the true pose), the
measured values and the filter. `scripts/plot_sensor_check.py` makes the figures
(`docs/sensor_check/`), `scripts/render_sensor_check.py` the video (`runs/sensor_check/sensor_check.mp4`,
beams drawn to the ground, LiDAR returns as points, live traces).

| Sensor | (measured − expected) / stated σ: mean, standard deviation | Samples |
| --- | --- | --- |
| IMU delta-velocity | −0.009, 1.000 | 6,738 |
| IMU delta-angle | +0.001, 0.990 | 6,738 |
| Radar altimeter range | +0.025, 0.965 | 899 |
| Doppler velocity | +0.011, 0.991 | 2,697 |
| Doppler range | −0.022, 0.986 | 2,697 |
| LiDAR range | +0.000, 0.999 | 458,752 |

Every sensor matches its stated noise (ideal 0 and 1). Seen in the traces: specific force 0 in
free fall, an impact spike near 100 m/s², then +1.62 m/s² at rest; gyros read the start rates;
the contact light comes on at 1.727 m footpad height; the altimeter and Doppler stop at their
minimum ranges (3.05 m, 2 m); filter errors stay inside ±3σ. The horizontal-velocity band of the
filter grows to about ±25 mm/s in the fall because a tilt error and a horizontal velocity look
alike to the Doppler beams at 9 m/s of descent; the real errors stay within ±3 mm/s. Over terrain
with the perfect map the horizontal position error also shrinks (terrain-relative ranging).

## Deviations from the design

- The design writes `sensors: [imu, radar_alt, doppler_vel]`; a slot holds one binding, so
  `terminal_suite` holds the three (plus gauge and probes), each with its own rate, noise, valid
  and stamp.
- Beams are ray-cast on the surface's own ground lookup instead of Isaac Lab's `RayCaster`: the
  same ground the plant and the metrics use, also for flat ground, no extra mesh structure.
- `lidar_cheap` is the optional `lidar` section of the suite (file `terminal_suite_with_lidar.yaml`); S2 does not use it, so S2 runs pay nothing for it. Test: `tests/smoke/test_lidar_in_environment.py`.

- part of: [[slot-loop-architecture]]
- feeds: [[classical-agents]]
- feeds: [[ppo-landing-agent]]
