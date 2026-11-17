# SPDX-License-Identifier: BSD-3-Clause
"""Render the S7 full-mission video of one lander: orbit and powered descent as plots, the terminal phase in 3D.

Example (after ``python scripts/run_mission.py mission=s7_full_mission num_envs=16 seed=0
run_name=s7_full_mission_video_seed0 record=0``)::

    python scripts/render_mission_video.py run=s7_full_mission_video_seed0 lander=0

Steps: title card; descent orbit (``orbit_speed``, default 100 times real time); braking (15x) and
approach (3x) as altitude over range (:mod:`lunarsim.tools.mission_video`); the PhysX terminal
phase rendered in 3D on the Apollo 17 terrain by ``scripts/render_replay.py`` (real time); an end
card with the landing result. ffmpeg joins them into ``runs/<run>_video/mission.mp4``.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import numpy
import pandas

from lunarsim.env.scenario_loader import load_scenario
from lunarsim.slots.plant.orbit_float64 import site_from_parameters
from lunarsim.tools.mission_video import FRAMES_PER_SECOND, descent_frames, load_phase, orbit_frames, title_card

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def encode(frames: Path, output: Path) -> None:
    """Join numbered PNG frames into an H.264 MP4 at the video frame rate."""
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(FRAMES_PER_SECOND), "-i", str(frames / "frame_%06d.png"),
         "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", "-r", str(FRAMES_PER_SECOND), str(output)],
        check=True,
    )


def main() -> None:
    """Parse ``key=value`` options, draw the plot parts, render the terminal phase, join everything."""
    options = {"run": "s7_full_mission_video_seed0", "lander": "0", "orbit_speed": "100", "braking_speed": "15", "approach_speed": "3"}
    for argument in sys.argv[1:]:
        key, _, value = argument.partition("=")
        if key not in options:
            raise SystemExit(f"unknown argument '{argument}' ({', '.join(f'{name}=' for name in options)})")
        options[key] = value
    run, lander = options["run"], int(options["lander"])
    runs = REPOSITORY_ROOT / "runs"
    output = runs / f"{run}_video"
    if output.exists():
        shutil.rmtree(output)
    frames = output / "frames"
    frames.mkdir(parents=True)

    braking_log = runs / f"{run}_descent_orbit" / "episodes.parquet"
    scenario = load_scenario("s7_phase_braking", [f"start.log_path={braking_log}"])
    site = site_from_parameters(scenario.slots["plant"].parameters["site"])
    site_direction = site.origin_mcmf().numpy()
    site_direction = site_direction / numpy.linalg.norm(site_direction)
    mass = scenario.lander["mass"]
    dry_mass = float(mass["mass_at_start_kg"]["value"]) - float(mass["descent_propellant_at_start_kg"]["value"]) + float(mass["unusable_descent_propellant_kg"]["value"])
    trajectory = f"trajectory_lander{lander}.npz"
    orbit = load_phase(runs / f"{run}_descent_orbit" / trajectory)
    braking = load_phase(runs / f"{run}_braking" / trajectory)
    approach = load_phase(runs / f"{run}_approach" / trajectory)

    number = title_card(frames, 0, [
        "LunarSim v2: S7 full mission, orbit to touchdown",
        f"One lander of the committed S7 run (seed 0, lander {lander}); Apollo 11 timeline, Apollo 17 site",
        "Descent orbit and powered descent: float64 orbit simulation, drawn as plots",
        "Terminal phase: PhysX on the Apollo 17 height map, rendered in 3D",
    ], 4.0)
    number = orbit_frames(frames, number, orbit, site_direction, site.radius_m, float(options["orbit_speed"]), dry_mass)
    number = title_card(frames, number, ["2  Powered descent", "PDI at the 15 km perilune; braking to the high gate, approach to the low gate"], 2.0)
    number = descent_frames(frames, number, [("BRAKING", braking, float(options["braking_speed"])), ("APPROACH", approach, float(options["approach_speed"]))], site_direction, site.radius_m, dry_mass)
    number = title_card(frames, number, ["3  Terminal phase in PhysX", "Handoff at the low gate (float64 orbit frame to the site frame), hazard map, site choice, touchdown", "real time"], 3.0)
    encode(frames, output / "part_1_plots.mp4")
    shutil.rmtree(frames)
    print(f"[video] plots: {number} frames", flush=True)

    replay = runs / f"{run}_terminal" / "replay.npz"
    subprocess.run([sys.executable, str(REPOSITORY_ROOT / "scripts" / "render_replay.py"), f"replay={replay}", "mode=video"], check=True)
    shutil.copy2(replay.with_suffix(".mp4"), output / "part_2_terminal.mp4")

    episodes = pandas.read_parquet(runs / f"{run}_terminal" / "episodes.parquet")
    row = episodes[episodes["environment_index"] == lander].iloc[0]
    summary = pandas.read_parquet(runs / f"{run}_approach" / "episodes.parquet")
    total_time = float(approach["time"][-1]) + float(row["episode_duration_s"])
    frames.mkdir()
    title_card(frames, 0, [
        f"Touchdown: {str(row['outcome']).upper()}",
        f"vertical {row['touchdown_vertical_speed_mps']:+.2f} m/s, horizontal {row['touchdown_horizontal_speed_mps']:.2f} m/s, "
        f"{float(numpy.hypot(row['touchdown_position_x_m'], row['touchdown_position_y_m'])):.1f} m from the site",
        f"mission time {total_time / 60:.1f} min from the DOI point; all {int(summary['gate_reached'].sum())} landers of the run reached every gate",
        "Source of the Apollo numbers: NASA TN D-6846 (Bennett, 1972)",
    ], 5.0)
    encode(frames, output / "part_3_end.mp4")
    shutil.rmtree(frames)

    parts = [output / "part_1_plots.mp4", output / "part_2_terminal.mp4", output / "part_3_end.mp4"]
    inputs = sum((["-i", str(part)] for part in parts), [])
    streams = "".join(f"[{index}:v]scale=1280:720,setsar=1,fps={FRAMES_PER_SECOND}[v{index}];" for index in range(len(parts)))
    joined = "".join(f"[v{index}]" for index in range(len(parts)))
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", *inputs, "-filter_complex", f"{streams}{joined}concat=n={len(parts)}:v=1:a=0[out]",
         "-map", "[out]", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", str(output / "mission.mp4")],
        check=True,
    )
    print(f"[video] wrote {output / 'mission.mp4'}", flush=True)


if __name__ == "__main__":
    main()
