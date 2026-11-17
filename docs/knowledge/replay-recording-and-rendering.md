---
id: replay-recording-and-rendering
title: Replay recording and rendering
type: component
status: current
updated: 2026-11-17
summary: How runs are recorded frame by frame and rendered later as videos or stills, on a physics-free stage, with footpad force arrows.
---

# Replay recording and rendering

Two steps in two processes, because the simulation runs fastest without a camera and the
renderer needs the camera extension:

1. **Record** (`src/lunarsim/tools/replay_recording.py`, `ReplayRecorder`): passed as
   `after_step` to `collect_episodes`. Keeps every environment's first episode at 25 frames per
   second: world pose, strut strokes, speeds, height above ground, tilt, time, body force, and
   with `include_contacts` per leg the world position, the contact-sensor force vector and the
   gap of the dish to the ground. `save()` writes the chosen environments to an `.npz` with the
   scenario name and overrides (to rebuild the same terrain), titles and caption lines.
2. **Render** (`scripts/render_replay.py replay=... mode=video|stills`, using
   `src/lunarsim/tools/replay_rendering.py`): builds a plain USD stage without physics (terrain
   from the same scenario and seed, low sun from the east, faint fill light, one referenced
   lander per clip), sets the transforms from the recording, draws a green force arrow beside
   each loaded footpad, follows the lander with the camera, writes captions with PIL, and joins
   the frames with ffmpeg.

Mission video (S7, 2026-11-17): `scripts/run_mission.py ... record=<lander>` saves that lander's
trajectory in every phase (`src/lunarsim/tools/mission_trajectory_recording.py`,
`trajectory_lander<i>.npz`) and the PhysX phase's `replay.npz`. `scripts/render_mission_video.py`
draws the descent orbit (100x) and the powered descent (braking 15x, approach 3x) as plots
(`src/lunarsim/tools/mission_video.py`; speeds and thrust axis against the local vertical, not the
site axes, which are tilted far from the site), renders the terminal phase in 3D with
`render_replay.py` (real time; `terminal=reuse` keeps an earlier render) and joins title cards,
plots, 3D and the end card with ffmpeg: `runs/<run>_video/mission.mp4` (3 min 46 s for
`s7_full_mission_video_seed0`, lander 0).

Users: `scripts/record_landing_videos.py` (landing successes and failures),
`scripts/run_drop_test.py` (`flips.npz`), `scripts/contact_check.py` (`contacts.npz`).

Why a physics-free stage: in a live Isaac Lab scene the renderer drew the lander at its first
spawn point, not at the pose written to PhysX (see [[isaac-lab-quirks]]); and on the 8 GB GPU
the PhysX scene and the RTX renderer together ran out of memory.

- used by: [[contact-check-results]]
- used by: [[drop-test-results]]
