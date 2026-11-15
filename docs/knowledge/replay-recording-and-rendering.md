---
id: replay-recording-and-rendering
title: Replay recording and rendering
type: component
status: current
updated: 2026-11-15
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

Users: `scripts/record_landing_videos.py` (landing successes and failures),
`scripts/run_drop_test.py` (`flips.npz`), `scripts/contact_check.py` (`contacts.npz`).

Why a physics-free stage: in a live Isaac Lab scene the renderer drew the lander at its first
spawn point, not at the pose written to PhysX (see [[isaac-lab-quirks]]); and on the 8 GB GPU
the PhysX scene and the RTX renderer together ran out of memory.

- used by: [[contact-check-results]]
- used by: [[drop-test-results]]
