---
id: contact-check-results
title: Footpad contact check on real terrain
type: result
status: current
updated: 2026-11-15
summary: Free-fall drops on eight kinds of ground; at rest every footpad's contact-sensor force agrees with its gap to the terrain lookup (32 of 32).
---

# Footpad contact check on real terrain (2026-11-15)

Asked by the user: use the per-footpad contact sensors at different places and render them, to
see that the surface and footpad contacts are correct. Script `scripts/contact_check.py`,
scenario `configs/scenarios/apollo17_contact_check.yaml`, stills and table in
`docs/contact_check/` (all eight stills in `runs/contact_check/contacts_stills/`, git-ignored).

## Method

- Eight landers fall freely (no agent output, engine off, no jets), upright, heading 0, from 1 m
  above the highest ground under their footprint, onto spots picked from the terrain itself:
  the flattest spot, the steepest footprint (plane fit, 9.8°), the middle of a 9.8 m crater, its
  rim, leg 1 on a 0.89 m rock, a 1.60 m rock under the descent stage, leg 2 in a 2.2 m crater,
  and the roughest spot.
- At rest (2 s still), each footpad's sensor force (the leg's contact sensor) is compared with
  its **gap**: the smallest vertical distance from the dish's lower surface (its mesh points) to
  the ground lookup, which uses the same triangles as PhysX. Rule: a pad with more than 100 N
  must have |gap| < 3 cm (contact offset 2 cm + 1 cm); a pad with no force must not be inside.
- `scripts/render_replay.py mode=stills` renders a wide view and a close-up of each pad, with a
  green arrow along the sensor force (1 m per 3 kN) and the numbers.

## Results

**32 of 32 footpads agree.** Loaded pads have gaps within ±0.4 cm; free pads are 15–108 cm
above the ground. Without body contact the four leg forces add up to the weight (11.90 kN)
within 2.4 %. On the rock under the descent stage, the body carries 10.3 kN and two legs
0.8 kN each, the other two hang free. On the flattest spot the legs carry 1.7 / 4.3 / 1.6 /
4.3 kN: four springs on ground that is not exactly level share the load unevenly (1 cm of height
difference is 0.8 kN of spring force), and the sum is still the weight.

- measured on: [[measured-terrain-surface]]
- see also: [[drop-test-results]]
