---
id: open-issues
title: Open issues
type: issue
status: open
updated: 2026-11-15
summary: Known weak points of the model and the code, with what each one affects.
---

# Open issues

| Issue | Affects | Next step |
| --- | --- | --- |
| Linear strut spring along body z; real struts are inclined honeycomb crushers. Peak leg forces 19–71 kN in successful landings, above the real crush loads (20 and 42 kN) | Leg loads; not touchdown speeds or tilt | Crush model later (milestone 5 footpads); see [[lander-articulation-model]] |
| Dish footpad is fixed to its leg (real one swivels on a ball joint); stages are a cylinder and a box | Contact on rough ground | Accept for now |
| Nozzle has no collision shape (crushable skirt); a rock under it is recorded as low clearance, not as a force | Rock strikes under the engine | `minimum_nozzle_clearance_m` in the records |
| Rocks are domes in the ground mesh; craters use a model-choice rim profile; no regolith sinkage | Terrain realism | Bekker footpads in milestone 5 |
| Height above ground is taken straight below the c.g.; under a rough footprint the pads can be up to about 1 m higher or lower | Agents' cut-off height on rough ground | Altimeter beams (milestone 2) |
| `procedural` surface has no height lookup | Its height above ground is from z = 0 | Use `measured_terrain` |
| Fixed throttle point (92.5 %) and gimbal not modelled; engine off below 5 % is a model choice | Engine realism | Later actuator bindings |
| RCS propellant not limited | Long hovers | Add a budget |
| Mass in PhysX updated once per control step; inertia scaled with mass | Small dynamics error | Accept for now |
| Landers slide on slopes above about 15° after an impact (friction 0.4 cannot hold above 21.8°); real footpads dig in | Drop tests and landings on steep ground | Regolith/footpad model (milestone 5) ([[drop-test-results]]) |
| 27 body strikes on rocks in 12,288 terrain landings; ZEM/ZEV does not look for a safe spot | Terrain success | Hazard maps and site selection (milestone 4) |
| Probe-mode cut-off (since 2026-11-15) drops the lander about 1 m with the engine off; a rock under a probe leg still switches the light on early. Body strikes on tall rocks under the stage cannot be avoided by any cut-off rule | Body strikes, touchdown speed | Hazard maps and site selection (milestone 4) |
| Thrust tail-off modelled with the 0.5 s design value; the flown engine took several seconds (Rogers 1973, no number) | Touchdown speed, toppling | Find a measured tail-off curve |
| Probes have no shape and never bend; the light is the probe tip against the ground lookup | Contact light timing on rocks | Accept |
| Landing miss up to 43 m when the guided phase is cut short | Precision landing | Agent work |
| No glide-slope constraint in `zem_zev` | Safety on terrain | Optional metric offered; not requested |
| Start mass = Apollo 11 touchdown mass with remaining propellant | Starts with more propellant need a sourced number | Find a source when needed |
| About 10 µm strut vibration, up to 5 mm/s velocity noise at rest | IMU model in M2 | Keep in mind |
| `surface: procedural` smoke-tested only | Terrain studies | Replaced by the terrain plan |
| Whole-environment speed on the development desktop varies by up to ±30 % between runs | Speed tables | RTX 5090 table measured 2026-11-15 ([[workstation-setup-and-machines]]); report filter-only times for filter changes |
| Isaac Lab 3.0 is a development branch; `isaaclab.sh` is deprecated | Upgrades, setup script | Pinned commit; [[isaac-lab-quirks]] |
| `setup_workstation.sh` never run on a clean machine; it uses `--install core`, a guess | First setup on the RTX 5090 | Fix when it fails ([[workstation-setup-and-machines]]) |
| Simulator process sometimes does not exit after tests, or stalls at launch | Test runs | Handled by `scripts/run_isaac_tests.py` (kill, shm clean-up, restart) |
| Fixed 2026-11-15: RCS torque arms 29 % too short; forces applied 0.1 m above the vehicle c.g.; Iyy text error; footpad friction 0.2 instead of 0.4 | M1a numbers were made with these | Evaluation rerun: the 48 unexplained M1a failures are gone ([[terrain-results]]) |

Results these issues belong to: [[milestone-1a-results]].
