---
id: open-issues
title: Open issues
type: issue
status: open
updated: 2026-11-14
summary: Known weak points of the model and the code, with what each one affects.
---

# Open issues

| Issue | Affects | Next step |
| --- | --- | --- |
| Linear strut spring along body z; real struts are inclined honeycomb crushers. Peak leg forces 19–71 kN in successful landings, above the real crush loads (20 and 42 kN) | Leg loads; not touchdown speeds or tilt | Crush model later (milestone 5 footpads); see [[lander-articulation-model]] |
| Sphere footpads, one-cylinder body, no nozzle | Contact on uneven ground, rock strikes | Planned in [[terrain-integration-plan]] |
| RCS cluster height marked ASSUMPTION in the preset | Cross-coupling of horizontal jets | MIT R-567 confirms it; update the source line ([[apollo-lm-geometry-research]]) |
| Inertia about y: 13,667 or 13,867 slug ft² | Pitch dynamics, about 1.5 % | Read the Mission Report page again ([[pending-user-decisions]]) |
| Fixed throttle point (92.5 %) and gimbal not modelled; engine off below 5 % is a model choice | Engine realism | Later actuator bindings |
| RCS propellant not limited | Long hovers | Add a budget |
| Mass in PhysX updated once per control step; inertia scaled with mass | Small dynamics error | Accept for now |
| 48 unexplained failures (0.39 %) outside the low-and-fast group | Agent quality | Look at their trajectories |
| Landing miss up to 43 m when the guided phase is cut short | Precision landing | Agent work |
| No glide-slope constraint in `zem_zev` | Safety on terrain | Optional metric offered; not requested |
| Start mass = Apollo 11 touchdown mass with remaining propellant | Starts with more propellant need a sourced number | Find a source when needed |
| About 10 µm strut vibration, up to 5 mm/s velocity noise at rest | IMU model in M2 | Keep in mind |
| `surface: procedural` smoke-tested only | Terrain studies | Replaced by the terrain plan |
| Isaac Lab 3.0 is a development branch; `isaaclab.sh` is deprecated | Upgrades, setup script | Pinned commit; [[isaac-lab-quirks]] |
| `setup_workstation.sh` never run on a clean machine; it uses `--install core`, a guess | First setup on the RTX 5090 | Fix when it fails ([[workstation-setup-and-machines]]) |
| Simulator process sometimes does not exit after tests | Test runs | Handled by `scripts/run_isaac_tests.py` |

Results these issues belong to: [[milestone-1a-results]].
