# PROGRESS

Last update: 2026-11-14

## Done
- Read the design docs and all diagrams; checked the machine; settled the open decisions (see PLAN.md).
- Repo initialised from the Isaac Lab external-project template layout.

## In progress
- Milestone 0: core package.

## Blocked
- Nothing.

## Decisions
| Date | Decision | Why |
| --- | --- | --- |
| 2026-11-14 | Pin Isaac Sim 6.0.1 + Isaac Lab 3.0.0 develop@e86463d, PhysX backend | It is what runs on this machine |
| 2026-11-14 | Forces go through the permanent wrench composer | `set_external_force_and_torque` is deprecated in this Isaac Lab and only forwards to it |
| 2026-11-14 | Plant state is read twice per control step (hook 3 and hook 6) | Metrics must see the state after physics |
| 2026-11-14 | Registry keys as in the design; descriptive names in code | CLAUDE.md rule 1 |
| 2026-11-14 | Architecture doc wins over the Scenario Bindings doc for `Binding` and `TruthState` | User: it is the main authority |
| 2026-11-14 | Package lives in `src/lunarsim` (PLAN.md said `source/lunarsim/lunarsim`) | That is the layout the template in this Isaac Lab version generates |
| 2026-11-14 | Build backend is setuptools, not the template's `uv_build` | `uv` is not installed here |
