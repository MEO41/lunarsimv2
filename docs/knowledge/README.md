# LunarSim v2 knowledge graph

The project memory lives here, in git, so it moves with the repository to every machine and to
every new Claude Code session. Local Claude memory files (`~/.claude/...`) do **not** travel;
this folder does.

## Start of a session: read in this order

1. [[rules-from-the-user]]: what must and must not be done. Read this first, every time.
2. [[pending-user-decisions]]: what is waiting for the user's answer. Do not start this work without it.
3. [[milestone-roadmap-and-status]]: where the project stands.
4. [[workstation-setup-and-machines]]: which machine, which Python environment, how to run things.
5. Then the notes the task needs: [[slot-loop-architecture]], [[measured-terrain-surface]], ...

[[project-goal-and-design-documents]] links to the design documents, which are the authority.
`PROGRESS.md` and `M1A_REPORT.md` at the repository root keep the detailed records; these
notes summarise them and point to them.

## All notes by type

| Type | Notes |
| --- | --- |
| rule | [[rules-from-the-user]] |
| design | [[project-goal-and-design-documents]] |
| milestone | [[milestone-roadmap-and-status]] |
| component | [[slot-loop-architecture]], [[analytic-simulator]], [[bindings-catalogue]], [[measured-terrain-surface]], [[lander-preset-apollo-lm]], [[lander-articulation-model]], [[physx-contact-settings]], [[classical-agents]], [[hazard-detection-and-avoidance]], [[orbit-to-touchdown-mission]], [[software-in-the-loop-bridge]], [[plume-regolith-and-dust]], [[sensors-and-ekf]], [[faults-and-detection]], [[ppo-landing-agent]], [[metrics-and-run-records]], [[replay-recording-and-rendering]], [[knowledge-graph-tooling]] |
| decision | [[decisions-log]] |
| result | [[milestone-1a-results]], [[milestone-1b-results]], [[paper-one-results]], [[terrain-results]], [[drop-test-results]], [[contact-check-results]], [[speed-benchmarks]] |
| issue | [[open-issues]] |
| plan | [[analytic-to-physx-transfer-study]], [[terrain-integration-plan]] (done), [[pending-user-decisions]], [[skipped-learning-work]] |
| research | [[apollo-lm-geometry-research]], [[lunar-surface-statistics-research]], [[apollo17-height-maps]], [[apollo-lander-modeling-folder]], [[sources-bibliography]] |
| environment | [[workstation-setup-and-machines]], [[isaac-lab-quirks]], [[testing-and-commands]] |
| history | [[session-history]] |

A diagram of all named relations is in [graph_overview.md](graph_overview.md); every node and
edge is in [graph.json](graph.json).

## Keeping it current (part of every step)

- When a decision is taken, a result is measured, or the user gives a new rule, update the
  matching note **in the same commit** as the code. Change `updated:` in its header.
- Link notes with `[[note-id]]`. Give a link a relation by writing it as a list line:
  `- depends on: [[slot-loop-architecture]]`.
- Then run `lunarsim knowledge-graph` (rebuilds `graph.json` and `graph_overview.md`) and
  `lunarsim knowledge-graph --check`. A CPU unit test runs the same check, so broken links fail CI.
- Never put secrets or the user's e-mail address in a note.
