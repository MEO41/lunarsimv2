---
id: rules-from-the-user
title: Rules from the user
type: rule
status: current
updated: 2026-11-14
summary: Standing instructions for all work in this repository; read first in every session.
---

# Rules from the user

These are the user's own instructions. They hold until the user changes them.

## Code and repository (CLAUDE.md, checked into the repo)

1. **No cryptic abbreviations** in variables, classes or files (not `M3`, `S2`, `HR7`). Use fully
   descriptive names. Registry keys and scenario file names from the design (`zem_zev`,
   `physx_site`, `s1_terminal_truth.yaml`) were approved as exceptions; Python names stay descriptive.
2. **Plan before coding.** List the exact files and changes, then wait for the user to say
   **"Proceed"**, unless the user has already told you to go ahead with that exact task.
3. **After each file change**, give a two-sentence summary of what changed.
4. **Docstring on every new function.**

## From the opening brief (build from scratch)

- "Write everything new. **Do not use, copy or look up code or numbers from the old LunarSim
  repo** (GulogluHaktan/LunarSim)." This includes every local copy, for example
  `~/Desktop/MEO/LunarSim` and its `LunarSim-wt-*` worktrees, `/media/noron/DISK02/LunarSim`
  and `LunarRocket`.
- "**Every physical constant** (lander mass, inertia, thrust range, Isp, RCS data, lunar gravity
  and radius) **comes from a public source.** Write the source next to the number in the preset file."
  A number with no source must be marked "model choice" or "ASSUMPTION".
- "All env code is **batched torch on the GPU**, with N envs as the first dimension. **No Python
  loop over envs.**"
- **No RL until the user opens milestone 1b.** "No reward functions, no training code, no rsl_rl
  setup." The session stopped at the M1a gate; see [[milestone-roadmap-and-status]].
- "**Ask before any choice that is hard to undo.**"
- "If something in the design does not work in Isaac Lab, **stop and explain. Propose a fix; do
  not change the design silently.** When you explain, **use simple words**."
- "**Small commits with clear messages.** Push them to https://github.com/MEO41/lunarsimv2.git".
  Keep `PROGRESS.md` up to date, and also this knowledge graph ([[knowledge-graph-tooling]]).

## Requests from 2026-11-14 about the terrain work (see [[terrain-integration-plan]])

- Do not use the NASA lander mesh in the simulation (too slow). Keep our own lander model, but
  make the **footpad contact areas, leg geometry and centre of mass match the real Apollo LM**.
- The **ground must not be flat**: real terrain with craters ("holes"), rocks and slopes, so the
  height above the ground is not constant.
- **Height maps are not committed to git.** They are public; users download them into the
  assets folder.
- **Keep the existing lunarsimv2 numbers**, and change lander values only where public LM data
  show a better value, with the source next to each.

## Privacy and attribution

- The user's e-mail address is private. Commits use `MEO41 <MEO41@users.noreply.github.com>`.
  Never send the address to an outside service and never write it into a note.
- Commit messages end with the `Co-Authored-By:` and `Claude-Session:` lines that the session
  gives (the model name in them changes with the model the user picks).

## How the user works

- The user reads every change and follows along on GitHub, sometimes from another device.
- The user switches models and effort levels during a session and moves between machines; that
  is why this knowledge graph exists.
- Design authority: [[project-goal-and-design-documents]]. Decisions taken so far: [[decisions-log]].
