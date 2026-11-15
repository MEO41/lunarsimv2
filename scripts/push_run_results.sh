#!/bin/bash
# Commit and push the results of every finished run (user rule 2026-11-15: after every run the
# results go to GitHub, so the commits show which change is responsible for which result).
#
# Usage, from the repository root, on any machine:   bash scripts/push_run_results.sh
# Repeating watcher (minsky):  setsid nohup bash scripts/push_run_results.sh watch > runs/push_run_results.log 2>&1 &
#
# The commit is made in a separate sparse clone that holds only results/ (RESULTS_CLONE, default
# ~/lunarsim-results-clone), never in this working copy: a queue running here keeps its code,
# and nothing is pulled under it. scripts/archive_run_results.py copies each finished run of
# runs/ that the clone does not have yet; then the index is rebuilt, committed and pushed. When
# another machine pushed first, the clone resets to the remote, rebuilds the index and retries.
# Python must have pandas, pyyaml and tensorboard (the Isaac environment has them).
REPOSITORY=$(cd "$(dirname "$0")/.." && pwd)
RESULTS_CLONE=${RESULTS_CLONE:-$HOME/lunarsim-results-clone}
REMOTE=https://github.com/MEO41/lunarsimv2.git
GIT_IDENTITY=(-c user.name=MEO41 -c user.email=MEO41@users.noreply.github.com)

push_once() {
  # Archive new runs into the clone, commit them and push; retry when the remote moved.
  if [ ! -d "$RESULTS_CLONE/.git" ]; then
    git clone -q --filter=blob:none --sparse "$REMOTE" "$RESULTS_CLONE" || return 1
    git -C "$RESULTS_CLONE" sparse-checkout set results || return 1
  fi
  git -C "$RESULTS_CLONE" pull -q --rebase || { git -C "$RESULTS_CLONE" rebase --abort 2>/dev/null; git -C "$RESULTS_CLONE" fetch -q && git -C "$RESULTS_CLONE" reset -q --mixed origin/main; }
  (cd "$REPOSITORY" && python scripts/archive_run_results.py results="$RESULTS_CLONE/results") || return 1
  for attempt in 1 2 3 4 5; do
    (cd "$RESULTS_CLONE" && python "$REPOSITORY/scripts/archive_run_results.py" index_only=true >/dev/null) || return 1
    git -C "$RESULTS_CLONE" add results
    if git -C "$RESULTS_CLONE" diff --cached --quiet; then return 0; fi
    names=$(git -C "$RESULTS_CLONE" diff --cached --name-only --diff-filter=A -- 'results/*/summary.yaml' | cut -d/ -f2 | paste -sd ' ')
    git -C "$RESULTS_CLONE" "${GIT_IDENTITY[@]}" commit -q -m "Results from $(hostname): ${names:-index}

Archived by scripts/push_run_results.sh; each run's manifest.yaml names the code commit it ran on.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01LFNHCUnC3t9o5rUk7TyebF" || return 1
    if git -C "$RESULTS_CLONE" push -q origin HEAD:main; then echo "[results] pushed: ${names:-index} $(date)"; return 0; fi
    # Another machine pushed first: keep our files, take the remote history, rebuild and retry.
    git -C "$RESULTS_CLONE" fetch -q && git -C "$RESULTS_CLONE" reset -q --mixed origin/main
    sleep $((attempt * 5))
  done
  echo "[results] push failed after 5 attempts $(date)"; return 1
}

if [ "$1" = watch ]; then
  while true; do push_once; sleep 300; done
else
  push_once
fi
