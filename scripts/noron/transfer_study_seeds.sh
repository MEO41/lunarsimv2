#!/bin/bash
# Analytic-to-PhysX transfer study, test T5 over several seeds (docs/knowledge/analytic-to-physx-transfer-study.md).
#
# Runs the one-seed gate script (train in PhysX and in the analytic simulator, fly each policy
# in both) for each seed in turn; a seed whose policies exist is not trained again. Seeds 1-4
# first (the planned five with seed 0), then 5-9 for ten in all, so a stop in the middle still
# leaves whole seeds. One job at a time on the GPU, so the wall-clock times of T7 stay clean.
#
# Runs on noron, only on DISK02. From the repository root:
#   setsid nohup bash scripts/noron/transfer_study_seeds.sh > runs/_jobs/transfer_study_seeds.log 2>&1 < /dev/null &
# Other seeds: SEEDS="10 11" bash scripts/noron/transfer_study_seeds.sh
set -u
REPOSITORY=/media/noron/DISK02/Speed_Versus_Fidelity/lunarsimv2
cd "$REPOSITORY" || exit 1
SEEDS=${SEEDS:-"1 2 3 4 5 6 7 8 9"}
echo "[seeds] commit $(git rev-parse --short HEAD), seeds $SEEDS, start $(date -Is)"
for seed in $SEEDS; do
  SEED=$seed bash scripts/noron/transfer_study_gate.sh > "runs/_jobs/transfer_study_gate_seed$seed.log" 2>&1
  echo "[seeds] seed $seed done $(date -Is): $(grep -h '\[lunarsim\] episodes' runs/_jobs/transfer_eval_*_seed$seed.log 2>/dev/null | sed 's/.*success rate: \([0-9.]*\).*/\1/' | tr '\n' ' ')"
  echo "[seeds]   free space on DISK02: $(df -h --output=avail /media/noron/DISK02 | tail -1)"
done
echo "[seeds] done $(date -Is)"
