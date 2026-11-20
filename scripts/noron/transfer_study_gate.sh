#!/bin/bash
# Analytic-to-PhysX transfer study, one-seed gate of test T5 (docs/knowledge/analytic-to-physx-transfer-study.md).
#
# Trains ppo_mlp once in PhysX and once in the analytic simulator (seed 0, 4,096 envs, the agent
# file's 600 iterations, a checkpoint every 25 iterations for the time-to-success curve of T7),
# then flies each exported policy in both simulators on the 4,096 held-out starts of seed 1000
# and prints the paired comparison of each cross-test.
#
# Runs on noron, only on DISK02 (user rule). From the repository root:
#   setsid nohup bash scripts/noron/transfer_study_gate.sh > runs/_jobs/transfer_study_gate.log 2>&1 < /dev/null &
#
# The jobs run one after the other, never side by side: two jobs on one GPU would slow each
# other and spoil the wall-clock times that test T7 compares.
set -u
REPOSITORY=/media/noron/DISK02/Speed_Versus_Fidelity/lunarsimv2
cd "$REPOSITORY" || exit 1
source /media/noron/DISK02/isaac-lab-native/env_isaaclab/bin/activate
# The Isaac environment has an editable lunarsim install of another clone; this clone's code wins.
export PYTHONPATH=$REPOSITORY/src OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
clean_shm() { for f in /dev/shm/sem.carb-RStringInternals-* /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] || continue; pid=${f##*-}; kill -0 "$pid" 2>/dev/null || rm -f "$f"; done; }
SEED=${SEED:-0}
TRAIN="python scripts/train_ppo.py scenario=s1_terminal_truth num_envs=4096 seed=$SEED agent.ppo.save_interval=25"
EVAL="python scripts/evaluate_agent.py scenario=s1_terminal_truth agent=ppo_mlp num_envs=4096 seed=1000 episodes_per_env=1 logged_envs=256"
ANALYTIC="plant=analytic_rigid_body"
echo "[gate] commit $(git rev-parse --short HEAD), seed $SEED, $(date -Is)"

for simulator in physx analytic; do
  name=transfer_ppo_${simulator}_seed$SEED
  if [ -f "runs/$name/policy.pt" ]; then echo "[gate] $name exists, not trained again"; continue; fi
  extra=""; [ "$simulator" = analytic ] && extra=$ANALYTIC
  clean_shm
  echo "[gate] training $name $(date -Is)"
  $TRAIN $extra run_name="$name" > "runs/_jobs/train_$name.log" 2>&1
  grep "\[lunarsim\] trained" "runs/_jobs/train_$name.log" || echo "[gate] WARNING: $name did not finish (runs/_jobs/train_$name.log)"
done

for trained_in in physx analytic; do
  policy=runs/transfer_ppo_${trained_in}_seed$SEED/policy.pt
  [ -f "$policy" ] || continue
  for flown_in in physx analytic; do
    run=transfer_eval_${trained_in}_policy_in_${flown_in}_seed$SEED
    extra=""; [ "$flown_in" = analytic ] && extra=$ANALYTIC
    clean_shm
    $EVAL $extra agent.policy_path="$policy" run_name="$run" > "runs/_jobs/$run.log" 2>&1
    echo "[gate] $run: $(grep '\[lunarsim\] episodes' "runs/_jobs/$run.log")"
  done
  echo "[gate] policy trained in $trained_in, flown in both simulators:"
  python -m lunarsim.tools.command_line compare-simulators \
    "runs/transfer_eval_${trained_in}_policy_in_analytic_seed$SEED" "runs/transfer_eval_${trained_in}_policy_in_physx_seed$SEED"
done
echo "[gate] done $(date -Is)"
