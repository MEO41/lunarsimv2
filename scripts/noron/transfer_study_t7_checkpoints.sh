#!/bin/bash
# Analytic-to-PhysX transfer study, test T7 (docs/knowledge/analytic-to-physx-transfer-study.md):
# every training checkpoint of the T5 runs (both simulators, seeds 0-9, every 25 iterations) is
# exported and flown in PhysX on the 4,096 held-out starts of seed 1000. With the checkpoints'
# wall-clock times (checkpoint_wall_clock.csv) this gives "PhysX success against training
# wall-clock time" and the time to 90 % success in PhysX for each training simulator.
#
# Iterations 0-250 run first for every run (90 % is reached near iteration 100), then 275-575,
# so a stop in the middle still leaves the part that decides the time to 90 %. The final
# policy (iteration 600) was flown in T5 already. Jobs run one at a time.
#
# Runs on noron, only on DISK02. From the repository root:
#   setsid nohup bash scripts/noron/transfer_study_t7_checkpoints.sh > runs/_jobs/transfer_study_t7.log 2>&1 < /dev/null &
set -u
REPOSITORY=/media/noron/DISK02/Speed_Versus_Fidelity/lunarsimv2
cd "$REPOSITORY" || exit 1
source /media/noron/DISK02/isaac-lab-native/env_isaaclab/bin/activate
export PYTHONPATH=$REPOSITORY/src OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
clean_shm() { for f in /dev/shm/sem.carb-RStringInternals-* /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] || continue; pid=${f##*-}; kill -0 "$pid" 2>/dev/null || rm -f "$f"; done; }
EVAL="python scripts/evaluate_agent.py scenario=s1_terminal_truth agent=ppo_mlp num_envs=4096 seed=1000 episodes_per_env=1 logged_envs=0"
echo "[t7] commit $(git rev-parse --short HEAD), start $(date -Is)"
for iterations in "$(seq -s ' ' 0 25 250)" "$(seq -s ' ' 275 25 575)"; do
  for iteration in $iterations; do
    for seed in 0 1 2 3 4 5 6 7 8 9; do
      for trained_in in physx analytic; do
        training=runs/transfer_ppo_${trained_in}_seed$seed
        run=t7_${trained_in}_seed${seed}_model$iteration
        [ -f "runs/$run/episodes.parquet" ] && continue
        policy=$training/checkpoint_policies/policy_$iteration.pt
        if [ ! -f "$policy" ]; then
          mkdir -p "$training/checkpoint_policies"
          python scripts/export_policy.py run="$training" checkpoint="model_$iteration.pt" output="$policy" > /dev/null 2>&1 \
            || { echo "[t7] WARNING: export failed for $training model_$iteration.pt"; continue; }
        fi
        clean_shm
        $EVAL agent.policy_path="$policy" run_name="$run" > "runs/_jobs/$run.log" 2>&1
        echo "[t7] $run: $(grep '\[lunarsim\] episodes' "runs/_jobs/$run.log" | sed 's/.*success rate: \([0-9.]*\).*/\1/')"
      done
    done
  done
  echo "[t7] block done $(date -Is)"
done
echo "[t7] done $(date -Is)"
