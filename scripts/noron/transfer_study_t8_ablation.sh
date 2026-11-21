#!/bin/bash
# Analytic-to-PhysX transfer study, test T8: model ablation (docs/knowledge/analytic-to-physx-transfer-study.md).
#
# PPO is trained in ever simpler analytic models and every policy is flown in full PhysX with the
# standard success rule, on the 4,096 held-out starts of seed 1000. Step 0 is the T5 result
# (spring legs, standard actuators).
#   b1_contact      spring-leg rigid body, episode ends at the first contact
#   b2_ideal        b1 + ideal actuators (no thrust lag, slew limit, tail-off, jet minimum pulse or valve delay)
#   b3_point_mass   point mass, thrust along the commanded direction at once, ideal actuators, ends at contact
# Seeds 0-4. It waits for the T7 queue, so no two jobs share the GPU.
#
# Runs on noron, only on DISK02. From the repository root:
#   setsid nohup bash scripts/noron/transfer_study_t8_ablation.sh > runs/_jobs/transfer_study_t8.log 2>&1 < /dev/null &
set -u
REPOSITORY=/media/noron/DISK02/Speed_Versus_Fidelity/lunarsimv2
cd "$REPOSITORY" || exit 1
source /media/noron/DISK02/isaac-lab-native/env_isaaclab/bin/activate
export PYTHONPATH=$REPOSITORY/src OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
clean_shm() { for f in /dev/shm/sem.carb-RStringInternals-* /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] || continue; pid=${f##*-}; kill -0 "$pid" 2>/dev/null || rm -f "$f"; done; }
declare -A STEP_OVERRIDES=(
  [b1_contact]="plant=analytic_rigid_body metrics.end_at_first_contact=true"
  [b2_ideal]="plant=analytic_rigid_body metrics.end_at_first_contact=true actuators.ideal_response=true"
  [b3_point_mass]="plant=analytic_point_mass metrics.end_at_first_contact=true actuators.ideal_response=true"
)
TRAIN="python scripts/train_ppo.py scenario=s1_terminal_truth num_envs=4096 agent.ppo.save_interval=25"
EVAL="python scripts/evaluate_agent.py scenario=s1_terminal_truth agent=ppo_mlp num_envs=4096 seed=1000 episodes_per_env=1 logged_envs=256"

while pgrep -f "^bash scripts/noron/transfer_study_t7_checkpoints\.sh" > /dev/null; do sleep 60; done
echo "[t8] commit $(git rev-parse --short HEAD), start $(date -Is)"
for seed in 0 1 2 3 4; do
  for step in b1_contact b2_ideal b3_point_mass; do
    name=t8_${step}_seed$seed
    if [ ! -f "runs/$name/policy.pt" ]; then
      $TRAIN seed=$seed ${STEP_OVERRIDES[$step]} run_name="$name" > "runs/_jobs/train_$name.log" 2>&1
      grep "\[lunarsim\] trained" "runs/_jobs/train_$name.log" || { echo "[t8] WARNING: $name did not finish (runs/_jobs/train_$name.log)"; continue; }
    fi
    run=t8_eval_${step}_policy_in_physx_seed$seed
    if [ ! -f "runs/$run/episodes.parquet" ]; then
      clean_shm
      $EVAL agent.policy_path="runs/$name/policy.pt" run_name="$run" > "runs/_jobs/$run.log" 2>&1
    fi
    echo "[t8] $name in PhysX: $(grep '\[lunarsim\] episodes' "runs/_jobs/$run.log" | sed 's/.*success rate: \([0-9.]*\).*/\1/')  (training success $(grep -o 'success_rate: [0-9.]*' "runs/_jobs/train_$name.log" | tail -1 | cut -d' ' -f2))"
  done
done
echo "[t8] done $(date -Is)"
