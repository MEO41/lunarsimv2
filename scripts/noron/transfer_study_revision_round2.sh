#!/bin/bash
# Analytic-to-PhysX transfer study, experiments for the second revision of the paper (simulated
# Acta Astronautica review, 2026-11-26; decision package and goals in
# paper/acta_astronautica/paper/ars_review_round2/). The user asked to set goals from the review
# and repeat until the reviewers are satisfied ("reviewerlar mutlu olana kadar tekrar eder misin").
# The reading rules were written into revizyon_hedefleri.md (goal H3, H5) before these runs.
#
#   R8 In the zero-rate control (5z) the zero rate reaches the policy input, the body-rate reward
#      term, the critic and the tilt-rate check at contact together, so it cannot tell which of
#      them the lagged point mass needs. This control keeps the physical rate everywhere except
#      the policy input:
#        t10_b5_rate_hidden_from_policy   b5 + agent.zero_body_rate_in_policy_input=true
#   R9 The observation holds no command history, so with a lag the commanded direction is hidden
#      state. (5z) plus the previous action in the policy input:
#        t10_b5_zero_rate_last_action     b5_zero_rate + agent.last_action_in_policy_input=true
#   S24 Numerical sensitivity of the PhysX reference: the final policies of both training routes
#      (T5, seeds 0-9) flown again in PhysX with a 2.5 ms step (8 substeps per control step) and
#      32 solver position iterations instead of 5 ms and 16 (evaluation only).
#        t10_refined_physx_{analytic,physx}_policy_seed<n>
# Same training (600 iterations, 4,096 envs) and PhysX evaluation (4,096 held-out starts of seed
# 1000) as T8 and T9, seeds 0-4. A policy trained with an input option is flown with the same option.
#
# Runs on noron, only on DISK02. From the repository root:
#   setsid nohup bash scripts/noron/transfer_study_revision_round2.sh > runs/_jobs/transfer_study_revision_round2.log 2>&1 < /dev/null &
set -u
REPOSITORY=/media/noron/DISK02/Speed_Versus_Fidelity/lunarsimv2
cd "$REPOSITORY" || exit 1
source /media/noron/DISK02/isaac-lab-native/env_isaaclab/bin/activate
export PYTHONPATH=$REPOSITORY/src OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
clean_shm() { for f in /dev/shm/sem.carb-RStringInternals-* /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] || continue; pid=${f##*-}; kill -0 "$pid" 2>/dev/null || rm -f "$f"; done; }

LAGGED_POINT_MASS="plant=analytic_point_mass metrics.end_at_first_contact=true plant.attitude_time_constant_s=1.3"
declare -A TRAINING_OVERRIDES=(
  [b5_rate_hidden_from_policy]="$LAGGED_POINT_MASS agent.zero_body_rate_in_policy_input=true"
  [b5_zero_rate_last_action]="$LAGGED_POINT_MASS plant.body_rate_report=zero agent.last_action_in_policy_input=true"
)
# The policy input options again for the PhysX flight. The zero-rate policy saw a zero rate in
# training, so it is flown with a zero rate input (as the matched-input evaluation of round 1).
declare -A EVALUATION_OVERRIDES=(
  [b5_rate_hidden_from_policy]="agent.zero_body_rate_in_policy_input=true"
  [b5_zero_rate_last_action]="agent.last_action_in_policy_input=true agent.zero_body_rate_in_policy_input=true"
)
TRAIN="python scripts/train_ppo.py scenario=s1_terminal_truth num_envs=4096 agent.ppo.save_interval=25"
EVAL="python scripts/evaluate_agent.py scenario=s1_terminal_truth agent=ppo_mlp num_envs=4096 seed=1000 episodes_per_env=1 logged_envs=256"
REFINED_PHYSX="env.physics_step_s=0.0025 env.physics_steps_per_control_step=8 plant.solver_position_iterations=32"
success_of() { grep '\[lunarsim\] episodes' "$1" | sed 's/.*success rate: \([0-9.]*\).*/\1/'; }

echo "[revision2] commit $(git rev-parse --short HEAD), start $(date -Is)"

# ---- R8 and R9: trainings, then PhysX evaluation ----
for control in b5_rate_hidden_from_policy b5_zero_rate_last_action; do
  for seed in 0 1 2 3 4; do
    name=t10_${control}_seed$seed
    if [ ! -f "runs/$name/policy.pt" ]; then
      $TRAIN seed=$seed ${TRAINING_OVERRIDES[$control]} run_name="$name" > "runs/_jobs/train_$name.log" 2>&1
      grep "\[lunarsim\] trained" "runs/_jobs/train_$name.log" || { echo "[revision2] WARNING: $name did not finish (runs/_jobs/train_$name.log)"; continue; }
    fi
    run=t10_eval_${control}_policy_in_physx_seed$seed
    if [ ! -f "runs/$run/episodes.parquet" ]; then
      clean_shm
      $EVAL agent.policy_path="runs/$name/policy.pt" ${EVALUATION_OVERRIDES[$control]} run_name="$run" > "runs/_jobs/$run.log" 2>&1
    fi
    echo "[revision2] $name in PhysX: $(success_of runs/_jobs/$run.log)  (training success $(grep -o 'success_rate: [0-9.]*' "runs/_jobs/train_$name.log" | tail -1 | cut -d' ' -f2))"
  done
done

# ---- S24: final policies of both routes in a refined PhysX (evaluation only) ----
for seed in 0 1 2 3 4 5 6 7 8 9; do
  for trained_in in analytic physx; do
    run=t10_refined_physx_${trained_in}_policy_seed$seed
    if [ ! -f "runs/$run/episodes.parquet" ]; then
      clean_shm
      $EVAL agent.policy_path="runs/transfer_ppo_${trained_in}_seed$seed/policy.pt" $REFINED_PHYSX run_name="$run" > "runs/_jobs/$run.log" 2>&1
    fi
    echo "[revision2] $trained_in policy seed $seed in refined PhysX: $(success_of runs/_jobs/$run.log)"
  done
done
echo "[revision2] done $(date -Is)"
