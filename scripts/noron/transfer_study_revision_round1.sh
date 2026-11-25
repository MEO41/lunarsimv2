#!/bin/bash
# Analytic-to-PhysX transfer study, experiments for the first revision of the paper (simulated
# Acta Astronautica review, 2026-11-25; decision package in
# paper/acta_astronautica/paper/ars_review_round1/editorial_decision_package.md). The user asked
# to carry out the revision ("revize edilmesi gereken yerleri tamamlar misin", 2026-11-25).
#
#   R20 observation channel. The point mass with the attitude at once (b4) reports a zero body
#       rate, so in training the three rate inputs of the policy are constant and rsl_rl's
#       normaliser divides PhysX rates by (0 + 0.01). Is the 0 % of b4 against 90.7 % of b5 the
#       lag, or the rate channel?
#         t9_eval_b4_masked   b4 policies flown in PhysX with a zero body rate in the policy input
#         t9_b4_held_rate     b4 + body_rate_report=held_turn_rate (attitude at once, with a rate)
#         t9_b5_zero_rate     b5 + body_rate_report=zero (lag of 1.3 s, no rate)
#   R17 agreement check (ZEM/ZEV in the training model against ZEM/ZEV in PhysX) on the ablation
#       models it can fly (b1, b2) and on the unmatched model; the point mass needs a commanded
#       direction from PPO, so ZEM/ZEV cannot fly it.
#   R15 a training model that breaks the solver matching: the full analytic model at a 10 ms
#       physics step (100 Hz) instead of PhysX's 5 ms.
#         t9_unmatched_step10ms
#   R12 tau sensitivity: b5 with tau 0.3, 0.65 and 2.2 s (1.3 s is the measured value).
#         t9_b5_tau0p3, t9_b5_tau0p65, t9_b5_tau2p2
# Same training (600 iterations, 4,096 envs) and PhysX evaluation (4,096 held-out starts of seed
# 1000) as T8, seeds 0-4.
#
# Runs on noron, only on DISK02. From the repository root:
#   setsid nohup bash scripts/noron/transfer_study_revision_round1.sh > runs/_jobs/transfer_study_revision_round1.log 2>&1 < /dev/null &
set -u
REPOSITORY=/media/noron/DISK02/Speed_Versus_Fidelity/lunarsimv2
cd "$REPOSITORY" || exit 1
source /media/noron/DISK02/isaac-lab-native/env_isaaclab/bin/activate
export PYTHONPATH=$REPOSITORY/src OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
clean_shm() { for f in /dev/shm/sem.carb-RStringInternals-* /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] || continue; pid=${f##*-}; kill -0 "$pid" 2>/dev/null || rm -f "$f"; done; }

POINT_MASS="plant=analytic_point_mass metrics.end_at_first_contact=true"
UNMATCHED_STEP="env.physics_step_s=0.01 env.physics_steps_per_control_step=2"
declare -A STEP_OVERRIDES=(
  [b4_held_rate]="$POINT_MASS plant.body_rate_report=held_turn_rate"
  [b5_zero_rate]="$POINT_MASS plant.attitude_time_constant_s=1.3 plant.body_rate_report=zero"
  [unmatched_step10ms]="plant=analytic_rigid_body $UNMATCHED_STEP"
  [b5_tau0p3]="$POINT_MASS plant.attitude_time_constant_s=0.3"
  [b5_tau0p65]="$POINT_MASS plant.attitude_time_constant_s=0.65"
  [b5_tau2p2]="$POINT_MASS plant.attitude_time_constant_s=2.2"
)
TRAIN="python scripts/train_ppo.py scenario=s1_terminal_truth num_envs=4096 agent.ppo.save_interval=25"
EVAL="python scripts/evaluate_agent.py scenario=s1_terminal_truth agent=ppo_mlp num_envs=4096 seed=1000 episodes_per_env=1 logged_envs=256"
success_of() { grep '\[lunarsim\] episodes' "$1" | sed 's/.*success rate: \([0-9.]*\).*/\1/'; }

echo "[revision] commit $(git rev-parse --short HEAD), start $(date -Is)"

# ---- R20, evaluation only: b4 policies with a zero body rate in the policy input ----
for seed in 0 1 2 3 4; do
  run=t9_eval_b4_masked_rate_policy_in_physx_seed$seed
  if [ ! -f "runs/$run/episodes.parquet" ]; then
    clean_shm
    $EVAL agent.policy_path="runs/t8_b4_point_mass_actuators_seed$seed/policy.pt" agent.zero_body_rate_in_policy_input=true run_name="$run" > "runs/_jobs/$run.log" 2>&1
  fi
  echo "[revision] b4 seed $seed, rate masked, in PhysX: $(success_of runs/_jobs/$run.log)"
done

# ---- R17, agreement check of ZEM/ZEV on the training models it can fly ----
declare -A GATE_OVERRIDES=(
  [b1_end_at_contact]="plant=analytic_rigid_body metrics.end_at_first_contact=true"
  [b2_ideal_actuators]="plant=analytic_rigid_body metrics.end_at_first_contact=true actuators.ideal_response=true"
  [unmatched_step10ms]="plant=analytic_rigid_body $UNMATCHED_STEP"
)
for model in b1_end_at_contact b2_ideal_actuators unmatched_step10ms; do
  run=t9_gate_zem_zev_${model}_seed1000
  if [ ! -f "runs/$run/episodes.parquet" ]; then
    python scripts/evaluate_agent.py scenario=s1_terminal_truth agent=zem_zev num_envs=4096 seed=1000 logged_envs=64 ${GATE_OVERRIDES[$model]} run_name=$run > "runs/_jobs/$run.log" 2>&1
  fi
  echo "[revision] gate $model: ZEM/ZEV success $(success_of runs/_jobs/$run.log)"
  python -m lunarsim.tools.command_line compare-simulators "runs/$run" runs/t3_s1_zem_zev_physx_seed1000 > "runs/_jobs/${run}_compare.txt" 2>&1
  tail -n 25 "runs/_jobs/${run}_compare.txt"
done

# ---- trainings, in order of priority ----
for step in b4_held_rate b5_zero_rate unmatched_step10ms b5_tau0p3 b5_tau0p65 b5_tau2p2; do
  for seed in 0 1 2 3 4; do
    name=t9_${step}_seed$seed
    if [ ! -f "runs/$name/policy.pt" ]; then
      $TRAIN seed=$seed ${STEP_OVERRIDES[$step]} run_name="$name" > "runs/_jobs/train_$name.log" 2>&1
      grep "\[lunarsim\] trained" "runs/_jobs/train_$name.log" || { echo "[revision] WARNING: $name did not finish (runs/_jobs/train_$name.log)"; continue; }
    fi
    run=t9_eval_${step}_policy_in_physx_seed$seed
    if [ ! -f "runs/$run/episodes.parquet" ]; then
      clean_shm
      $EVAL agent.policy_path="runs/$name/policy.pt" run_name="$run" > "runs/_jobs/$run.log" 2>&1
    fi
    echo "[revision] $name in PhysX: $(success_of runs/_jobs/$run.log)  (training success $(grep -o 'success_rate: [0-9.]*' "runs/_jobs/train_$name.log" | tail -1 | cut -d' ' -f2))"
  done
done
# ---- R20 follow-up, evaluation only: b5_zero_rate policies with a zero body rate in the policy
# input, so that their input in PhysX matches their training input (added after b5_zero_rate fell
# to about 0 % with the physical PhysX rate in the input) ----
for seed in 0 1 2 3 4; do
  run=t9_eval_b5_zero_rate_masked_rate_policy_in_physx_seed$seed
  if [ ! -f "runs/$run/episodes.parquet" ]; then
    clean_shm
    $EVAL agent.policy_path="runs/t9_b5_zero_rate_seed$seed/policy.pt" agent.zero_body_rate_in_policy_input=true run_name="$run" > "runs/_jobs/$run.log" 2>&1
  fi
  echo "[revision] b5_zero_rate seed $seed, rate masked, in PhysX: $(success_of runs/_jobs/$run.log)"
done
echo "[revision] done $(date -Is)"
