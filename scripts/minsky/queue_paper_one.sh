#!/bin/bash
# Queue for the RTX 5090 machine (minsky): after the S2 trainings, evaluate them (M2), train the
# three S3 encoders (5 seeds each, one encoder at a time) and evaluate everything on the fault grid.
# Run from the repository root: setsid nohup bash scripts/minsky/queue_paper_one.sh > runs/queue_paper_one.log 2>&1 &
source ~/lunarsim-workstation/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
clean_shm() { for f in /dev/shm/sem.carb-RStringInternals-* /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] || continue; pid=${f##*-}; kill -0 $pid 2>/dev/null || rm -f "$f"; done; }
wait_for_training() { while pgrep -f "[t]rain_ppo.py scenario=$1" >/dev/null; do sleep 60; done; }
SEEDS="0 1 2 3 4"
EVAL="python scripts/evaluate_agent.py num_envs=4096 seed=1000 episodes_per_env=1"

echo "[queue] waiting for the S2 trainings $(date)"; wait_for_training s2_terminal_ekf
git pull -q
for s in $SEEDS; do
  clean_shm; $EVAL scenario=s2_terminal_ekf agent=ppo_mlp agent.policy_path=runs/ppo_mlp_s2_seed$s/policy.pt run_name=eval_ppo_s2policy_seed${s}_s2_terminal_ekf_heldout > runs/eval_ppo_s2policy_seed${s}_s2.log 2>&1
done
echo "[queue] S2 policies evaluated $(date)"

for kind in mlp gru attention; do
  for s in $SEEDS; do
    clean_shm
    (setsid nohup python scripts/train_ppo.py scenario=s3_faults agent=ppo_$kind num_envs=4096 seed=$s run_name=ppo_${kind}_s3_seed$s > runs/train_ppo_${kind}_s3_seed$s.log 2>&1 < /dev/null &)
    sleep 45
  done
  sleep 120; wait_for_training s3_faults
  echo "[queue] S3 $kind trained $(date)"
done

clean_shm; $EVAL scenario=s3_faults fault_grid=true run_name=grid_zem_zev_ekf_fdi > runs/grid_zem_zev_ekf_fdi.log 2>&1
clean_shm; $EVAL scenario=s3_faults fault_grid=true nav=ekf run_name=grid_zem_zev_ekf > runs/grid_zem_zev_ekf.log 2>&1
clean_shm; $EVAL scenario=s2_terminal_ekf run_name=grid_reference_zem_zev_no_faults > runs/grid_reference_zem_zev_no_faults.log 2>&1
for kind in mlp gru attention; do
  for s in $SEEDS; do
    clean_shm; $EVAL scenario=s3_faults fault_grid=true agent=ppo_$kind agent.policy_path=runs/ppo_${kind}_s3_seed$s/policy.pt run_name=grid_ppo_${kind}_seed$s > runs/grid_ppo_${kind}_seed$s.log 2>&1
  done
done
echo "[queue] fault grid evaluated $(date)"
