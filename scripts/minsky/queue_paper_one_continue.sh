#!/bin/bash
# Paper 1, second step (user decision 2026-11-15): the GRU and attention policies were still
# learning at iteration 600, so each of their runs continues from model_600.pt to iteration
# TARGET (default 1500) in a new run directory ppo_<kind>_s3_seed<s>_to<TARGET>; the MLP stays
# at 600 (it settles by iteration 200). Then every continued policy flies the held-out fault grid.
# Usage, from the repository root:
#   setsid nohup bash scripts/minsky/queue_paper_one_continue.sh [TARGET] > runs/queue_paper_one_continue.log 2>&1 &
# Waits first for a running queue_paper_one.sh (the attention rerun) to finish.
source ~/lunarsim-workstation/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
clean_shm() { for f in /dev/shm/sem.carb-RStringInternals-* /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] || continue; pid=${f##*-}; kill -0 $pid 2>/dev/null || rm -f "$f"; done; }
running() { pgrep -f "[t]rain_ppo.py scenario=s3_faults" | wc -l; }
parallel_for() { case $1 in attention) echo 2 ;; *) echo 5 ;; esac; }  # attention needs about 11 GB per run
TARGET=${1:-1500}
SEEDS="0 1 2 3 4"
EVAL="python scripts/evaluate_agent.py num_envs=4096 seed=1000 episodes_per_env=1"

while pgrep -f "[q]ueue_paper_one.sh" >/dev/null; do sleep 60; done
echo "[queue] continue GRU and attention to iteration $TARGET $(date)"
export CUDA_MPS_PIPE_DIRECTORY=$HOME/mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/mps/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "[queue] MPS daemon started" || { echo "[queue] MPS did not start"; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; }

for kind in gru attention; do
  for s in $SEEDS; do
    checkpoint=runs/ppo_${kind}_s3_seed$s/model_600.pt
    [ -f $checkpoint ] || { echo "[queue] WARNING: no $checkpoint, seed $s skipped"; continue; }
    while [ "$(running)" -ge "$(parallel_for $kind)" ]; do sleep 60; done
    clean_shm
    name=ppo_${kind}_s3_seed${s}_to$TARGET
    (setsid nohup python scripts/train_ppo.py scenario=s3_faults agent=ppo_$kind num_envs=4096 seed=$s run_name=$name max_iterations=$TARGET resume_from=$checkpoint > runs/train_$name.log 2>&1 < /dev/null &)
    sleep 45
  done
  sleep 120; while [ "$(running)" -gt 0 ]; do sleep 60; done
  for s in $SEEDS; do
    [ -f runs/ppo_${kind}_s3_seed${s}_to$TARGET/policy.pt ] || echo "[queue] WARNING: ppo_${kind}_s3_seed${s}_to$TARGET has no policy.pt"
  done
  echo "[queue] $kind continued to $TARGET $(date)"
done
[ -n "$CUDA_MPS_PIPE_DIRECTORY" ] && { echo quit | nvidia-cuda-mps-control; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; echo "[queue] MPS daemon stopped"; }

for kind in gru attention; do
  for s in $SEEDS; do
    name=ppo_${kind}_s3_seed${s}_to$TARGET
    [ -f runs/$name/policy.pt ] || continue
    clean_shm; $EVAL scenario=s3_faults fault_grid=true agent=ppo_$kind agent.policy_path=runs/$name/policy.pt run_name=grid_ppo_${kind}_seed${s}_to$TARGET > runs/grid_ppo_${kind}_seed${s}_to$TARGET.log 2>&1
  done
done
echo "[queue] fault grid evaluated $(date)"
