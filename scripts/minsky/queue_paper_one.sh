#!/bin/bash
# Paper 1 on the RTX 5090 (minsky): train the three S3 encoders (5 seeds each, one encoder at a
# time) and evaluate everything on the held-out fault grid.
# Usage, from the repository root: setsid nohup bash scripts/minsky/queue_paper_one.sh [LANDERS_PER_RUN] [mps|no_mps] > runs/queue_paper_one.log 2>&1 &
# LANDERS_PER_RUN (default 4096) is chosen from scripts/minsky/benchmark_5090.sh. With mps (the
# default, user decision 2026-11-15) the five trainings of an encoder share the GPU through CUDA
# MPS, which the benchmark measured at twice the speed of time-slicing (350k vs 172k env steps/s
# at 5 x 4,096). The MPS daemon stops before the evaluations.
# Rerun part of it: TRAIN_KINDS="attention" TRAIN_SEEDS="2 3 4" SKIP_BASELINES=1 bash scripts/minsky/queue_paper_one.sh
source ~/lunarsim-workstation/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
clean_shm() { for f in /dev/shm/sem.carb-RStringInternals-* /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] || continue; pid=${f##*-}; kill -0 $pid 2>/dev/null || rm -f "$f"; done; }
wait_for_training() { while pgrep -f "[t]rain_ppo.py scenario=$1" >/dev/null; do sleep 60; done; }
SEEDS="0 1 2 3 4"
LANDERS=${1:-4096}
SHARING=${2:-mps}
if [ "$SHARING" = mps ]; then
  export CUDA_MPS_PIPE_DIRECTORY=$HOME/mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/mps/log
  mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
  if nvidia-cuda-mps-control -d; then echo "[queue] MPS daemon started $(date)"; else echo "[queue] MPS daemon did not start, training without it"; SHARING=no_mps; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; fi
fi
echo "[queue] paper 1: $LANDERS landers per run, GPU sharing $SHARING $(date)"
EVAL="python scripts/evaluate_agent.py num_envs=4096 seed=1000 episodes_per_env=1"

# Trainings of one encoder side by side, at most PARALLEL at once: an attention training needs
# about 11 GB at 4,096 landers (2026-11-15: five did not fit on the 32 GB card and three stopped
# with CUDA out of memory), MLP and GRU trainings about 4 GB.
parallel_for() { case $1 in attention) echo 2 ;; *) echo 5 ;; esac; }
running() { pgrep -f "[t]rain_ppo.py scenario=s3_faults" | wc -l; }
TRAIN_KINDS=${TRAIN_KINDS:-mlp gru attention}
TRAIN_SEEDS=${TRAIN_SEEDS:-$SEEDS}
for kind in $TRAIN_KINDS; do
  for s in $TRAIN_SEEDS; do
    while [ "$(running)" -ge "$(parallel_for $kind)" ]; do sleep 60; done
    clean_shm
    (setsid nohup python scripts/train_ppo.py scenario=s3_faults agent=ppo_$kind num_envs=$LANDERS seed=$s run_name=ppo_${kind}_s3_seed$s > runs/train_ppo_${kind}_s3_seed$s.log 2>&1 < /dev/null &)
    sleep 45
  done
  sleep 120; wait_for_training s3_faults
  for s in $TRAIN_SEEDS; do
    [ -f runs/ppo_${kind}_s3_seed$s/policy.pt ] || echo "[queue] WARNING: ppo_${kind}_s3_seed$s has no policy.pt (training failed, see runs/train_ppo_${kind}_s3_seed$s.log)"
  done
  echo "[queue] S3 $kind trained $(date)"
done
if [ "$SHARING" = mps ]; then echo quit | nvidia-cuda-mps-control; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; echo "[queue] MPS daemon stopped"; fi

if [ -z "$SKIP_BASELINES" ]; then
  clean_shm; $EVAL scenario=s3_faults fault_grid=true run_name=grid_zem_zev_ekf_fdi > runs/grid_zem_zev_ekf_fdi.log 2>&1
  clean_shm; $EVAL scenario=s3_faults fault_grid=true nav=ekf run_name=grid_zem_zev_ekf > runs/grid_zem_zev_ekf.log 2>&1
  clean_shm; $EVAL scenario=s2_terminal_ekf run_name=grid_reference_zem_zev_no_faults > runs/grid_reference_zem_zev_no_faults.log 2>&1
fi
for kind in $TRAIN_KINDS; do
  for s in $TRAIN_SEEDS; do
    [ -f runs/ppo_${kind}_s3_seed$s/policy.pt ] || { echo "[queue] skip grid of ppo_${kind}_s3_seed$s (no policy)"; continue; }
    clean_shm; $EVAL scenario=s3_faults fault_grid=true agent=ppo_$kind agent.policy_path=runs/ppo_${kind}_s3_seed$s/policy.pt run_name=grid_ppo_${kind}_seed$s > runs/grid_ppo_${kind}_seed$s.log 2>&1
  done
done
echo "[queue] fault grid evaluated $(date)"
