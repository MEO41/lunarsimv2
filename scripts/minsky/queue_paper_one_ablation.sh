#!/bin/bash
# Paper 1 ablation (user decision 2026-11-16): the same S3 fault study with `ekf_fdi`, but the
# policy does not see the filter's health flags (observation.include_navigation_health=false), so
# only a history of readings can reveal a fault. MLP 600 iterations, GRU and attention 1,500, all
# from scratch, 5 seeds each; runs ppo_<kind>_s3_noflags_seed<s>, then every policy flies the
# held-out fault grid (grid_ppo_<kind>_noflags_seed<s>).
# Usage, from the repository root:
#   setsid nohup bash scripts/minsky/queue_paper_one_ablation.sh > runs/queue_paper_one_ablation.log 2>&1 &
# Trainings share the GPU under CUDA MPS. Longest first; a training starts only when the GPU has
# the free memory it needs (attention about 11 GB, MLP and GRU about 4 GB, plus a margin), checked
# 3 minutes after the previous start so that its memory has grown to its working size.
source ~/lunarsim-workstation/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
clean_shm() { for f in /dev/shm/sem.carb-RStringInternals-* /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] || continue; pid=${f##*-}; kill -0 $pid 2>/dev/null || rm -f "$f"; done; }
free_mib() { nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1; }
needed_mib() { case $1 in attention) echo 13500 ;; *) echo 6000 ;; esac; }
iterations_for() { case $1 in mlp) echo 600 ;; *) echo 1500 ;; esac; }
NO_FLAGS="observation.include_navigation_health=false"
SEEDS="0 1 2 3 4"
EVAL="python scripts/evaluate_agent.py num_envs=4096 seed=1000 episodes_per_env=1"

export CUDA_MPS_PIPE_DIRECTORY=$HOME/mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/mps/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
nvidia-cuda-mps-control -d && echo "[queue] MPS daemon started $(date)" || { echo "[queue] MPS did not start"; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; }

# Pending trainings, longest first. Each round starts the first pending one that fits in the
# free memory, so a GRU or MLP run can start beside two attention runs.
PENDING=()
for kind in attention gru mlp; do for s in $SEEDS; do PENDING+=("$kind:$s"); done; done
while [ ${#PENDING[@]} -gt 0 ]; do
  free=$(free_mib); started=""
  for index in "${!PENDING[@]}"; do
    kind=${PENDING[$index]%%:*}; s=${PENDING[$index]##*:}
    if [ "$free" -ge "$(needed_mib $kind)" ]; then
      clean_shm
      name=ppo_${kind}_s3_noflags_seed$s
      (setsid nohup python scripts/train_ppo.py scenario=s3_faults agent=ppo_$kind num_envs=4096 seed=$s run_name=$name max_iterations=$(iterations_for $kind) $NO_FLAGS > runs/train_$name.log 2>&1 < /dev/null &)
      echo "[queue] started $name (free $free MiB) $(date)"
      unset "PENDING[$index]"; PENDING=("${PENDING[@]}"); started=yes
      break
    fi
  done
  if [ -n "$started" ]; then sleep 180; else sleep 60; fi
done
sleep 60; while pgrep -f "[t]rain_ppo.py scenario=s3_faults" >/dev/null; do sleep 60; done
[ -n "$CUDA_MPS_PIPE_DIRECTORY" ] && { echo quit | nvidia-cuda-mps-control; unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY; echo "[queue] MPS daemon stopped"; }
for kind in mlp gru attention; do
  for s in $SEEDS; do
    name=ppo_${kind}_s3_noflags_seed$s
    [ -f runs/$name/policy.pt ] || { echo "[queue] WARNING: $name has no policy.pt (see runs/train_$name.log)"; continue; }
    clean_shm; $EVAL scenario=s3_faults fault_grid=true $NO_FLAGS agent=ppo_$kind agent.policy_path=runs/$name/policy.pt run_name=grid_ppo_${kind}_noflags_seed$s > runs/grid_ppo_${kind}_noflags_seed$s.log 2>&1
  done
done
echo "[queue] fault grid evaluated $(date)"
