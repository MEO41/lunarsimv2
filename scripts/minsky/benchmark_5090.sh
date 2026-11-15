#!/bin/bash
# Speed benchmark on the RTX 5090 (minsky), run with the GPU otherwise idle. Measures PPO training
# speed in S3 (the paper 1 scenario, ppo_mlp) as seconds per iteration:
#   1. one run alone at 4,096 / 16,384 / 32,768 landers;
#   2. five runs side by side at 4,096 without MPS (the setting used so far);
#   3. five runs side by side at 4,096 with CUDA MPS (kernels of all runs at the same time);
#   4. two runs side by side at 16,384 with MPS.
# Plus the environment-only speed table (zem_zev in S1, benchmark_environment.py) at 4,096 to 65,536.
# Output: runs/benchmark_5090/summary.txt (one line per case). About 20-30 minutes.
source ~/lunarsim-workstation/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
OUT=runs/benchmark_5090; mkdir -p $OUT; SUMMARY=$OUT/summary.txt; : > $SUMMARY
clean_shm() { for f in /dev/shm/sem.carb-RStringInternals-* /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] || continue; pid=${f##*-}; kill -0 $pid 2>/dev/null || rm -f "$f"; done; }
ITERATIONS=15
train() {  # train <name> <num_envs>; writes $OUT/<name>.log
  python scripts/train_ppo.py scenario=s3_faults agent=ppo_mlp num_envs=$2 seed=0 run_name=benchmark_$1 max_iterations=$ITERATIONS > $OUT/$1.log 2>&1
}
seconds_per_iteration() {  # mean of the last 10 iteration times in a log
  grep "Iteration time" $1 | tail -10 | awk '{s+=$3} END {if (NR) printf "%.2f", s/NR; else printf "failed"}'
}
report() {  # report <case> <num_envs> <runs> <log...>
  local case=$1 envs=$2 runs=$3; shift 3
  local times=""; local total=0
  for log in "$@"; do t=$(seconds_per_iteration $log); times="$times $t"; done
  local mean=$(echo $times | tr ' ' '\n' | awk '{s+=$1} END {printf "%.2f", s/NR}')
  # steps per iteration: 24 decisions x 5 control steps x landers
  local throughput=$(awk -v e=$envs -v r=$runs -v t=$mean 'BEGIN {printf "%.0f", r*24*5*e/t}')
  echo "$case | landers per run $envs | runs $runs | s/iteration$times | total env steps/s $throughput" | tee -a $SUMMARY
}

for envs in 4096 16384 32768; do clean_shm; train single_$envs $envs; report "one run alone" $envs 1 $OUT/single_$envs.log; done

clean_shm
for s in 1 2 3 4 5; do (train parallel_$s 4096 &); sleep 30; done
sleep 60; while pgrep -f "[b]enchmark_parallel_" >/dev/null; do sleep 10; done
report "five side by side, no MPS" 4096 5 $OUT/parallel_{1,2,3,4,5}.log

export CUDA_MPS_PIPE_DIRECTORY=$HOME/mps/pipe CUDA_MPS_LOG_DIRECTORY=$HOME/mps/log
mkdir -p $CUDA_MPS_PIPE_DIRECTORY $CUDA_MPS_LOG_DIRECTORY
if nvidia-cuda-mps-control -d; then
  clean_shm
  for s in 1 2 3 4 5; do (train mps_$s 4096 &); sleep 30; done
  sleep 60; while pgrep -f "[b]enchmark_mps_" >/dev/null; do sleep 10; done
  report "five side by side, MPS" 4096 5 $OUT/mps_{1,2,3,4,5}.log
  clean_shm
  for s in 1 2; do (train mps_large_$s 16384 &); sleep 30; done
  sleep 60; while pgrep -f "[b]enchmark_mps_large_" >/dev/null; do sleep 10; done
  report "two side by side, MPS" 16384 2 $OUT/mps_large_{1,2}.log
  echo quit | nvidia-cuda-mps-control
else
  echo "MPS daemon did not start" | tee -a $SUMMARY
fi
unset CUDA_MPS_PIPE_DIRECTORY CUDA_MPS_LOG_DIRECTORY

for envs in 4096 16384 32768 65536; do
  clean_shm
  line=$(python scripts/benchmark_environment.py scenario=s1_terminal_truth num_envs=$envs steps=300 2>&1 | grep "^\[benchmark\]")
  echo "environment only, S1 zem_zev, $envs landers: $line" | tee -a $SUMMARY
done
rm -rf runs/benchmark_single_* runs/benchmark_parallel_* runs/benchmark_mps_*
echo "[benchmark] done $(date)" | tee -a $SUMMARY
