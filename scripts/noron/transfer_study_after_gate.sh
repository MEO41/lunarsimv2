#!/bin/bash
# Analytic-to-PhysX transfer study: tests that need no training, queued after the one-seed gate
# (docs/knowledge/analytic-to-physx-transfer-study.md).
#
# 1. T3: ZEM/ZEV in both simulators on the 4,096 S1 starts of seed 1000; then the wide start range: ZEM/ZEV in both simulators on the same 4,096 starts, footpad altitude
#    5-150 m and heading 0-360 degrees (the rest of the S1 box unchanged), seeds 1000-1002, each
#    pair compared with `lunarsim compare-simulators`.
# 2. T2, throughput: benchmark_environment.py for both simulators, environment counts 1 to 16,384,
#    5 repeats, with ZEM/ZEV and with the gate's PhysX-trained PPO policy in the loop. One process
#    per measurement; every result line goes to runs/t2_throughput/benchmark.jsonl.
#
# Runs on noron, only on DISK02. It waits until the gate queue has finished, so no two jobs share
# the GPU. From the repository root:
#   setsid nohup bash scripts/noron/transfer_study_after_gate.sh > runs/_jobs/transfer_study_after_gate.log 2>&1 < /dev/null &
set -u
REPOSITORY=/media/noron/DISK02/Speed_Versus_Fidelity/lunarsimv2
cd "$REPOSITORY" || exit 1
source /media/noron/DISK02/isaac-lab-native/env_isaaclab/bin/activate
export PYTHONPATH=$REPOSITORY/src OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
clean_shm() { for f in /dev/shm/sem.carb-RStringInternals-* /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] || continue; pid=${f##*-}; kill -0 "$pid" 2>/dev/null || rm -f "$f"; done; }
ANALYTIC="plant=analytic_rigid_body"

while pgrep -f "[t]ransfer_study_gate.sh" > /dev/null; do sleep 60; done
echo "[after-gate] commit $(git rev-parse --short HEAD), start $(date -Is)"

# ---- 1. T3 on the S1 box (seed 1000), then the wide range ----
for simulator in physx analytic; do
  run=t3_s1_zem_zev_${simulator}_seed1000
  [ -f "runs/$run/episodes.parquet" ] && continue
  extra=""; [ "$simulator" = analytic ] && extra=$ANALYTIC
  clean_shm
  python scripts/evaluate_agent.py scenario=s1_terminal_truth agent=zem_zev num_envs=4096 seed=1000 logged_envs=64 $extra run_name=$run > "runs/_jobs/$run.log" 2>&1
  echo "[after-gate] $run: $(grep '\[lunarsim\] episodes' "runs/_jobs/$run.log")"
done
python -m lunarsim.tools.command_line compare-simulators runs/t3_s1_zem_zev_analytic_seed1000 runs/t3_s1_zem_zev_physx_seed1000
WIDE="start.altitude_m=[5,150] +start.heading_deg=[0,360]"
for seed in 1000 1001 1002; do
  for simulator in physx analytic; do
    run=t3_wide_zem_zev_${simulator}_seed$seed
    [ -f "runs/$run/episodes.parquet" ] && continue
    extra=""; [ "$simulator" = analytic ] && extra=$ANALYTIC
    clean_shm
    python scripts/evaluate_agent.py scenario=s1_terminal_truth agent=zem_zev num_envs=4096 seed=$seed logged_envs=64 $WIDE $extra run_name=$run > "runs/_jobs/$run.log" 2>&1
    echo "[after-gate] $run: $(grep '\[lunarsim\] episodes' "runs/_jobs/$run.log")"
  done
  python -m lunarsim.tools.command_line compare-simulators runs/t3_wide_zem_zev_analytic_seed$seed runs/t3_wide_zem_zev_physx_seed$seed
done

# ---- 2. T2 throughput ----
mkdir -p runs/t2_throughput
OUT=runs/t2_throughput/benchmark.jsonl
git rev-parse HEAD > runs/t2_throughput/commit.txt
POLICY=runs/transfer_ppo_physx_seed0/policy.pt
AGENTS=("agent=zem_zev")
[ -f "$POLICY" ] && AGENTS+=("agent=ppo_mlp agent.policy_path=$POLICY")
for repeat in 1 2 3 4 5; do
  for agent in "${AGENTS[@]}"; do
    for count in 1 4 16 64 256 1024 4096 8192 16384; do
      for simulator in physx analytic; do
        extra=""; [ "$simulator" = analytic ] && extra=$ANALYTIC
        clean_shm
        line=$(python scripts/benchmark_environment.py scenario=s1_terminal_truth num_envs=$count steps=300 $agent $extra 2>&1 | grep "\[benchmark\]")
        if [ -n "$line" ]; then echo "${line#\[benchmark\] }" | python -c "import json,sys; row=json.loads(sys.stdin.read()); row['repeat']=$repeat; print(json.dumps(row))" >> "$OUT"
        else echo "[after-gate] WARNING: no benchmark line for $simulator $count envs ($agent), repeat $repeat"; fi
      done
    done
  done
  echo "[after-gate] throughput repeat $repeat done $(date -Is)"
done
echo "[after-gate] done $(date -Is)"
