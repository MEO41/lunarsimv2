#!/bin/bash
# Queue on minsky: wait for the S2 trainings, evaluate them on the held-out starts (M2: S1 policy
# vs S2 policy), then run the speed benchmark and stop (the paper 1 setting is chosen from it).
source ~/lunarsim-workstation/env_isaaclab/bin/activate
export OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
clean_shm() { for f in /dev/shm/sem.carb-RStringInternals-* /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] || continue; pid=${f##*-}; kill -0 $pid 2>/dev/null || rm -f "$f"; done; }
echo "[queue] waiting for the S2 trainings $(date)"
while pgrep -f "[t]rain_ppo.py scenario=s2_terminal_ekf" >/dev/null; do sleep 60; done
git pull -q
for s in 0 1 2 3 4; do
  clean_shm
  python scripts/evaluate_agent.py num_envs=4096 seed=1000 episodes_per_env=1 scenario=s2_terminal_ekf agent=ppo_mlp \
    agent.policy_path=runs/ppo_mlp_s2_seed$s/policy.pt run_name=eval_ppo_s2policy_seed${s}_s2_terminal_ekf_heldout > runs/eval_ppo_s2policy_seed${s}_s2.log 2>&1
done
echo "[queue] S2 policies evaluated $(date)"
bash scripts/minsky/benchmark_5090.sh > runs/benchmark_5090.log 2>&1
echo "[queue] benchmark done, stopping for the user's choice $(date)"
