#!/bin/bash
# Analytic-to-PhysX transfer study: 3D videos of trained policies flown in PhysX (user approval 2026-11-23).
#
# Each line records one policy on the study's 4,096 held-out starts (seed 1000) in PhysX, keeps the
# asked outcomes (scripts/record_policy_landings.py) and renders them (scripts/render_replay.py):
#   video_t5_analytic        T5: trained in the analytic simulator, lands in PhysX
#   video_b1_contact         T8 b1: trained with episodes that end at first contact, hops after contact
#   video_b2_ideal           T8 b2: trained with ideal actuators, hard landings
#   video_b3_point_mass      T8 b3: trained as a point mass (attitude at once), hovers or flies off
#   video_b5_point_mass_lag  T8 b5: point mass with a 1.3 s attitude lag, lands again
# Videos: runs/<name>/landings.mp4. Waits for the T8 queues, so no two jobs share the GPU.
#
# Runs on noron, only on DISK02. From the repository root:
#   setsid nohup bash scripts/noron/transfer_study_videos.sh > runs/_jobs/transfer_study_videos.log 2>&1 < /dev/null &
set -u
REPOSITORY=/media/noron/DISK02/Speed_Versus_Fidelity/lunarsimv2
cd "$REPOSITORY" || exit 1
source /media/noron/DISK02/isaac-lab-native/env_isaaclab/bin/activate
export PYTHONPATH=$REPOSITORY/src OMNI_KIT_ACCEPT_EULA=YES PYTHONUNBUFFERED=1
clean_shm() { for f in /dev/shm/sem.carb-RStringInternals-* /dev/shm/carb-RStringInternals-*; do [ -e "$f" ] || continue; pid=${f##*-}; kill -0 "$pid" 2>/dev/null || rm -f "$f"; done; }

while pgrep -f "^bash scripts/noron/transfer_study_t8" > /dev/null; do sleep 60; done
echo "[videos] commit $(git rev-parse --short HEAD), start $(date -Is)"

video() {  # name policy outcomes label
  local name=$1 policy=$2 outcomes=$3 label=$4
  if [ ! -f "runs/$name/landings.npz" ]; then
    clean_shm
    python scripts/record_policy_landings.py policy="$policy" run_name="$name" outcomes="$outcomes" label="$label" \
      > "runs/_jobs/record_$name.log" 2>&1 || { echo "[videos] WARNING: recording $name failed (runs/_jobs/record_$name.log)"; return; }
  fi
  grep "\[videos\]" "runs/_jobs/record_$name.log" | head -1
  clean_shm
  python scripts/render_replay.py replay="runs/$name/landings.npz" mode=video > "runs/_jobs/render_$name.log" 2>&1 \
    && echo "[videos] $name: $(ls runs/$name/landings.mp4) $(date -Is)" \
    || echo "[videos] WARNING: rendering $name failed (runs/_jobs/render_$name.log)"
}

video video_t5_analytic runs/transfer_ppo_analytic_seed0/policy.pt "success:2" "Trained in the analytic simulator"
video video_b1_contact runs/t8_b1_contact_seed3/policy.pt "not_settled:2" "Trained with episodes ending at first contact"
video video_b2_ideal runs/t8_b2_ideal_seed2/policy.pt "hard_vertical:1,hard_horizontal:1" "Trained with ideal actuators"
video video_b3_point_mass runs/t8_b3_point_mass_seed0/policy.pt "propellant_depleted:1,hard_horizontal:1" "Trained as a point mass, attitude at once"
video video_b5_point_mass_lag runs/t8_b5_point_mass_lag_seed0/policy.pt "success:2" "Trained as a point mass with a 1.3 s attitude lag"
echo "[videos] done $(date -Is)"
