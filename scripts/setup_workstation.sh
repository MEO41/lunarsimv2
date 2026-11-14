#!/usr/bin/env bash
# Set up a new workstation for LunarSim v2: Python environment, Isaac Sim, Isaac Lab, this package.
#
#   scripts/setup_workstation.sh [INSTALL_ROOT]      (default INSTALL_ROOT: $HOME/lunarsim-workstation)
#
# What it does, in order (each step is skipped if it is already done, so you can run it again):
#   1. checks the GPU driver and Python 3.12
#   2. creates a virtual environment  INSTALL_ROOT/env_isaaclab
#   3. installs torch (CUDA 13 build), then Isaac Sim 6.0.1 from pypi.nvidia.com
#   4. clones Isaac Lab and checks out the pinned commit, installs it
#   5. installs the pinned scientific packages and this repository (pip install -e)
#   6. runs scripts/check_installation.py, which compares every version with the pinned ones
#
# Everything is printed. If a step fails, the script stops and says which one.
# Override with environment variables:  TORCH_INDEX_URL, ISAACLAB_COMMIT, ISAACLAB_REPOSITORY.
# NOT tested on a clean machine yet: the first run on the new workstation is the test.
set -euo pipefail

REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INSTALL_ROOT="${1:-$HOME/lunarsim-workstation}"
ENVIRONMENT_DIRECTORY="$INSTALL_ROOT/env_isaaclab"
ISAACLAB_DIRECTORY="$INSTALL_ROOT/IsaacLab"
ISAACLAB_REPOSITORY="${ISAACLAB_REPOSITORY:-https://github.com/isaac-sim/IsaacLab.git}"
ISAACLAB_COMMIT="${ISAACLAB_COMMIT:-e86463df8a8d8dff275e50fdcfd892e10ba32ec1}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu130}"
NVIDIA_INDEX_URL="https://pypi.nvidia.com"
MINIMUM_DRIVER_MAJOR=580   # Isaac Sim 6 needs a recent driver; the development machine has 595

step() { printf '\n==== %s\n' "$*"; }
fail() { printf '\nSETUP FAILED: %s\n' "$*" >&2; exit 1; }

step "1/6 Checking GPU driver and Python"
command -v nvidia-smi >/dev/null || fail "nvidia-smi not found: install the NVIDIA driver first."
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv
DRIVER_MAJOR="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
[ "$DRIVER_MAJOR" -ge "$MINIMUM_DRIVER_MAJOR" ] || fail "driver $DRIVER_MAJOR is older than $MINIMUM_DRIVER_MAJOR."
command -v python3.12 >/dev/null || fail "python3.12 not found (Ubuntu 24.04: sudo apt install python3.12 python3.12-venv)."
command -v git >/dev/null || fail "git not found."
mkdir -p "$INSTALL_ROOT"

step "2/6 Virtual environment: $ENVIRONMENT_DIRECTORY"
if [ ! -x "$ENVIRONMENT_DIRECTORY/bin/python" ]; then
  python3.12 -m venv "$ENVIRONMENT_DIRECTORY"
fi
# shellcheck disable=SC1091
source "$ENVIRONMENT_DIRECTORY/bin/activate"
python -m pip install --upgrade pip "setuptools==81.0.0" wheel

step "3/6 torch (CUDA 13) and Isaac Sim 6.0.1"
pip install "torch==2.11.0" "torchvision==0.26.0" --index-url "$TORCH_INDEX_URL"
pip install "isaacsim[all,extscache]==6.0.1.0" --extra-index-url "$NVIDIA_INDEX_URL"

step "4/6 Isaac Lab at commit $ISAACLAB_COMMIT"
if [ ! -d "$ISAACLAB_DIRECTORY/.git" ]; then
  git clone "$ISAACLAB_REPOSITORY" "$ISAACLAB_DIRECTORY"
fi
git -C "$ISAACLAB_DIRECTORY" fetch origin
git -C "$ISAACLAB_DIRECTORY" checkout "$ISAACLAB_COMMIT"
(cd "$ISAACLAB_DIRECTORY" && ./isaaclab.sh --install core)

step "5/6 Pinned packages and LunarSim v2"
pip install -r "$REPOSITORY_ROOT/requirements-pinned.txt" --extra-index-url "$NVIDIA_INDEX_URL"
pip install -e "$REPOSITORY_ROOT[dev]"

step "6/6 Checking the installation"
python "$REPOSITORY_ROOT/scripts/check_installation.py"

cat <<DONE

Setup finished. Next:
  source "$ENVIRONMENT_DIRECTORY/bin/activate"
  cd "$REPOSITORY_ROOT"
  python -m pytest tests/unit -q                 # 89 CPU tests
  python scripts/run_isaac_tests.py              # GPU tests, one simulator process per file
  python scripts/benchmark_environment.py --help # speed on this GPU
DONE
