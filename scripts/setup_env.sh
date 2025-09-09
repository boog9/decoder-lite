#!/usr/bin/env bash
# Copyright 2024
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Decoder-lite environment bootstrap.
# Example usage:
#   make venv
# Example output:
#   [setup_env] Done.

set -euo pipefail

# Usage:
#   bash scripts/setup_env.sh             # full setup
#   bash scripts/setup_env.sh --vision-only  # only install Torch/Torchvision
MODE="${1:-full}"
if [[ "${MODE}" != "full" && "${MODE}" != "--vision-only" ]]; then
  echo "Usage: bash scripts/setup_env.sh [--vision-only]" >&2
  exit 1
fi

# Ensure virtual environment exists and is activated.
if [[ ! -d .venv ]]; then
  python -m venv .venv
fi
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[setup_env] Python: $(python -V 2>&1)"

# 0) Ensure ByteTrack sources are present (NO submodules; use the clone script)
if [[ "${MODE}" != "--vision-only" ]]; then
  if [[ ! -f third_party/ByteTrack/yolox/__init__.py ]]; then
    echo "[setup_env] Cloning ByteTrack via scripts/clone_bytetrack.sh ..."
    bash scripts/clone_bytetrack.sh
  else
    echo "[setup_env] ByteTrack already present."
  fi
fi

# 1) Base build tools (from PyPI)
# Clean up broken pip artifacts that show up as "Ignoring invalid distribution ~ip"
python - <<'PY'
import site, glob, os, shutil
paths = set(site.getsitepackages() + [site.getusersitepackages()])
for p in paths:
    if not os.path.isdir(p):
        continue
    for bad in glob.glob(os.path.join(p, '*~ip*')):
        try:
            (shutil.rmtree if os.path.isdir(bad) else os.remove)(bad)
        except Exception:
            pass
PY
python -m pip install --force-reinstall --no-cache-dir "pip==25.2" "setuptools>=68" "wheel"
python -m pip install -U packaging ninja cython pybind11

# 2) Install Torch from the official CUDA 12.1 wheel index (GPU build)
echo "[setup_env] Installing torch 2.5.1+cu121 ..."
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.5.1+cu121

# 3) Build torchvision v0.20.1 from source with CUDA for Python 3.13
echo "[setup_env] Building torchvision 0.20.1 from source with CUDA ..."
export FORCE_CUDA=1
export MAX_JOBS="${MAX_JOBS:-"$(/usr/bin/nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"}"
python -m pip install --no-build-isolation --no-cache-dir -v \
  "git+https://github.com/pytorch/vision.git@v0.20.1"

# 4) Verify that torchvision registered CUDA ops including nms
echo "[setup_env] Verifying torchvision::nms ..."
python scripts/verify_torchvision_nms.py --quiet || {
  echo "[setup_env] torchvision::nms missing; aborting." >&2
  exit 1
}

# 5) Install runtime dependencies for ByteTrack (avoid requirements.txt)
if [[ "${MODE}" != "--vision-only" ]]; then
  echo "[setup_env] Installing runtime dependencies for ByteTrack ..."
  python -m pip install -U "numpy<2.3" opencv-python loguru "easydict>=1.10" "scikit-image<0.25" thop motmetrics
  # Try to install the faster 'lap'; fall back to 'lapx' if wheels are missing
  python -m pip install lap || python -m pip install lapx
  echo "[setup_env] Installing ByteTrack in editable mode ..."
  # Ensure torch is already present; disable build isolation so setup.py can import torch
  python -m pip install -e third_party/ByteTrack --no-deps --no-build-isolation --config-settings editable_mode=compat
fi

echo "[setup_env] Done."
