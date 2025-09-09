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
python -m pip install -U pip setuptools wheel packaging
python -m pip install -U ninja cython pybind11

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

# 5) Install remaining project/ByteTrack deps (do not modify 3rd-party sources)
if [[ "${MODE}" != "--vision-only" ]]; then
  echo "[setup_env] Installing project and ByteTrack dependencies ..."
  if [[ -f third_party/ByteTrack/requirements.txt ]]; then
    # Keep PyTorch stack intact; other deps from PyPI; do not fail the whole setup
    # if onnx-simplifier/onnxsim is inconsistent on Py3.13.
    python -m pip install -r third_party/ByteTrack/requirements.txt || true
  fi
  # ByteTrack imports 'thop' unconditionally in yolox/utils/model_utils.py
  # Install it explicitly without deps to avoid touching torch/vision.
  python -m pip install --no-deps "thop==0.1.1.post2209072238"
  # Useful at runtime; harmless if already present
  python -m pip install -U loguru opencv-python
  # Editable install of ByteTrack WITHOUT changing its sources
  python -m pip install -e third_party/ByteTrack
fi

echo "[setup_env] Done."
