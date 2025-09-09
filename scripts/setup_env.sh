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

# Prepare Python environment and install ByteTrack.
# Example usage:
#   bash scripts/setup_env.sh
# Example output:
#   Torch OK 2.8.0 cu124

set -euo pipefail

# Ensure ByteTrack vendor tree is present before installation.
bash scripts/ensure_bytetrack.sh

# Create or reuse a virtual environment.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

install_pytorch() {
  set -euo pipefail
  # 1) If torch already present, do nothing.
  if python - <<'PY' 2>/dev/null; then
import sys
import torch  # noqa
sys.exit(0)
PY
  then
    echo "[setup] torch already installed"
    return 0
  fi

  OS_NAME="$(uname -s || echo Unknown)"
  WANT_CPU="0"
  # 2) Honour CPU mode and macOS
  if [ "${ALLOW_CPU_ORT:-0}" = "1" ] || [ "${FORCE_TORCH_CPU:-0}" = "1" ] || [ "$OS_NAME" = "Darwin" ]; then
    WANT_CPU="1"
  fi

  # 3) If not forced to CPU, detect NVIDIA GPU
  if [ "$WANT_CPU" = "0" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L 2>/dev/null | grep -q "GPU"; then
      TORCH_MODE="cu124"
    else
      TORCH_MODE="cpu"
    fi
  else
    TORCH_MODE="cpu"
  fi

  python -m pip install --upgrade pip
  if [ "$TORCH_MODE" = "cu124" ]; then
    echo "[setup] Installing PyTorch CUDA 12.4 wheels…"
    if ! python -m pip install "torch==2.8.*" "torchvision==0.23.*" --index-url https://download.pytorch.org/whl/cu124; then
      echo "[setup][warn] CUDA wheels install failed — falling back to CPU torch."
      TORCH_MODE="cpu"
    fi
  fi

  if [ "$TORCH_MODE" = "cpu" ]; then
    echo "[setup] Installing PyTorch CPU wheels…"
    python -m pip install "torch==2.8.*" "torchvision==0.23.*" --index-url https://download.pytorch.org/whl/cpu
  fi

  # 4) Sanity check (assert only for CUDA path)
  if [ "$TORCH_MODE" = "cu124" ]; then
    python - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA build of torch installed, but CUDA is not available at runtime"
print("Torch CUDA OK", torch.__version__, torch.version.cuda)
PY
  else
    python - <<'PY'
import torch, platform
print("Torch CPU OK", torch.__version__, "cuda_tag:", getattr(torch.version, "cuda", None))
PY
  fi
}

# main sequence
install_pytorch

# Project and ByteTrack dependencies.
python -m pip install -r requirements.txt
python -m pip install -r third_party/ByteTrack/requirements.txt
PIP_NO_BUILD_ISOLATION=1 python third_party/ByteTrack/setup.py develop
