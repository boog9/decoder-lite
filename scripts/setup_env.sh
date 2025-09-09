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

# Idempotent environment setup for decoder-lite.
# Example usage:
#   bash scripts/setup_env.sh
# Example output:
#   [setup_env] DONE

set -euo pipefail

# 0) Ensure ByteTrack submodule is present.
if [ ! -f third_party/ByteTrack/yolox/__init__.py ]; then
  git submodule update --init --recursive third_party/ByteTrack
fi

# 1) Upgrade build tools.
python -m pip install -U pip setuptools wheel

# 2) Install ninja from PyPI (no PyTorch index).
python -m pip install ninja

# 3) Install torch (CUDA 12.1 wheels) for cp313; fall back to nightly if needed.
set +e
python -m pip install --index-url https://download.pytorch.org/whl/cu121 'torch>=2.5,<2.7'
TORCH_RC=$?
set -e
if [ $TORCH_RC -ne 0 ]; then
  echo "[setup_env] Stable torch for cp313 not found, trying nightly cu121..."
  python -m pip install --pre --index-url https://download.pytorch.org/whl/nightly/cu121 torch
fi

# 4) Quick torch sanity check.
python - <<'PY'
import sys, torch
print("python:", sys.version.split()[0])
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
PY

# 5) Install ByteTrack editable using existing torch.
PIP_NO_BUILD_ISOLATION=1 python -m pip install -v -e third_party/ByteTrack

# 6) ORT: only GPU variant.
python -m pip uninstall -y onnxruntime || true
python -m pip install onnxruntime-gpu==1.22.0

# 7) onnx-simplifier (onnxsim).
python -m pip install onnxsim==0.4.36

# 8) Final import checks.
python - <<'PY'
import torch, importlib
print("torch ok:", torch.cuda.is_available())
yolox = importlib.import_module("yolox")
print("yolox ok:", hasattr(yolox, "__version__"))
PY

echo "[setup_env] DONE"

