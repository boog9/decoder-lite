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
#   [decoder-lite] Done.

set -euo pipefail

# 0) Ensure ByteTrack sources are present (not a submodule in this repo).
if [ ! -f third_party/ByteTrack/yolox/__init__.py ]; then
  bash scripts/clone_bytetrack.sh
fi

python -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate

echo "[decoder-lite] Upgrading pip tooling…"
python -m pip install -U pip setuptools wheel

echo "[decoder-lite] Installing build prerequisites…"
python -m pip install -U ninja cython "packaging>=24" pybind11

echo "[decoder-lite] Installing PyTorch (cu121)…"
python -m pip install --extra-index-url https://download.pytorch.org/whl/cu121 'torch>=2.5,<2.7'

echo "[decoder-lite] Installing ByteTrack (editable, no deps, no build isolation)…"
python -m pip install -e third_party/ByteTrack --no-deps --no-build-isolation --config-settings editable_mode=compat

echo "[decoder-lite] Installing ONNX/ORT GPU stack…"
python -m pip install 'onnxruntime-gpu==1.22.0' onnx onnxsim

if [ -f requirements.txt ]; then
  echo "[decoder-lite] Installing project requirements.txt…"
  python -m pip install -U -r requirements.txt
fi

echo "[decoder-lite] Sanity check…"
python - <<'PY'
import sys, importlib.util
import torch
ok_cuda = torch.cuda.is_available()
spec = importlib.util.find_spec("yolox")
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", ok_cuda)
print("yolox origin", getattr(spec, "origin", None))
PY

echo "[decoder-lite] Done."

