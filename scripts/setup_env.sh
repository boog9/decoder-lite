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

set -euo pipefail

# Create or reuse a virtual environment.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  python3 -m venv .venv
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

# Warn about very new Python versions (e.g., 3.12/3.13) that may conflict with dependencies.
PYV=$(python <<'PY'
import sys
print(f"{sys.version_info.major}.{sys.version_info.minor}")
PY
)
case "$PYV" in
  3.12|3.13*)
    echo "[setup_env] WARNING: Detected Python $PYV. Some packages in YOLOX/ByteTrack may not support it well. Python 3.10–3.11 is recommended."
    ;;
esac

pip install -U pip wheel

if [[ ! -f third_party/ByteTrack/requirements.txt ]]; then
  echo "[setup_env] ERROR: third_party/ByteTrack/requirements.txt not found. Run: make clone"
  exit 1
fi
pip install -r requirements.txt

OS="$(uname -s)"
: "${PIP_PREFER_BINARY:=1}"
export PIP_PREFER_BINARY

install_pytorch_cu124() {
  if python -c "import torch, sys; print(torch.__version__); sys.exit(0)" >/dev/null 2>&1; then
    echo "[setup_env] PyTorch already installed"
  else
    python -m pip install --upgrade pip
    python -m pip install "torch==2.8.*" "torchvision==0.23.*" --index-url https://download.pytorch.org/whl/cu124
  fi
  python - <<'PY'
import torch
assert torch.cuda.is_available()
print('Torch OK', torch.__version__, torch.version.cuda)
PY
}

install_bytetrack_develop() {
  python -m pip install -r third_party/ByteTrack/requirements.txt
  PIP_NO_BUILD_ISOLATION=1 python third_party/ByteTrack/setup.py develop
}

install_gpu_ort() {
  python -m pip install --only-binary=:all: "onnxruntime-gpu==1.22.0"
}

install_cpu_ort() {
  python -m pip install --only-binary=:all: "onnxruntime==1.22.1"
}

post_install_check() {
  python - <<'PY'
import sys, platform
try:
    import onnxruntime as ort
    prov = ort.get_available_providers()
    print("ORT:", ort.__version__, "providers:", prov)
    if platform.system() in ("Linux", "Windows"):
        assert "CUDAExecutionProvider" in prov, f"CUDAExecutionProvider not available: {prov}"
    print("onnxruntime check: OK")
except Exception as e:
    print("onnxruntime check: FAIL:", e, file=sys.stderr)
    sys.exit(1)
PY
}

case "$OS" in
  Darwin)
    # macOS: CPU ORT only
    install_cpu_ort
    ;;
  Linux|MINGW*|MSYS*|CYGWIN*)
    # Linux/Windows: try GPU ORT first
    if ! install_gpu_ort; then
      # Optional fallback to CPU only if explicitly allowed
      if [[ "${ALLOW_CPU_ORT:-}" == "1" ]]; then
        install_cpu_ort
      else
        echo "ERROR: onnxruntime-gpu install failed and ALLOW_CPU_ORT!=1; aborting." >&2
        exit 1
      fi
    fi
    ;;
  *)
    echo "WARN: Unknown OS '$OS'; not installing ORT automatically." >&2
    ;;
esac

# Run post-install verification (fails on Linux/Windows if CUDAExecutionProvider missing)
post_install_check
pip install cython cython_bbox
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install loguru opencv-python-headless numpy

install_pytorch_cu124
install_bytetrack_develop
