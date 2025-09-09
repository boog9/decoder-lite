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

# Upgrade pip to avoid build issues.
python -m pip install --upgrade pip

# Install PyTorch with CUDA 12.4 only if not already available.
if ! python - <<'PY'
import sys
try:
    import torch  # noqa: F401
    sys.exit(0)
except Exception:
    sys.exit(1)
PY
then
  python -m pip install "torch==2.8.*" "torchvision==0.23.*" \
    --index-url https://download.pytorch.org/whl/cu124
fi

# Project and ByteTrack dependencies.
python -m pip install -r requirements.txt
python -m pip install -r third_party/ByteTrack/requirements.txt
PIP_NO_BUILD_ISOLATION=1 python third_party/ByteTrack/setup.py develop
