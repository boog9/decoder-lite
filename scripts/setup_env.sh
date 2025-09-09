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
pip install -r third_party/ByteTrack/requirements.txt
pip install cython cython_bbox
pip install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
pip install loguru opencv-python-headless numpy

pushd third_party/ByteTrack >/dev/null
python setup.py develop
popd >/dev/null
